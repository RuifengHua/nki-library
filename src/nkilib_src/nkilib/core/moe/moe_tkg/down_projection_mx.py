# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Down projection sub-kernels with LNC sharding support.

Supports multiple LNC sharding strategies (see SUPPORTED_MOE_SHARDING_STRATEGIES in all_expert_mx_utils.py):
- NO_SHARD: No sharding, each NC computes full result independently. Used when LNC=1.
- SHARD_I: Shard on I (intermediate) dimension. Default for most workloads.
- SHARD_T: Shard on T (token) dimension. Useful when T is large.
- TODO: SHARD_E: Shard on E (expert) dimension. When E_L is divisible by 2 and T is large,
  better to shard on E than I because we can support higher TP and get better DMA throughput
  by loading larger packets.

These sub-kernels can be used by any algorithm that requires LNC-sharded down projection,
including all-expert, selective-load, or custom MoE implementations.
"""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode

# Shared MX constants
from ...mlp.mlp_tkg.projection_mx_constants import (
    NUM_QUADRANTS_IN_SBUF,
    SBUF_QUADRANT_SIZE,
    SCALE_P_ELEM_PER_QUADRANT,
)

# Common utils
from ...utils.common_types import ExpertAffinityScaleMode, MoELNCShardingStrategy
from ...utils.interleave_copy import interleave_copy
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ...utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from ...utils.tensor_view import TensorView
from .all_expert_mx_utils import BF16_PER_INT32, SUPPORTED_MOE_SHARDING_STRATEGIES


@nki.jit
def load_broadcast_down_weight_scale_bias(
    weight: nl.ndarray,
    scale: nl.ndarray,
    bias: Optional[nl.ndarray],
    expert_idx: int,
    H: int,
    tile_I: int,
    n_I512_tiles: int,
    tile_offset: int,
    tile_T: int,
    activation_compute_dtype=nl.bfloat16,
    use_PE_bias_broadcast: bool = True,
    sharding_strategy: MoELNCShardingStrategy = MoELNCShardingStrategy.SHARD_I,
    skip_scale_load: bool = False,
) -> tuple[nl.ndarray, nl.ndarray, Optional[nl.ndarray]]:
    """
    Load down projection weight, scale, and bias (optional) for one expert using static DMA.

    When executed with LNC=2, weights and scales are sharded on I dimension. Bias is sharded on H dimension,
    with NC0 loading the first half of H and NC1 loading the second half.

    Args:
        weight (nl.ndarray): [E_L, 128_I, I/512, H], Down projection weight tensor from HBM (4_I packed in x4 dtype).
        scale (nl.ndarray): [E_L, 16_I, I/512, H], Down projection MX scale tensor from HBM (uint8 MX scales).
        bias (Optional[nl.ndarray]): [E_L, H], Optional down projection bias tensor from HBM.
        expert_idx (int): Index of the current expert to load.
        H (int): Hidden dimension size.
        tile_I (int): Tile size for I dimension (typically 128).
        n_I512_tiles (int): Number of I/512 tiles to load (local shard size when LNC=2).
        tile_offset (int): Starting tile offset for NC's tiles (pre-computed for tile-based sharding).
        tile_T (int): Tile size for T dimension (for bias broadcast).
        activation_compute_dtype: Data type for bias buffer (default: nl.bfloat16).
        use_PE_bias_broadcast (bool): If True, use PE (matmul with ones) for bias broadcast; else use DVE
            stream_shuffle_broadcast.
        sharding_strategy (MoELNCShardingStrategy): LNC sharding strategy. Determines bias H-sharding behavior.

    Returns:
        weight_sb (nl.ndarray): [128_I, n_I512_tiles, H], Weight in SBUF (4_I packed in x4 dtype).
        scale_sb (nl.ndarray): [128_I, n_I512_tiles, H], Scales in SBUF (in leading 4P of each SBUF quadrant).
        bias_sb (Optional[nl.ndarray]): [tile_T, H], Broadcasted bias in SBUF (zeros when bias=None, sharded on H
            when LNC=2).

    Notes:
        - tile_offset is pre-computed to ensure alignment with gate_up projection's tile-based I-sharding
        - Based on experiments, static DMA demonstrates better performance
        - Can revert to DGE if HBM out-of-memory (OOM) issues occur
    """

    # Calculate shapes / tiling
    _, n_prgs, prg_id = get_verified_program_sharding_info("down_projection_mx", (0, 1))
    weight_sb_shape = (tile_I, n_I512_tiles, H)
    bias_sb_shape = (tile_T, H)

    # Allocate buffers
    base_weight = TensorView(weight).base_tensor
    weight_sb = nl.ndarray(weight_sb_shape, dtype=base_weight.dtype, buffer=nl.sbuf)
    scale_dtype = nl.uint8 if skip_scale_load else scale.dtype
    scale_sb = nl.ndarray(weight_sb_shape, dtype=scale_dtype, buffer=nl.sbuf)
    bias_sb: Optional[nl.ndarray] = None

    actual_prg_offset = tile_offset
    I_p_in_hbm = base_weight.shape[1]

    # Load weight: index expert, then slice I/512 tiles
    # Shape: [E_L, I_p, I/512, H] -> [I_p, n_I512_tiles, H] -> padded to [128_I, n_I512_tiles, H]
    if I_p_in_hbm < tile_I:
        nisa.memset(dst=weight_sb[...], value=0)
        weight_view = (
            TensorView(base_weight)
            .select(dim=0, index=expert_idx)
            .slice(dim=1, start=actual_prg_offset, end=actual_prg_offset + n_I512_tiles)
        )
        nisa.dma_copy(src=weight_view.get_view(), dst=weight_sb[:I_p_in_hbm, :, :], dge_mode=nisa.dge_mode.none)
    else:
        weight_view = (
            TensorView(base_weight)
            .select(dim=0, index=expert_idx)
            .slice(dim=1, start=actual_prg_offset, end=actual_prg_offset + n_I512_tiles)
        )
        nisa.dma_copy(src=weight_view.get_view(), dst=weight_sb[...], dge_mode=nisa.dge_mode.none)
    weight_sb = weight_sb.view(weight.dtype)

    """
    Load scale: index expert, then slice I/512 tiles.
    Shape: [E_L, I_p//8, I/512, H] -> [I_p//8, n_I512_tiles, H] -> padded to [128_I, n_I512_tiles, H]
    Note: scales have I_p//8 (not 128), need to map to first 4 partitions of each quadrant.
    Scale layout: 16 partitions map to partitions [0-3, 32-35, 64-67, 96-99] in 128-partition buffer.
    """
    if skip_scale_load:
        # STATIC_MX: fill with dummy 127 scales (scale factor 1.0)
        nisa.memset(dst=scale_sb[...], value=127)
    else:
        I_p_scale_in_hbm = scale.shape[1]
        n_quadrants_needed = div_ceil(I_p_scale_in_hbm, SCALE_P_ELEM_PER_QUADRANT)

        if I_p_scale_in_hbm < tile_I // 8:
            for quadrant_idx in nl.affine_range(NUM_QUADRANTS_IN_SBUF):
                nisa.memset(
                    dst=scale_sb[nl.ds(SBUF_QUADRANT_SIZE * quadrant_idx, SCALE_P_ELEM_PER_QUADRANT), :, :], value=0.0
                )

        for quadrant_idx in nl.affine_range(n_quadrants_needed):
            actual_scale_p = min(SCALE_P_ELEM_PER_QUADRANT, I_p_scale_in_hbm - SCALE_P_ELEM_PER_QUADRANT * quadrant_idx)
            if actual_scale_p > 1:
                scale_view = (
                    TensorView(scale)
                    .select(dim=0, index=expert_idx)
                    .slice(
                        dim=0,
                        start=SCALE_P_ELEM_PER_QUADRANT * quadrant_idx,
                        end=SCALE_P_ELEM_PER_QUADRANT * quadrant_idx + actual_scale_p,
                    )
                    .slice(dim=1, start=actual_prg_offset, end=actual_prg_offset + n_I512_tiles)
                )
                nisa.dma_copy(
                    src=scale_view.get_view(),
                    dst=scale_sb[nl.ds(SBUF_QUADRANT_SIZE * quadrant_idx, actual_scale_p), :, :],
                    dge_mode=nisa.dge_mode.none,
                )
            else:
                hbm_p_idx = SCALE_P_ELEM_PER_QUADRANT * quadrant_idx
                sb_p_idx = SBUF_QUADRANT_SIZE * quadrant_idx
                scale_f_per_partition = n_I512_tiles * H
                expert_stride = I_p_scale_in_hbm * scale_f_per_partition
                hbm_offset = expert_idx * expert_stride + hbm_p_idx * scale_f_per_partition + actual_prg_offset * H
                nisa.dma_copy(
                    src=scale.ap(
                        pattern=[[scale_f_per_partition, 1], [1, scale_f_per_partition]],
                        offset=hbm_offset,
                    ),
                    dst=scale_sb.ap(
                        pattern=[[scale_f_per_partition, 1], [1, scale_f_per_partition]],
                        offset=sb_p_idx * scale_f_per_partition,
                    ),
                    dge_mode=nisa.dge_mode.none,
                )

    """
    Load + broadcast bias, sharding on H dim when LNC=2.
    In LNC=2, NC0 bias_sb will have first half of H filled with bias,
    second half with zeros; NC1 will have the inverse.
    Shape: [E_L, H] -> [1, H_size_local]
    """
    if bias != None:
        bias_sb = nl.ndarray(bias_sb_shape, dtype=activation_compute_dtype, buffer=nl.sbuf)
        H_size_local = H if (sharding_strategy == MoELNCShardingStrategy.SHARD_T) else (H // 2 if n_prgs > 1 else H)
        H_offset = 0 if (sharding_strategy == MoELNCShardingStrategy.SHARD_T) else (H_size_local * prg_id)
        if H_size_local < H:
            other_H_offset = H_size_local * (1 - prg_id)
            nisa.memset(dst=bias_sb[:, nl.ds(other_H_offset, H_size_local)], value=0.0, engine=nisa.gpsimd_engine)
        else:
            nisa.memset(dst=bias_sb[...], value=0.0, engine=nisa.gpsimd_engine)
        H_slice_local = nl.ds(H_offset, H_size_local)
        bias_view = (
            TensorView(bias)
            .slice(dim=0, start=expert_idx, end=expert_idx + 1)
            .slice(dim=1, start=H_offset, end=H_offset + H_size_local)
        )
        nisa.dma_copy(src=bias_view.get_view(), dst=bias_sb[0:1, H_slice_local], dge_mode=nisa.dge_mode.none)

        # Broadcast bias using PE
        if use_PE_bias_broadcast:
            is_bias_16bit = activation_compute_dtype in (nl.bfloat16, nl.float16)
            psum_fmax = nl.tile_size.psum_fmax * 2 if is_bias_16bit else nl.tile_size.psum_fmax
            psum_dtype = activation_compute_dtype if is_bias_16bit else nl.float32
            H_tile_size_local = min(H_size_local, psum_fmax)
            ones_mask = nl.ndarray((1, tile_T), dtype=bias_sb.dtype, buffer=nl.sbuf)
            nisa.memset(dst=ones_mask[...], value=1.0, engine=nisa.gpsimd_engine)
            n_H_tiles_local = div_ceil(H_size_local, H_tile_size_local)
            for h_tile_idx in nl.affine_range(n_H_tiles_local):
                h_tile_actual = min(H_tile_size_local, H_size_local - h_tile_idx * H_tile_size_local)
                bias_bc_psum = nl.ndarray((tile_T, h_tile_actual), dtype=psum_dtype, buffer=nl.psum)
                H_tile_slice = nl.ds(H_offset + h_tile_idx * H_tile_size_local, h_tile_actual)
                nisa.nc_matmul(
                    dst=bias_bc_psum,
                    stationary=ones_mask[...],
                    moving=bias_sb[0:1, H_tile_slice],
                    is_stationary_onezero=True,
                )
                nisa.tensor_copy(
                    dst=bias_sb[:, H_tile_slice],
                    src=bias_bc_psum[...],
                    engine=nisa.scalar_engine,
                )

        # Broadcast on DVE
        else:
            stream_shuffle_broadcast(src=bias_sb, dst=bias_sb)

    return weight_sb, scale_sb, bias_sb


@nki.jit
def down_projection_mx(
    act_sb: nl.ndarray,
    act_scale_sb: nl.ndarray,
    weight_sb: nl.ndarray,
    weight_scale_sb: nl.ndarray,
    bias_sb: Optional[nl.ndarray],
    expert_affinities_masked_sb: nl.ndarray,
    expert_idx: int,
    out_sb: nl.ndarray,
    out_hbm: Optional[nl.ndarray] = None,
    token_position_to_id_T: Optional[nl.ndarray] = None,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    activation_compute_dtype=nl.bfloat16,
    is_first_expert: bool = False,
    is_last_expert: bool = False,
    sharding_strategy: MoELNCShardingStrategy = MoELNCShardingStrategy.SHARD_I,
    T_offset: int = 0,
    down_dequant_scale: Optional[nl.ndarray] = None,
    down_input_dequant_scale: Optional[nl.ndarray] = None,
    global_token_indices: Optional[nl.ndarray] = None,
) -> nl.ndarray:
    """
    Computes down projection, expert affinity scaling, expert add, LNC reduction, and SB->HBM spill.
    Supports multiple LNC sharding strategies (see SUPPORTED_MOE_SHARDING_STRATEGIES in all_expert_mx_utils.py).

    Usage:
        Tuned for: mx all-expert MoE algorithm
        Applicable to: any algorithm requiring mx LNC-sharded down projection

    Args:
        act_sb (nl.ndarray): [16_I * 8_I, I/512, T], Activation tensor in SBUF (4_I packed in x4 dtype).
        act_scale_sb (nl.ndarray): [16_I * 8_I, I/512, T], Activation scales in SBUF
            (in leading 4P of each SBUF quadrant).
        weight_sb (nl.ndarray): [16_I * 8_I, I/512, H], Weight tensor in SBUF (4_I packed in x4 dtype).
        weight_scale_sb (nl.ndarray): [16_I * 8_I, I/512, H], Weight scales in SBUF
            (in leading 4P of each SBUF quadrant).
        bias_sb (Optional[nl.ndarray]): [1, H], Optional bias tensor in SBUF.
        expert_affinities_masked_sb (nl.ndarray): [T, E_L] or [128_T, T/128, E_L],
            Expert affinity scores in SBUF.
        expert_idx (int): Index of the current expert.
        out_sb (nl.ndarray): [min(T, 128), ⌈T/128⌉, H], Output tensor in SBUF.
        out_hbm (Optional[nl.ndarray]): [T, H], Optional output tensor in HBM for spill.
        token_position_to_id_T (Optional[nl.ndarray]): [128_T, T/128], Token position indices for indirect
            DMA scatter. When provided, enables blockwise output spill.
        expert_affinities_scaling_mode (ExpertAffinityScaleMode): Scaling mode for expert affinities.
        activation_compute_dtype: Compute dtype for activations (default: bfloat16).
        is_first_expert (bool): Whether the current expert is the first expert.
        is_last_expert (bool): Whether the current expert is the last expert.
        sharding_strategy (MoELNCShardingStrategy): LNC sharding strategy.
            Supported: see SUPPORTED_MOE_SHARDING_STRATEGIES in all_expert_mx_utils.py.
        T_offset (int): Offset for T dimension in HBM output (used with direct DMA).
        down_dequant_scale (Optional[nl.ndarray]): Dequant scale for down projection.
            STATIC_MX: [tile_T, 1] combined (input * weight) scale. ROW_MX: [tile_T, H//_pmax] per-row weight scale.
        down_input_dequant_scale (Optional[nl.ndarray]): [_pmax, T, 1], ROW_MX per-token intermediate dequant scale.
        global_token_indices (Optional[nl.ndarray]): [_pmax, 1], Optional token indices to store in final columns of output.

    Returns:
        out_sb (nl.ndarray): [min(T, 128), ⌈T/128⌉, H], Output tensor in SBUF with accumulated results.
    """

    # Validate sharding strategy is supported (use explicit equality checks for NKI tracing compatibility)
    _is_supported_strategy = (
        (sharding_strategy == MoELNCShardingStrategy.NO_SHARD)
        or (sharding_strategy == MoELNCShardingStrategy.SHARD_I)
        or (sharding_strategy == MoELNCShardingStrategy.SHARD_T)
    )
    kernel_assert(
        _is_supported_strategy,
        f"Unsupported sharding strategy: {sharding_strategy}. Supported: {SUPPORTED_MOE_SHARDING_STRATEGIES}",
    )

    # Extract / validate shapes
    TILE_I, n_I512_tiles, T = act_sb.shape
    TILE_I_, n_I512_tiles_, H = weight_sb.shape
    kernel_assert(
        TILE_I == TILE_I_, f"Expected same number of partitions in activation and weight, got {TILE_I}, {TILE_I_}"
    )
    kernel_assert(
        n_I512_tiles == n_I512_tiles_,
        f"Expected same number of I tiles in activation and weight, got {n_I512_tiles}, {n_I512_tiles_}",
    )
    kernel_assert(H % 512 == 0, f"Expected H divisible by 512, got {H=}")
    kernel_assert(
        expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE,
        f"Expected expert_affinities_scaling_mode={ExpertAffinityScaleMode.POST_SCALE}, "
        f"got: {expert_affinities_scaling_mode=}",
    )
    kernel_assert(
        out_hbm != None,
        f"Output in SBUF is not yet supported, got out_hbm=None",
    )

    # LNC config
    _, n_prgs, prg_id = get_verified_program_sharding_info("down_projection_mx", (0, 1))

    # When not sharding on I, treat as single-program for LNC reduction purposes
    effective_n_prgs = n_prgs if (sharding_strategy == MoELNCShardingStrategy.SHARD_I) else 1

    # Algorithm + tiling strategy
    pmax = nl.tile_size.pmax
    TILE_T = min(T, pmax)  # T will be partition dim in output
    TILE_H = min(H, nl.tile_size.psum_fmax * 2)  # use 2 * fmax with bf16 PSUM
    n_T128_tiles = div_ceil(T, pmax)
    n_H1024_tiles = H // TILE_H
    is_static_mx = down_dequant_scale != None
    is_blockwise = token_position_to_id_T != None
    is_a2av = global_token_indices != None

    # Cast expert affinities to fp32 for tensor_scalar on scalar engine
    # FIXME[perf]: skip TensorCopy and use strided AP in below TensorScalar when affinities are already fp32
    is_3D_affinities = len(expert_affinities_masked_sb.shape) == 3
    expert_affinities_masked_fp32_sb = nl.ndarray((TILE_T, n_T128_tiles), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=expert_affinities_masked_fp32_sb[...],
        src=expert_affinities_masked_sb[:, :, expert_idx]
        if is_3D_affinities
        else expert_affinities_masked_sb[:, expert_idx],
        engine=nisa.scalar_engine,
    )

    # Tiled MM: compute activation_mxfp8 (stationary) @ W_mxfp4/8 (moving)
    for tile_t in nl.sequential_range(n_T128_tiles):
        # T dim slicing + masking for case when T tile < TILE_T
        tile_T_offset = TILE_T * tile_t
        tile_T_actual = min(TILE_T, T - tile_T_offset)
        tile_T_slice = nl.ds(tile_T_offset, tile_T_actual)
        for tile_h in nl.sequential_range(n_H1024_tiles):
            # H dim slicing
            tile_H_offset = TILE_H * tile_h
            weight_H_slice = nl.ds(tile_H_offset, TILE_H)
            out_psum = nl.ndarray((TILE_T, TILE_H), dtype=nl.bfloat16, buffer=nl.psum)
            expert_out_tile_sb = nl.ndarray((TILE_T, TILE_H), dtype=activation_compute_dtype, buffer=nl.sbuf)
            for tile_i in nl.sequential_range(n_I512_tiles):
                nisa.nc_matmul_mx(
                    dst=out_psum[:tile_T_actual, :],
                    stationary=act_sb[:, tile_i, tile_T_slice],
                    moving=weight_sb[:, tile_i, weight_H_slice],
                    stationary_scale=act_scale_sb[:, tile_i, tile_T_slice],
                    moving_scale=weight_scale_sb[:, tile_i, weight_H_slice],
                )

            # Accumulate bias during PSUM eviction (skip for software dequant paths)
            if bias_sb != None and down_dequant_scale == None:
                nisa.tensor_tensor(
                    dst=expert_out_tile_sb[:tile_T_actual, :],
                    data1=out_psum[:tile_T_actual, :],
                    op=nl.add,
                    data2=bias_sb[:tile_T_actual, tile_H_offset : tile_H_offset + TILE_H],
                )
            else:
                nisa.tensor_copy(
                    dst=expert_out_tile_sb[:tile_T_actual, :],
                    src=out_psum[:tile_T_actual, :],
                )

            # Software dequant: STATIC_MX [_pmax, 1] or ROW_MX [tile_T, H//_pmax], then bias
            if is_static_mx:
                if down_dequant_scale.shape[1] == 1:
                    # STATIC_MX: combined scale broadcasts over TILE_H
                    nisa.activation(
                        dst=expert_out_tile_sb[:tile_T_actual, :],
                        op=nl.copy,
                        data=expert_out_tile_sb[:tile_T_actual, :],
                        scale=down_dequant_scale[:tile_T_actual, :],
                    )
                else:
                    # ROW_MX: per-column weight dequant, then per-token input dequant
                    n_H128_in_tile = TILE_H // pmax
                    for i_h128 in nl.affine_range(n_H128_in_tile):
                        h_col = tile_H_offset // pmax + i_h128
                        h_slice = nl.ds(i_h128 * pmax, pmax)
                        interleave_copy(
                            dst=expert_out_tile_sb[:tile_T_actual, h_slice],
                            src=expert_out_tile_sb[:tile_T_actual, h_slice],
                            scale=TensorView(down_dequant_scale[:tile_T_actual, h_col : h_col + 1]),
                            index=i_h128,
                        )
                    if down_input_dequant_scale != None:
                        token_scale_sb = nl.ndarray((TILE_T, 1), dtype=nl.float32, buffer=nl.sbuf)
                        token_scale_psum = nl.ndarray((TILE_T, 1), dtype=nl.float32, buffer=nl.psum)
                        token_scale_1d = down_input_dequant_scale[0:1, tile_T_offset : tile_T_offset + tile_T_actual, 0]
                        nisa.nc_transpose(
                            data=token_scale_1d,
                            dst=token_scale_psum[:tile_T_actual, 0],
                        )
                        nisa.tensor_copy(dst=token_scale_sb[:tile_T_actual, :], src=token_scale_psum[:tile_T_actual, :])
                        nisa.activation(
                            dst=expert_out_tile_sb[:tile_T_actual, :],
                            op=nl.copy,
                            data=expert_out_tile_sb[:tile_T_actual, :],
                            scale=token_scale_sb[:tile_T_actual, :],
                        )
                if bias_sb != None:
                    nisa.tensor_tensor(
                        dst=expert_out_tile_sb[:tile_T_actual, :],
                        data1=expert_out_tile_sb[:tile_T_actual, :],
                        op=nl.add,
                        data2=bias_sb[:tile_T_actual, tile_H_offset : tile_H_offset + TILE_H],
                    )

            # Expert affinity scaling, expert add
            if is_first_expert or is_blockwise:
                nisa.tensor_scalar(
                    dst=out_sb[:tile_T_actual, tile_t : tile_t + 1, tile_H_offset : tile_H_offset + TILE_H],
                    data=expert_out_tile_sb.ap([[TILE_H, tile_T_actual], [1, TILE_H]]),
                    op0=nl.multiply,
                    operand0=expert_affinities_masked_fp32_sb[:tile_T_actual, tile_t : tile_t + 1],
                    engine=nisa.scalar_engine,
                )
            # Expert [1, ..., E] must compute out_sb += expert_out after affinity scaling
            else:
                nisa.tensor_scalar(
                    dst=expert_out_tile_sb[:tile_T_actual, :],
                    data=expert_out_tile_sb.ap([[TILE_H, tile_T_actual], [1, TILE_H]]),
                    op0=nl.multiply,
                    operand0=expert_affinities_masked_fp32_sb[:tile_T_actual, tile_t : tile_t + 1],
                    engine=nisa.scalar_engine,
                )
                # Expert add
                nisa.tensor_tensor(
                    dst=out_sb[:tile_T_actual, tile_t : tile_t + 1, tile_H_offset : tile_H_offset + TILE_H],
                    data1=out_sb[:tile_T_actual, tile_t : tile_t + 1, tile_H_offset : tile_H_offset + TILE_H],
                    op=nl.add,
                    data2=expert_out_tile_sb[:tile_T_actual, :],
                )

        # LNC reduce and SB->HBM spill when computing final expert.
        # TODO: (1) add support for output_in_sbuf=True (2) handle E>1 + K>1 + dynamic all-expert
        if is_last_expert:
            # Sharded LNC reduction + SB->HBM spill when LNC=2 and I is sharded
            if effective_n_prgs > 1:
                H_local = H // n_prgs
                H_offset_local = H_local * prg_id
                send_to_rank = recv_from_rank = 1 - prg_id
                H_local_slice = nl.ds(H_offset_local, H_local)
                H_send_slice = nl.ds(H_local * (1 - prg_id), H_local)
                out_sb_reduced = nl.ndarray((TILE_T, H_local), out_sb.dtype, buffer=nl.sbuf)
                PIPE_ID_OUTPUT = 0

                # NC0 recieves 1st half of output from NC1 and reduces locally on DVE; NC1 does the inverse
                # Perform sendrecv directly on sliced views
                nisa.sendrecv(
                    send_to_rank=send_to_rank,
                    recv_from_rank=recv_from_rank,
                    src=out_sb[:tile_T_actual, tile_t, H_send_slice],
                    dst=out_sb_reduced[:tile_T_actual, :],
                    pipe_id=PIPE_ID_OUTPUT,
                )

                # Reduce
                nisa.tensor_tensor(
                    dst=out_sb_reduced[:tile_T_actual, :],
                    data1=out_sb[:tile_T_actual, tile_t, H_local_slice],
                    op=nl.add,
                    data2=out_sb_reduced[:tile_T_actual, :],
                )

                # Sharded SB->HBM spill
                if is_blockwise:
                    # Indirect DMA to scatter block into HBM tensor
                    nisa.dma_copy(
                        src=out_sb_reduced[:tile_T_actual, :],
                        dst=out_hbm.ap(
                            pattern=[[H_local, tile_T_actual], [1, H_local]],
                            offset=H_offset_local,
                            vector_offset=token_position_to_id_T.ap(
                                pattern=[[n_T128_tiles, tile_T_actual], [1, 1]],
                                offset=tile_t,
                            ),
                            indirect_dim=0,
                        ),
                        # When a token is not routed to a given expert, vector_offset[token] = -1 and we skip DMA
                        oob_mode=oob_mode.skip,
                    )

                    if is_a2av:
                        # Reinterpret global token indices to out_hbm dtype, scatter into final 2 columns of out_hbm [T, H + 2]
                        # TODO[perf]: Combine this with block spill on NC1 to avoid duplicate indirect DMAs
                        nisa.dma_copy(
                            src=global_token_indices.ap(
                                pattern=[[n_T128_tiles * BF16_PER_INT32, tile_T_actual], [1, BF16_PER_INT32]],
                                offset=BF16_PER_INT32 * tile_t,
                                dtype=out_hbm.dtype,
                            ),
                            dst=out_hbm.ap(
                                pattern=[[H + BF16_PER_INT32, tile_T_actual], [1, BF16_PER_INT32]],
                                offset=H,
                                vector_offset=token_position_to_id_T.ap(
                                    pattern=[[n_T128_tiles, tile_T_actual], [1, 1]],
                                    offset=tile_t,
                                ),
                                indirect_dim=0,
                            ),
                            # When a token is not routed to a given expert, vector_offset[token] = -1 and we skip DMA
                            oob_mode=oob_mode.skip,
                        )
                else:
                    # Direct DMA
                    nisa.dma_copy(
                        src=out_sb_reduced[:tile_T_actual, :],
                        dst=out_hbm[nl.ds(T_offset + TILE_T * tile_t, tile_T_actual), H_local_slice],
                        dge_mode=nisa.dge_mode.none,
                    )

            # LNC1, SHARD_T, or redundant compute fallback: each NC spills its H portion
            else:
                H_local = H if (n_prgs == 1 or sharding_strategy == MoELNCShardingStrategy.SHARD_T) else H // n_prgs
                H_offset_local = (
                    0 if (n_prgs == 1 or sharding_strategy == MoELNCShardingStrategy.SHARD_T) else H_local * prg_id
                )
                nisa.dma_copy(
                    src=out_sb[:tile_T_actual, tile_t : tile_t + 1, nl.ds(H_offset_local, H_local)],
                    dst=out_hbm[nl.ds(T_offset + TILE_T * tile_t, tile_T_actual), nl.ds(H_offset_local, H_local)],
                    dge_mode=nisa.dge_mode.none,
                )

    return out_sb
