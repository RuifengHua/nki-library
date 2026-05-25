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

"""MXFP8 SwiGLU MLP backward kernels with activation checkpointing and recompute support."""

import nki.isa as nisa
import nki.language as nl
from nki.dtype import float8_e4m3fn_x4

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil
from ...matmul_mxfp8.matmul_mxfp8_generic_api import generic_matmul_mxfp8_api
from ...mxfp_utils.mxfp8_utils.common_dataclasses import TensorDescriptor
from ...mxfp_utils.mxfp8_utils.common_utils import create_and_set_active_sbm, get_active_sbm
from ...mxfp_utils.mxfp8_utils.quantize_mxfp8_utils import INTERLEAVE_FACTOR
from ..common_utils import (
    DGT_MIN_K,
    MAX_TILES_IN_LOAD_M,
    NUM_LNC2_CORES,
    _allocate_spill_buffer,
    _build_matmul_params,
    _compute_load_tile_shape,
    _store_unswizzled_sbuf_block_to_hbm,
    get_tile_sizes,
)
from .recompute import recompute_gate_act, recompute_gate_up_projection, recompute_hidden


def _transpose_tile_to_scratch(src_hbm, dst_scratch, global_s, i_off, dst_i_offset, I, dtype, tile_m, tile_n):
    """Transpose a tile from HBM and store to scratch in transposed layout.

    Args:
        tile_m: M size of the tile (may be smaller than TILE_M for remainder tiles).
        tile_n: N size of the tile (may be smaller than TILE_N for remainder tiles).
    """
    sbm = get_active_sbm()

    NUM_SUB = div_ceil(tile_n, tile_m)
    for subtile_idx in nl.affine_range(NUM_SUB):
        sub_off = subtile_idx * tile_m
        sub_width = min(tile_m, tile_n - sub_off)
        tile_t = sbm.alloc_stack(shape=(sub_width, tile_m), dtype=dtype, buffer=nl.sbuf)
        src_offset = global_s * I + i_off + sub_off
        nisa.dma_transpose(
            dst=tile_t.ap(pattern=[[sub_width, tile_m], [1, 1], [1, 1], [1, sub_width]], offset=0),
            src=src_hbm.ap(pattern=[[I, tile_m], [1, 1], [1, 1], [1, sub_width]], offset=src_offset),
        )
        nisa.dma_copy(
            dst=dst_scratch[dst_i_offset + sub_off : dst_i_offset + sub_off + sub_width, global_s : global_s + tile_m],
            src=tile_t,
        )


def get_program_sharding_info(run_with_lnc2: bool) -> tuple:
    """Return (num_cores, shard_id) for LNC2 sharding."""
    if run_with_lnc2:
        return NUM_LNC2_CORES, nl.program_id(axis=0)
    return 1, 0


def compute_phase1_down_proj_mm_grad_mxfp8(
    output_grad_td: TensorDescriptor,
    gate_pre_td: TensorDescriptor,
    gate_act_td: TensorDescriptor,
    up_td: TensorDescriptor,
    d_gate_td: TensorDescriptor,
    d_up_td: TensorDescriptor,
    scratch_td: TensorDescriptor,
    down_weight_td: TensorDescriptor,
    s_base: int,
    dtype: type,
    fp8_x4_dtype: type,
    TILES_IN_BLOCK_M: int = 8,
    TILES_IN_BLOCK_N: int = 1,
    TILES_IN_BLOCK_K: int = 8,
    spill_reload: bool = True,
    use_scale_packing: bool = True,
    run_with_lnc2: bool = True,
) -> None:
    """Phase 1: Compute gradient through the down projection and SwiGLU gate.

    Step A — matmul: backprop through down projection
        d_intermediate = output_grad @ W_down_T.T       [S, I]

    Step B — SwiGLU backward: split d_intermediate into gate and up gradients
        Using checkpointed forward activations:
          gate_pre:  [S, I]  gate pre-activation  (hidden @ W_gate.T, before SiLU)
          gate_act:  [S, I]  gate post-activation (SiLU(gate_pre))
          up:        [S, I]  up projection        (hidden @ W_up.T)

        silu_dx       = SiLU'(gate_pre)                  — SiLU derivative
        d_gate_act    = silu_dx * up                      — chain rule through gating
        d_gate        = d_intermediate * d_gate_act       — gradient for gate path
        d_up          = d_intermediate * gate_act         — gradient for up path

    Also transposes d_gate/d_up into scratch for phase 3 weight grad computation.

    Dimensions derived from tensor descriptors:
        output_grad_td.logical_shape = (H, S)  — K=H, F=S
        down_weight_td.logical_shape = (H, I)  — K=H, F=I

    Args:
        output_grad_td (TensorDescriptor): [S, H], incoming gradient (is_f_by_k=True).
        gate_pre_td (TensorDescriptor): [S, I], checkpointed gate pre-activation.
        gate_act_td (TensorDescriptor): [S, I], checkpointed gate post-activation.
        up_td (TensorDescriptor): [S, I], checkpointed up projection.
        d_gate_td (TensorDescriptor): [S, I], output: gate gradient.
        d_up_td (TensorDescriptor): [S, I], output: up gradient.
        scratch_td (TensorDescriptor): [2I, S], output: transposed d_gate || d_up.
        down_weight_td (TensorDescriptor): [I, H], transposed down projection weights (is_f_by_k=True).
        s_base (int): Row offset into the full [S, ...] tensors for this LNC core.
        dtype: Data type for computation (nl.bfloat16).
        fp8_x4_dtype: MXFP8 quantized data type (e.g. float8_e4m3fn_x4).
        TILES_IN_BLOCK_M (int): Number of M tiles per block.
        TILES_IN_BLOCK_N (int): Number of N tiles per block.
        TILES_IN_BLOCK_K (int): Number of K tiles to accumulate in PSUM.

    Returns:
        None. Results are written to d_gate_td.data, d_up_td.data, and scratch_td.data.

    Pseudocode:
        for each m_block in S tiles:
            for each n_block in I tiles:
                acc = zeros()
                for each k_block in H tiles:
                    load output_grad, down_weight via DGT + quantize
                    acc += output_grad @ down_weight
                # SwiGLU backward
                silu_dx = SiLU'(gate_pre)
                d_gate = acc * (silu_dx * up)
                d_up = acc * gate_act
                store d_gate, d_up to HBM
                transpose d_gate, d_up into scratch
    """
    sbm = get_active_sbm()

    # Phase 1: output_grad[S,H] @ down_weight[I,H].T -> [S,I]
    # K=H, M=S_local, N=I
    H = output_grad_td.logical_shape[0]
    I = down_weight_td.logical_shape[1]
    S_local = output_grad_td.sharded_logical_shape[1]

    tiles = get_tile_sizes(H, S_local, I)
    tile_m = tiles['tile_m']
    tile_n = tiles['tile_n']
    l_tile_k = tiles['l_tile_k']

    NUM_S_TILES_LOCAL = div_ceil(S_local, tile_m)
    BLOCK_N = TILES_IN_BLOCK_N * tile_n
    NUM_K_TILES = div_ceil(H, l_tile_k)
    NUM_I_TILES = div_ceil(I, tile_n)
    NUM_M_BLOCKS = div_ceil(NUM_S_TILES_LOCAL, TILES_IN_BLOCK_M)
    NUM_N_BLOCKS = div_ceil(NUM_I_TILES, TILES_IN_BLOCK_N)
    NUM_K_BLOCKS = div_ceil(NUM_K_TILES, TILES_IN_BLOCK_K)

    # Compute load tile shapes from TD state
    lhs_load_tile_shape = _compute_load_tile_shape(output_grad_td, tiles, tile_m)
    rhs_load_tile_shape = _compute_load_tile_shape(down_weight_td, tiles, tile_n)

    # Convert s_base to physical offset for pre-swizzled inputs
    s_base_physical = (
        s_base * INTERLEAVE_FACTOR if (output_grad_td.is_swizzled and not output_grad_td.is_quantized) else s_base
    )

    bd = _build_matmul_params(
        TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K,
        lhs_load_tile_shape=lhs_load_tile_shape,
        rhs_load_tile_shape=rhs_load_tile_shape,
        tiles=tiles,
    )

    # Spill/reload buffers (skip for pre-quantized inputs)
    output_gradq_td = None
    down_weightq_td = None
    if spill_reload:
        data_buffer = nl.private_hbm if run_with_lnc2 else nl.hbm

        if not output_grad_td.is_quantized:
            output_gradq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_M_BLOCKS,
                block_f_logical=bd.BLOCK_M_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=use_scale_packing,
                data_buffer=data_buffer,
            )
        if not down_weight_td.is_quantized:
            down_weightq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_N_BLOCKS,
                block_f_logical=bd.BLOCK_N_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=use_scale_packing,
                data_buffer=data_buffer,
            )

    sbuf_step_p = TILES_IN_BLOCK_M * BLOCK_N

    for m_block_idx in nl.sequential_range(NUM_M_BLOCKS):
        m_block_start = m_block_idx * TILES_IN_BLOCK_M

        for n_block_idx in range(NUM_N_BLOCKS):
            n_block_start = n_block_idx * TILES_IN_BLOCK_N

            # SBUF accumulator for matmul result — same layout as generic API produces
            acc_sbuf = sbm.alloc_stack(shape=(tile_m, TILES_IN_BLOCK_M * BLOCK_N), dtype=nl.float32, buffer=nl.sbuf)
            acc_td = TensorDescriptor(data=acc_sbuf)

            for k_block_idx in nl.sequential_range(NUM_K_BLOCKS):
                generic_matmul_mxfp8_api(
                    lhs_hbm_td=output_grad_td,
                    rhs_hbm_td=down_weight_td,
                    bd=bd,
                    output_td=acc_td,
                    block_idx_m=(m_block_idx, m_block_idx + 1),
                    block_idx_n=(n_block_idx, n_block_idx + 1),
                    block_idx_k=(k_block_idx, k_block_idx + 1),
                    lhs_m_offset=s_base_physical,
                    TILES_IN_LOAD_M=min(TILES_IN_BLOCK_M, MAX_TILES_IN_LOAD_M),
                    TILES_IN_LOAD_N=1,
                    lhs_matmul_tile_shape_physical=tiles['lhs_matmul_tile_physical'],
                    rhs_matmul_tile_shape_physical=tiles['rhs_matmul_tile_physical'],
                    lhs_load_tile_shape=lhs_load_tile_shape,
                    rhs_load_tile_shape=rhs_load_tile_shape,
                    lhs_quantize_tile_shape=tiles['lhs_quantize_tile'],
                    rhs_quantize_tile_shape=tiles['rhs_quantize_tile'],
                    initialize_accumulator=(k_block_idx == 0),
                    spill_reload=spill_reload,
                    lhsq_td=output_gradq_td,
                    rhsq_td=down_weightq_td,
                    use_scale_packing=use_scale_packing,
                )

            # SwiGLU backward: read matmul result from SBUF via BIR AP
            num_m_tiles_in_block = min(TILES_IN_BLOCK_M, div_ceil(S_local - m_block_start * tile_m, tile_m))
            num_n_tiles_in_block = min(TILES_IN_BLOCK_N, div_ceil(I - n_block_start * tile_n, tile_n))
            for tile_m_idx in nl.affine_range(num_m_tiles_in_block):
                m_off = m_block_start * tile_m + tile_m_idx * tile_m
                global_s = s_base + m_off
                actual_m = min(tile_m, S_local - m_off)

                silu_dx_up_tiles = []
                gate_act_checkpoint_tiles = []
                for tile_n_idx in nl.affine_range(num_n_tiles_in_block):
                    i_off = (n_block_start + tile_n_idx) * tile_n
                    actual_n = min(tile_n, I - i_off)

                    up_checkpoint = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=dtype, buffer=nl.sbuf)
                    gate_act_checkpoint = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=dtype, buffer=nl.sbuf)
                    gate_pre_checkpoint = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=up_checkpoint, src=up_td.data[global_s : global_s + actual_m, i_off : i_off + actual_n]
                    )
                    nisa.dma_copy(
                        dst=gate_act_checkpoint,
                        src=gate_act_td.data[global_s : global_s + actual_m, i_off : i_off + actual_n],
                    )
                    nisa.dma_copy(
                        dst=gate_pre_checkpoint,
                        src=gate_pre_td.data[global_s : global_s + actual_m, i_off : i_off + actual_n],
                    )

                    silu_dx = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=dtype, buffer=nl.sbuf)
                    nisa.activation(dst=silu_dx, op=nl.silu_dx, data=gate_pre_checkpoint, bias=None, scale=1.0)

                    d_gate_act = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor(
                        dst=d_gate_act, data1=silu_dx, data2=up_checkpoint, op=nl.multiply, engine=nisa.vector_engine
                    )

                    silu_dx_up_tiles.append(d_gate_act)
                    gate_act_checkpoint_tiles.append(gate_act_checkpoint)

                for tile_n_idx in nl.affine_range(num_n_tiles_in_block):
                    i_off = (n_block_start + tile_n_idx) * tile_n
                    actual_n = min(tile_n, I - i_off)
                    sbuf_offset = tile_m_idx * BLOCK_N + tile_n_idx * tile_n

                    acc_tile = acc_sbuf.ap(pattern=[[sbuf_step_p, actual_m], [1, actual_n]], offset=sbuf_offset)

                    d_gate = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=d_gate, data1=acc_tile, data2=silu_dx_up_tiles[tile_n_idx], op=nl.multiply)

                    d_up = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor(
                        dst=d_up, data1=acc_tile, data2=gate_act_checkpoint_tiles[tile_n_idx], op=nl.multiply
                    )

                    nisa.dma_copy(
                        dst=d_gate_td.data[global_s : global_s + actual_m, i_off : i_off + actual_n], src=d_gate
                    )
                    nisa.dma_copy(
                        dst=d_up_td.data[global_s : global_s + actual_m, i_off : i_off + actual_n],
                        src=d_up,
                    )

                    _transpose_tile_to_scratch(
                        d_gate_td.data,
                        scratch_td.data,
                        global_s,
                        i_off,
                        i_off,
                        I,
                        dtype,
                        tile_m=actual_m,
                        tile_n=actual_n,
                    )
                    _transpose_tile_to_scratch(
                        d_up_td.data,
                        scratch_td.data,
                        global_s,
                        i_off,
                        I + i_off,
                        I,
                        dtype,
                        tile_m=actual_m,
                        tile_n=actual_n,
                    )


def compute_phase2_hidden_states_grad_mxfp8(
    hidden_states_grad_td: TensorDescriptor,
    gate_weight_td: TensorDescriptor,
    up_weight_td: TensorDescriptor,
    d_gate_td: TensorDescriptor,
    d_up_td: TensorDescriptor,
    s_base: int,
    dtype: type,
    fp8_x4_dtype: type,
    TILES_IN_BLOCK_M: int = 8,
    TILES_IN_BLOCK_N: int = 1,
    TILES_IN_BLOCK_K: int = 8,
    spill_reload: bool = True,
    use_scale_packing: bool = True,
    run_with_lnc2: bool = True,
) -> None:
    """Phase 2: Compute gradient w.r.t. input hidden states.

    Computes:
        hidden_states_grad[s_base:s_base+S_local, :] = d_gate @ W_gate + d_up @ W_up

    Uses lhs_m_offset for DGT loads (LNC2 shard offset) and relative block_idx_m
    starting at 0. The store adds s_base explicitly to write to the correct shard.

    Dimensions derived from tensor descriptors:
        gate_weight_td.logical_shape = (I, H)  — K=I, F=H
        d_gate_td.sharded_logical_shape[1] = S_local

    Args:
        hidden_states_grad_td (TensorDescriptor): [S, H], output: dL/d_hidden.
        gate_weight_td (TensorDescriptor): [H, I], transposed gate projection weights (is_f_by_k=True).
        up_weight_td (TensorDescriptor): [H, I], transposed up projection weights (is_f_by_k=True).
        d_gate_td (TensorDescriptor): [S, I], gate gradient (is_f_by_k=True).
        d_up_td (TensorDescriptor): [S, I], up gradient (is_f_by_k=True).
        s_base (int): Row offset for this LNC core's shard.
        dtype: Data type for computation (nl.bfloat16).
        fp8_x4_dtype: MXFP8 quantized data type (e.g. float8_e4m3fn_x4).
        TILES_IN_BLOCK_M (int): Number of M tiles per block.
        TILES_IN_BLOCK_N (int): Number of N tiles per block.
        TILES_IN_BLOCK_K (int): Number of K tiles to accumulate in PSUM.

    Returns:
        None. Results are written to hidden_states_grad_td.data[s_base:s_base+S_local, :].
    """
    sbm = get_active_sbm()

    # Phase 2: d_gate[S,I] @ gate_weight[H,I].T -> [S,H]
    # K=I, M=S_local, N=H
    I = gate_weight_td.logical_shape[0]
    H = gate_weight_td.logical_shape[1]
    S_local = d_gate_td.sharded_logical_shape[1]

    tiles = get_tile_sizes(I, S_local, H)
    tile_m = tiles['tile_m']
    tile_n = tiles['tile_n']
    l_tile_k = tiles['l_tile_k']

    NUM_S_TILES_LOCAL = div_ceil(S_local, tile_m)

    BLOCK_M = TILES_IN_BLOCK_M * tile_m
    BLOCK_N = TILES_IN_BLOCK_N * tile_n
    NUM_K_TILES_GATE = div_ceil(I, l_tile_k)
    NUM_H_TILES = div_ceil(H, tile_n)
    NUM_M_BLOCKS = div_ceil(NUM_S_TILES_LOCAL, TILES_IN_BLOCK_M)
    NUM_N_BLOCKS = div_ceil(NUM_H_TILES, TILES_IN_BLOCK_N)

    # RHS load tile shapes from TD state (LHS is always internal BF16 scratch — uses defaults)
    rhs_gate_load_tile_shape = _compute_load_tile_shape(gate_weight_td, tiles, tile_n)
    rhs_up_load_tile_shape = _compute_load_tile_shape(up_weight_td, tiles, tile_n)

    bd = _build_matmul_params(
        TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K, rhs_load_tile_shape=rhs_gate_load_tile_shape, tiles=tiles
    )

    # Spill/reload buffer allocation (LHS always BF16; skip RHS for pre-quantized)
    d_gateq_td = None
    d_upq_td = None
    gate_weightq_td = None
    up_weightq_td = None
    if spill_reload:
        data_buffer = nl.private_hbm if run_with_lnc2 else nl.hbm
        NUM_K_BLOCKS = div_ceil(NUM_K_TILES_GATE, TILES_IN_BLOCK_K)

        d_gateq_td = _allocate_spill_buffer(
            num_k_blocks=NUM_K_BLOCKS,
            num_f_blocks=NUM_M_BLOCKS,
            block_f_logical=bd.BLOCK_M_LOGICAL,
            tiles_in_block_k=TILES_IN_BLOCK_K,
            use_scale_packing=use_scale_packing,
            data_buffer=data_buffer,
        )
        d_upq_td = _allocate_spill_buffer(
            num_k_blocks=NUM_K_BLOCKS,
            num_f_blocks=NUM_M_BLOCKS,
            block_f_logical=bd.BLOCK_M_LOGICAL,
            tiles_in_block_k=TILES_IN_BLOCK_K,
            use_scale_packing=use_scale_packing,
            data_buffer=data_buffer,
        )
        if not gate_weight_td.is_quantized:
            gate_weightq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_N_BLOCKS,
                block_f_logical=bd.BLOCK_N_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=use_scale_packing,
                data_buffer=data_buffer,
            )
        if not up_weight_td.is_quantized:
            up_weightq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_N_BLOCKS,
                block_f_logical=bd.BLOCK_N_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=use_scale_packing,
                data_buffer=data_buffer,
            )

    LHS_MATMUL_TILE_M = tiles['lhs_matmul_tile_physical'][1]

    # Fuse gate + up matmul results in SBUF; block_idx_m is relative, lhs_m_offset handles LNC2 load offset.
    # Store uses s_base to write to the correct shard of hidden_states_grad.
    for m_block_idx in range(NUM_M_BLOCKS):
        for n_block_idx in range(NUM_N_BLOCKS):
            output_sbuf = sbm.alloc_stack(shape=(tile_m, TILES_IN_BLOCK_M, BLOCK_N), dtype=nl.float32, buffer=nl.sbuf)
            output_sbuf_td = TensorDescriptor(data=output_sbuf)

            # Gate path: d_gate @ W_gate → accumulator (zeroed then accumulated)
            generic_matmul_mxfp8_api(
                lhs_hbm_td=d_gate_td,
                rhs_hbm_td=gate_weight_td,
                bd=bd,
                output_td=output_sbuf_td,
                block_idx_m=(m_block_idx, m_block_idx + 1),
                block_idx_n=(n_block_idx, n_block_idx + 1),
                lhs_m_offset=s_base,
                TILES_IN_LOAD_M=min(TILES_IN_BLOCK_M, MAX_TILES_IN_LOAD_M),
                TILES_IN_LOAD_N=1,
                lhs_matmul_tile_shape_physical=tiles['lhs_matmul_tile_physical'],
                rhs_matmul_tile_shape_physical=tiles['rhs_matmul_tile_physical'],
                lhs_load_tile_shape=tiles['lhs_load_tile'],
                rhs_load_tile_shape=rhs_gate_load_tile_shape,
                lhs_quantize_tile_shape=tiles['lhs_quantize_tile'],
                rhs_quantize_tile_shape=tiles['rhs_quantize_tile'],
                spill_reload=spill_reload,
                lhsq_td=d_gateq_td,
                rhsq_td=gate_weightq_td,
                use_scale_packing=use_scale_packing,
            )

            # Up path: d_up @ W_up → accumulate into same SBUF (no zero-init)
            generic_matmul_mxfp8_api(
                lhs_hbm_td=d_up_td,
                rhs_hbm_td=up_weight_td,
                bd=bd,
                output_td=output_sbuf_td,
                block_idx_m=(m_block_idx, m_block_idx + 1),
                block_idx_n=(n_block_idx, n_block_idx + 1),
                lhs_m_offset=s_base,
                TILES_IN_LOAD_M=min(TILES_IN_BLOCK_M, MAX_TILES_IN_LOAD_M),
                TILES_IN_LOAD_N=1,
                lhs_matmul_tile_shape_physical=tiles['lhs_matmul_tile_physical'],
                rhs_matmul_tile_shape_physical=tiles['rhs_matmul_tile_physical'],
                lhs_load_tile_shape=tiles['lhs_load_tile'],
                rhs_load_tile_shape=rhs_up_load_tile_shape,
                lhs_quantize_tile_shape=tiles['lhs_quantize_tile'],
                rhs_quantize_tile_shape=tiles['rhs_quantize_tile'],
                spill_reload=spill_reload,
                lhsq_td=d_upq_td,
                rhsq_td=up_weightq_td,
                use_scale_packing=use_scale_packing,
                initialize_accumulator=False,
            )

            # Store fused result to HBM — s_base offsets into the correct shard
            _store_unswizzled_sbuf_block_to_hbm(
                output_sbuf=output_sbuf,
                dst_hbm=hidden_states_grad_td.data,
                row_base=s_base,
                col_base=n_block_idx * BLOCK_N,
                tiles_in_block_m=TILES_IN_BLOCK_M,
                block_m=BLOCK_M,
                block_n=BLOCK_N,
                lhs_matmul_tile_m=LHS_MATMUL_TILE_M,
                block_idx_m=m_block_idx,
                m_logical=S_local,
                n_logical=H,
            )


def compute_phase3_gate_up_weight_grad_mxfp8(
    weight_grad_td: TensorDescriptor,
    hidden_states_T_td: TensorDescriptor,
    grad_T_td: TensorDescriptor,
    dtype: type,
    fp8_x4_dtype: type,
    TILES_IN_BLOCK_M: int = 4,
    TILES_IN_BLOCK_N: int = 1,
    TILES_IN_BLOCK_K: int = 8,
    spill_reload: bool = True,
    use_scale_packing: bool = True,
    run_with_lnc2: bool = True,
) -> None:
    """Phase 3: Compute gradient w.r.t. gate and up weight matrices as a single matmul.

    Computes:
        [dW_gate; dW_up] = grad_T[2I, S] @ hidden_states[S, H] -> [2I, H]

    Uses pre-transposed inputs:
        grad_T_td: [2I, S]  transposed [d_gate || d_up] from phase 1
            rows [0:I]  = d_gate.T
            rows [I:2I] = d_up.T
        hidden_states_T_td:  [H, S]   transposed input hidden states

    LNC2 sharding: core 0 computes rows [0:I] (gate grad),
                   core 1 computes rows [I:2I] (up grad).

    Dimensions:
        S: Sequence length.
        H: Hidden dimension size.
        I: Intermediate dimension size.

    Args:
        weight_grad_td (TensorDescriptor): [2I, H], output: [dW_gate; dW_up].
        hidden_states_T_td (TensorDescriptor): [H, S], transposed input hidden states (is_f_by_k=True).
        grad_T_td (TensorDescriptor): [2I, S], transposed gate+up gradients
            (is_f_by_k=True, is_col_parallel_sharded=True for LNC2).
        dtype: Data type for computation (nl.bfloat16).
        fp8_x4_dtype: MXFP8 quantized data type.
        TILES_IN_BLOCK_M (int): Number of M tiles per block.
        TILES_IN_BLOCK_N (int): Number of N tiles per block.
        TILES_IN_BLOCK_K (int): Number of K tiles to accumulate in PSUM.

    Returns:
        None. Results are written to weight_grad_td.data.

    Pseudocode:
        # LNC2: core 0 handles rows [0:I], core 1 handles rows [I:2I]
        weight_grad_local = weight_grad[i_base : i_base + I_local, :]
        weight_grad_local = grad_T[i_base : i_base + I_local, :] @ hidden_states_T.T
    """
    # Derive dimensions and LNC shard offset
    S = hidden_states_T_td.logical_shape[0]
    H = hidden_states_T_td.logical_shape[1]
    I = grad_T_td.logical_shape[1] // 2
    _, shard_id = get_program_sharding_info(run_with_lnc2)
    i_base = shard_id * I if run_with_lnc2 else 0

    # M_LOCAL = I per core (2I / 2), or 2I without LNC2
    I_local = I if run_with_lnc2 else 2 * I

    tiles = get_tile_sizes(S, I_local, H)
    tile_m = tiles['tile_m']
    tile_n = tiles['tile_n']
    l_tile_k = tiles['l_tile_k']

    NUM_I_TILES_LOCAL = div_ceil(I_local, tile_m)
    NUM_K_TILES = div_ceil(S, l_tile_k)
    NUM_H_TILES = div_ceil(H, tile_n)
    NUM_M_BLOCKS = div_ceil(NUM_I_TILES_LOCAL, TILES_IN_BLOCK_M)
    NUM_N_BLOCKS = div_ceil(NUM_H_TILES, TILES_IN_BLOCK_N)
    NUM_K_BLOCKS = div_ceil(NUM_K_TILES, TILES_IN_BLOCK_K)

    # RHS load tile shape from TD state
    rhs_load_tile_shape = _compute_load_tile_shape(hidden_states_T_td, tiles, tile_n)
    lhs_load_tile_shape = _compute_load_tile_shape(grad_T_td, tiles, tile_m)

    bd = _build_matmul_params(
        TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K,
        lhs_load_tile_shape=lhs_load_tile_shape,
        rhs_load_tile_shape=rhs_load_tile_shape,
        tiles=tiles,
    )

    # Spill/reload buffers — single LHS buffer, single RHS buffer
    grad_Tq_td = None
    hidden_states_Tq_td = None
    if spill_reload:
        data_buffer = nl.private_hbm if run_with_lnc2 else nl.hbm

        grad_Tq_td = _allocate_spill_buffer(
            num_k_blocks=NUM_K_BLOCKS,
            num_f_blocks=NUM_M_BLOCKS,
            block_f_logical=bd.BLOCK_M_LOGICAL,
            tiles_in_block_k=TILES_IN_BLOCK_K,
            use_scale_packing=use_scale_packing,
            data_buffer=data_buffer,
        )
        if not hidden_states_T_td.is_quantized:
            hidden_states_Tq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_N_BLOCKS,
                block_f_logical=bd.BLOCK_N_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=use_scale_packing,
                data_buffer=data_buffer,
            )

    # Slice output for this core's shard and let the generic API handle all loops + HBM store
    weight_grad_local = weight_grad_td.data[i_base : i_base + I_local, :]
    output_td = TensorDescriptor(data=weight_grad_local)

    generic_matmul_mxfp8_api(
        lhs_hbm_td=grad_T_td,
        rhs_hbm_td=hidden_states_T_td,
        bd=bd,
        output_td=output_td,
        lhs_m_offset=i_base,
        TILES_IN_LOAD_M=min(TILES_IN_BLOCK_M, MAX_TILES_IN_LOAD_M),
        TILES_IN_LOAD_N=1,
        lhs_matmul_tile_shape_physical=tiles['lhs_matmul_tile_physical'],
        rhs_matmul_tile_shape_physical=tiles['rhs_matmul_tile_physical'],
        lhs_load_tile_shape=tiles['lhs_load_tile'],
        rhs_load_tile_shape=rhs_load_tile_shape,
        lhs_quantize_tile_shape=tiles['lhs_quantize_tile'],
        rhs_quantize_tile_shape=tiles['rhs_quantize_tile'],
        spill_reload=spill_reload,
        lhsq_td=grad_Tq_td,
        rhsq_td=hidden_states_Tq_td,
        use_scale_packing=use_scale_packing,
    )


def compute_phase4_down_weight_grad_mxfp8(
    down_weight_grad_td: TensorDescriptor,
    output_grad_T_td: TensorDescriptor,
    hidden_T_td: TensorDescriptor,
    h_base: int,
    dtype: type,
    fp8_x4_dtype: type,
    TILES_IN_BLOCK_M: int = 4,
    TILES_IN_BLOCK_N: int = 1,
    TILES_IN_BLOCK_K: int = 8,
    spill_reload: bool = True,
    use_scale_packing: bool = True,
    run_with_lnc2: bool = True,
) -> None:
    """Phase 4: Compute gradient w.r.t. down projection weight matrix.

    Computes:
        dW_down = output_grad.T @ intermediate     [H, I]

    Uses pre-transposed inputs:
        output_grad_T_td: [H, S]  transposed incoming gradient
        hidden_T_td:      [I, S]  transposed gated intermediate activations

    H-sharded across LNC cores (each core computes a slice of H rows).

    Dimensions derived from tensor descriptors:
        output_grad_T_td.logical_shape = (S, H)  — K=S, F=H
        hidden_T_td.logical_shape = (S, I)  — K=S, F=I
        H_local from output_grad_T_td.sharded_logical_shape[1]

    Args:
        down_weight_grad_td (TensorDescriptor): [H, I], output: dW_down.
        output_grad_T_td (TensorDescriptor): [H, S], transposed output gradient (is_f_by_k=True).
        hidden_T_td (TensorDescriptor): [I, S], transposed intermediate activations (is_f_by_k=True).
        h_base (int): Row offset into the H dimension for this LNC core.
        dtype: Data type for computation (nl.bfloat16).
        fp8_x4_dtype: MXFP8 quantized data type.
        TILES_IN_BLOCK_M (int): Number of M tiles per block.
        TILES_IN_BLOCK_N (int): Number of N tiles per block.
        TILES_IN_BLOCK_K (int): Number of K tiles to accumulate in PSUM.

    Returns:
        None. Results are written to down_weight_grad_td.data.

    Pseudocode:
        for each m_block in H tiles (sharded):
            for each n_block in I tiles:
                acc = zeros()
                for each k_block in S tiles:
                    load output_grad.T, hidden.T via DGT + quantize
                    acc += output_grad.T @ hidden.T
                store acc to down_proj_weight_grad
    """
    sbm = get_active_sbm()

    # Phase 4: output_grad.T[H,S] @ hidden.T[I,S].T -> [H,I]
    # K=S, M=H_local, N=I
    S = output_grad_T_td.logical_shape[0]
    H = output_grad_T_td.logical_shape[1]
    I = hidden_T_td.logical_shape[1]
    H_local = H // NUM_LNC2_CORES if run_with_lnc2 else H

    tiles = get_tile_sizes(S, H_local, I)
    tile_m = tiles['tile_m']
    tile_n = tiles['tile_n']
    l_tile_k = tiles['l_tile_k']

    NUM_H_TILES_LOCAL = div_ceil(H_local, tile_m)
    NUM_K_TILES = div_ceil(S, l_tile_k)
    NUM_I_TILES = div_ceil(I, tile_n)
    NUM_M_BLOCKS = div_ceil(NUM_H_TILES_LOCAL, TILES_IN_BLOCK_M)

    BLOCK_M = TILES_IN_BLOCK_M * tile_m
    BLOCK_N = TILES_IN_BLOCK_N * tile_n
    NUM_N_BLOCKS = div_ceil(NUM_I_TILES, TILES_IN_BLOCK_N)
    LHS_MATMUL_TILE_M = tiles['lhs_matmul_tile_physical'][1]

    # Compute load tile shapes from TD state (supports unswizzled BF16, pre-swizzled, pre-quantized)
    lhs_load_tile_shape = _compute_load_tile_shape(output_grad_T_td, tiles, tile_m)
    rhs_load_tile_shape = _compute_load_tile_shape(hidden_T_td, tiles, tile_n)

    # Convert h_base to physical offset for pre-swizzled inputs
    h_base_physical = (
        h_base * INTERLEAVE_FACTOR if (output_grad_T_td.is_swizzled and not output_grad_T_td.is_quantized) else h_base
    )

    bd = _build_matmul_params(
        TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K,
        lhs_load_tile_shape=lhs_load_tile_shape,
        rhs_load_tile_shape=rhs_load_tile_shape,
        tiles=tiles,
    )

    # Spill/reload buffer allocation (skip for pre-quantized inputs)
    output_grad_Tq_td = None
    hidden_Tq_td = None
    if spill_reload:
        data_buffer = nl.private_hbm if run_with_lnc2 else nl.hbm
        NUM_K_BLOCKS = div_ceil(NUM_K_TILES, TILES_IN_BLOCK_K)

        if not output_grad_T_td.is_quantized:
            output_grad_Tq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_M_BLOCKS,
                block_f_logical=bd.BLOCK_M_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=use_scale_packing,
                data_buffer=data_buffer,
            )
        if not hidden_T_td.is_quantized:
            hidden_Tq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_N_BLOCKS,
                block_f_logical=bd.BLOCK_N_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=use_scale_packing,
                data_buffer=data_buffer,
            )

    # Caller owns M and N loops: SBUF output with lhs_m_offset for loads,
    # h_base offset in store address
    for m_block_idx in range(NUM_M_BLOCKS):
        for n_block_idx in range(NUM_N_BLOCKS):
            output_sbuf = sbm.alloc_stack(shape=(tile_m, TILES_IN_BLOCK_M, BLOCK_N), dtype=nl.float32, buffer=nl.sbuf)
            output_sbuf_td = TensorDescriptor(data=output_sbuf)

            generic_matmul_mxfp8_api(
                lhs_hbm_td=output_grad_T_td,
                rhs_hbm_td=hidden_T_td,
                bd=bd,
                output_td=output_sbuf_td,
                block_idx_m=(m_block_idx, m_block_idx + 1),
                block_idx_n=(n_block_idx, n_block_idx + 1),
                lhs_m_offset=h_base_physical,
                TILES_IN_LOAD_M=min(TILES_IN_BLOCK_M, MAX_TILES_IN_LOAD_M),
                TILES_IN_LOAD_N=1,
                lhs_matmul_tile_shape_physical=tiles['lhs_matmul_tile_physical'],
                rhs_matmul_tile_shape_physical=tiles['rhs_matmul_tile_physical'],
                lhs_load_tile_shape=lhs_load_tile_shape,
                rhs_load_tile_shape=rhs_load_tile_shape,
                lhs_quantize_tile_shape=tiles['lhs_quantize_tile'],
                rhs_quantize_tile_shape=tiles['rhs_quantize_tile'],
                spill_reload=spill_reload,
                lhsq_td=output_grad_Tq_td,
                rhsq_td=hidden_Tq_td,
                use_scale_packing=use_scale_packing,
            )

            # Store result to HBM with h_base offset
            _store_unswizzled_sbuf_block_to_hbm(
                output_sbuf=output_sbuf,
                dst_hbm=down_weight_grad_td.data,
                row_base=h_base,
                col_base=n_block_idx * BLOCK_N,
                tiles_in_block_m=TILES_IN_BLOCK_M,
                block_m=BLOCK_M,
                block_n=BLOCK_N,
                lhs_matmul_tile_m=LHS_MATMUL_TILE_M,
                block_idx_m=m_block_idx,
                m_logical=H_local,
                n_logical=I,
            )


def mlp_backward_mxfp8_base_nki(
    output_grad_td: TensorDescriptor,
    hidden_states_td: TensorDescriptor,
    gate_pre_td: TensorDescriptor,
    gate_act_td: TensorDescriptor,
    up_td: TensorDescriptor,
    hidden_td: TensorDescriptor,
    gate_weight_T_td: TensorDescriptor,
    up_weight_T_td: TensorDescriptor,
    down_weight_T_td: TensorDescriptor,
    d_gate_td: TensorDescriptor,
    d_up_td: TensorDescriptor,
    hidden_states_T_td: TensorDescriptor,
    output_grad_T_td: TensorDescriptor,
    hidden_T_td: TensorDescriptor,
    scratch_td: TensorDescriptor,
    hidden_states_grad_td: TensorDescriptor,
    weight_grad_td: TensorDescriptor,
    down_weight_grad_td: TensorDescriptor,
    run_with_lnc2: bool = True,
    phase1_tiles_m: int = 8,
    phase1_tiles_n: int = 1,
    phase1_tiles_k: int = 8,
    phase2_tiles_m: int = 8,
    phase2_tiles_n: int = 1,
    phase2_tiles_k: int = 8,
    phase3_tiles_m: int = 4,
    phase3_tiles_n: int = 1,
    phase3_tiles_k: int = 8,
    phase4_tiles_m: int = 4,
    phase4_tiles_n: int = 1,
    phase4_tiles_k: int = 8,
    fp8_x4_dtype: type = float8_e4m3fn_x4,
    spill_reload: bool = True,
    use_scale_packing: bool = True,
) -> tuple:
    """MXFP8 SwiGLU MLP backward pass (base kernel).

    Accepts SEPARATE gate_proj_weight_T [H, I] and up_proj_weight_T [H, I].
    Returns fused gate_up_proj_weight_grad [2I, H] via weight_grad_td.

    All four intermediate tensors (gate_pre, gate_act, up, hidden) are REQUIRED.
    Use the wrapper mlp_backward_mxfp8_nki for checkpoint/recompute support.

    Forward recap:
        gate_pre     = hidden @ W_gate.T
        gate_act     = SiLU(gate_pre)
        up           = hidden @ W_up.T
        intermediate = gate_act * up
        output       = intermediate @ W_down.T

    Backward phases:
        Phase 1: d_intermediate = output_grad @ W_down.T, then SwiGLU bwd
        Phase 2: hidden_states_grad = d_gate @ W_gate + d_up @ W_up
        Phase 3: [dW_gate; dW_up] = grad_T[2I,S] @ hidden_states[S,H] -> [2I,H]
        Phase 4: dW_down = output_grad.T @ intermediate

    Dimensions:
        S: Sequence length.
        H: Hidden dimension size.
        I: Intermediate dimension size.

    Args:
        output_grad_td (TensorDescriptor): [S, H], incoming gradient dL/d_output (is_f_by_k=True).
        hidden_states_td (TensorDescriptor): [S, H], original input (for phase 3 weight grad).
        gate_pre_td (TensorDescriptor): [S, I], gate pre-activation (before SiLU).
        gate_act_td (TensorDescriptor): [S, I], gate post-activation (SiLU(gate_pre)).
        up_td (TensorDescriptor): [S, I], up projection (hidden @ W_up.T).
        hidden_td (TensorDescriptor): [S, I], gated intermediate (gate_act * up, for phase 4).
        gate_weight_T_td (TensorDescriptor): [H, I], transposed gate projection weights.
        up_weight_T_td (TensorDescriptor): [H, I], transposed up projection weights.
        down_weight_T_td (TensorDescriptor): [I, H], transposed down projection weights.
        d_gate_td (TensorDescriptor): [S, I], scratch: gate gradient.
        d_up_td (TensorDescriptor): [S, I], scratch: up gradient.
        hidden_states_T_td (TensorDescriptor): [H, S], pre-transposed input hidden states.
        output_grad_T_td (TensorDescriptor): [H, S], pre-transposed output gradient.
        hidden_T_td (TensorDescriptor): [I, S], pre-transposed intermediate activations.
        scratch_td (TensorDescriptor): [2I, S], scratch: transposed d_gate || d_up.
        hidden_states_grad_td (TensorDescriptor): [S, H], output: dL/d_hidden.
        weight_grad_td (TensorDescriptor): [2I, H], output: fused [dW_gate; dW_up].
        down_weight_grad_td (TensorDescriptor): [H, I], output: dL/dW_down.
        run_with_lnc2 (bool): Whether to shard across 2 LNC cores.
        phase1_tiles_m (int): M blocking for phase 1.
        phase1_tiles_n (int): N blocking for phase 1.
        phase1_tiles_k (int): K blocking for phase 1.
        phase2_tiles_m (int): M blocking for phase 2.
        phase2_tiles_n (int): N blocking for phase 2.
        phase2_tiles_k (int): K blocking for phase 2.
        phase3_tiles_m (int): M blocking for phase 3.
        phase3_tiles_n (int): N blocking for phase 3.
        phase3_tiles_k (int): K blocking for phase 3.
        phase4_tiles_m (int): M blocking for phase 4.
        phase4_tiles_n (int): N blocking for phase 4.
        phase4_tiles_k (int): K blocking for phase 4.
        fp8_x4_dtype: MXFP8 quantized data type.

    Returns:
        tuple: (hidden_states_grad [S, H], gate_up_weight_grad [2I, H], down_weight_grad [H, I]).

    Pseudocode:
        Phase 1: d_intermediate = output_grad @ W_down.T; SwiGLU bwd -> d_gate, d_up
        Phase 2: hidden_states_grad = d_gate @ W_gate + d_up @ W_up
        Phase 3: [dW_gate; dW_up] = grad_T[2I,S] @ hidden_states[S,H] -> [2I,H]
        Phase 4: dW_down = output_grad.T @ intermediate
    """
    H, S = output_grad_td.logical_shape
    I = gate_weight_T_td.logical_shape[0]
    dtype = nl.bfloat16

    if run_with_lnc2:
        kernel_assert(S % NUM_LNC2_CORES == 0, f"S ({S}) must be even for LNC2")
        kernel_assert(I % NUM_LNC2_CORES == 0, f"I ({I}) must be even for LNC2 (phase 3 I-sharding)")
        kernel_assert(H % NUM_LNC2_CORES == 0, f"H ({H}) must be even for LNC2 (phase 4 H-sharding)")

    # DGT requires K dimension divisible by DGT_MIN_K for unswizzled BF16 inputs
    # Phase 1: K=H (output_grad, down_weight)
    kernel_assert(
        output_grad_td.is_quantized or output_grad_td.is_swizzled or H % DGT_MIN_K == 0,
        f"H ({H}) must be divisible by {DGT_MIN_K} for DGT when output_grad is unswizzled BF16",
    )
    kernel_assert(
        down_weight_T_td.is_quantized or down_weight_T_td.is_swizzled or H % DGT_MIN_K == 0,
        f"H ({H}) must be divisible by {DGT_MIN_K} for DGT when down_weight is unswizzled BF16",
    )
    # Phase 2: K=I (d_gate/d_up LHS are always unswizzled BF16; gate_weight/up_weight RHS)
    kernel_assert(
        I % DGT_MIN_K == 0,
        f"I ({I}) must be divisible by {DGT_MIN_K} for DGT (phase 2 d_gate/d_up are always unswizzled BF16)",
    )
    # Phase 3: K=S (scratch grad_T is always unswizzled BF16, hidden_states_T)
    kernel_assert(
        S % DGT_MIN_K == 0,
        f"S ({S}) must be divisible by {DGT_MIN_K} for DGT (phase 3 grad_T is always unswizzled BF16)",
    )
    # Phase 4: K=S (output_grad_T, hidden_T)
    kernel_assert(
        output_grad_T_td.is_quantized or output_grad_T_td.is_swizzled or S % DGT_MIN_K == 0,
        f"S ({S}) must be divisible by {DGT_MIN_K} for DGT when output_grad_T is unswizzled BF16",
    )
    kernel_assert(
        hidden_T_td.is_quantized or hidden_T_td.is_swizzled or S % DGT_MIN_K == 0,
        f"S ({S}) must be divisible by {DGT_MIN_K} for DGT when hidden_T is unswizzled BF16",
    )

    _, shard_id = get_program_sharding_info(run_with_lnc2)

    s_base = shard_id * (S // NUM_LNC2_CORES) if run_with_lnc2 else 0
    h_base = shard_id * (H // NUM_LNC2_CORES) if run_with_lnc2 else 0

    compute_phase1_down_proj_mm_grad_mxfp8(
        output_grad_td=output_grad_td,
        gate_pre_td=gate_pre_td,
        gate_act_td=gate_act_td,
        up_td=up_td,
        d_gate_td=d_gate_td,
        d_up_td=d_up_td,
        scratch_td=scratch_td,
        down_weight_td=down_weight_T_td,
        s_base=s_base,
        dtype=dtype,
        fp8_x4_dtype=fp8_x4_dtype,
        TILES_IN_BLOCK_M=phase1_tiles_m,
        TILES_IN_BLOCK_N=phase1_tiles_n,
        TILES_IN_BLOCK_K=phase1_tiles_k,
        spill_reload=spill_reload,
        use_scale_packing=use_scale_packing,
        run_with_lnc2=run_with_lnc2,
    )

    nisa.core_barrier(data=d_gate_td.data, cores=(0, 1))
    nisa.core_barrier(data=d_up_td.data, cores=(0, 1))

    compute_phase2_hidden_states_grad_mxfp8(
        hidden_states_grad_td=hidden_states_grad_td,
        gate_weight_td=gate_weight_T_td,
        up_weight_td=up_weight_T_td,
        d_gate_td=d_gate_td,
        d_up_td=d_up_td,
        s_base=s_base,
        dtype=dtype,
        fp8_x4_dtype=fp8_x4_dtype,
        TILES_IN_BLOCK_M=phase2_tiles_m,
        TILES_IN_BLOCK_N=phase2_tiles_n,
        TILES_IN_BLOCK_K=phase2_tiles_k,
        spill_reload=spill_reload,
        use_scale_packing=use_scale_packing,
        run_with_lnc2=run_with_lnc2,
    )

    nisa.core_barrier(data=scratch_td.data, cores=(0, 1))

    compute_phase3_gate_up_weight_grad_mxfp8(
        weight_grad_td=weight_grad_td,
        hidden_states_T_td=hidden_states_T_td,
        grad_T_td=scratch_td,
        dtype=dtype,
        fp8_x4_dtype=fp8_x4_dtype,
        TILES_IN_BLOCK_M=phase3_tiles_m,
        TILES_IN_BLOCK_N=phase3_tiles_n,
        TILES_IN_BLOCK_K=phase3_tiles_k,
        spill_reload=spill_reload,
        use_scale_packing=use_scale_packing,
        run_with_lnc2=run_with_lnc2,
    )

    compute_phase4_down_weight_grad_mxfp8(
        down_weight_grad_td=down_weight_grad_td,
        output_grad_T_td=output_grad_T_td,
        hidden_T_td=hidden_T_td,
        h_base=h_base,
        dtype=dtype,
        fp8_x4_dtype=fp8_x4_dtype,
        TILES_IN_BLOCK_M=phase4_tiles_m,
        TILES_IN_BLOCK_N=phase4_tiles_n,
        TILES_IN_BLOCK_K=phase4_tiles_k,
        spill_reload=spill_reload,
        use_scale_packing=use_scale_packing,
        run_with_lnc2=run_with_lnc2,
    )

    return hidden_states_grad_td.data, weight_grad_td.data, down_weight_grad_td.data


def mlp_backward_mxfp8_nki(
    output_hidden_states_grad: nl.ndarray,
    hidden_states: nl.ndarray,
    gate_proj_weight_T: nl.ndarray,
    up_proj_weight_T: nl.ndarray,
    down_proj_weight_T: nl.ndarray,
    gate_up_weights: nl.ndarray,
    d_gate_scratch: nl.ndarray,
    d_up_scratch: nl.ndarray,
    hidden_states_T: nl.ndarray,
    output_grad_T: nl.ndarray,
    hidden_T: nl.ndarray,
    silu_up_mul_gate_grad_T_scratch: nl.ndarray,
    gate_pre_scratch: nl.ndarray,
    gate_act_scratch: nl.ndarray,
    up_scratch: nl.ndarray,
    hidden_scratch: nl.ndarray,
    gate_pre: nl.ndarray = None,
    gate_act: nl.ndarray = None,
    up: nl.ndarray = None,
    hidden: nl.ndarray = None,
    run_with_lnc2: bool = True,
    phase1_tiles_m: int = 8,
    phase1_tiles_n: int = 1,
    phase1_tiles_k: int = 8,
    phase2_tiles_m: int = 8,
    phase2_tiles_n: int = 1,
    phase2_tiles_k: int = 8,
    phase3_tiles_m: int = 4,
    phase3_tiles_n: int = 1,
    phase3_tiles_k: int = 8,
    phase4_tiles_m: int = 4,
    phase4_tiles_n: int = 1,
    phase4_tiles_k: int = 8,
    recompute_tiles_m: int = 8,
    recompute_tiles_n: int = 1,
    recompute_tiles_k: int = 8,
    fp8_x4_dtype: type = float8_e4m3fn_x4,
    spill_reload: bool = True,
    use_scale_packing: bool = True,
    # Phase 1 pre-swizzled/pre-quantized input support
    output_grad_scales: nl.ndarray = None,
    output_grad_is_swizzled: bool = False,
    down_weight_scales: nl.ndarray = None,
    down_weight_is_swizzled: bool = False,
    # Phase 2 pre-swizzled/pre-quantized input support
    gate_weight_scales: nl.ndarray = None,
    gate_weight_is_swizzled: bool = False,
    up_weight_scales: nl.ndarray = None,
    up_weight_is_swizzled: bool = False,
    # Phase 3 pre-swizzled/pre-quantized input support
    hidden_states_T_scales: nl.ndarray = None,
    hidden_states_T_is_swizzled: bool = False,
    # Recompute pre-swizzled/pre-quantized input support
    hidden_states_scales: nl.ndarray = None,
    hidden_states_is_swizzled: bool = False,
    gate_up_weights_scales: nl.ndarray = None,
    gate_up_weights_is_swizzled: bool = False,
    # Phase 4 pre-swizzled/pre-quantized input support
    output_grad_T_scales: nl.ndarray = None,
    output_grad_T_is_swizzled: bool = False,
    hidden_T_scales: nl.ndarray = None,
    hidden_T_is_swizzled: bool = False,
) -> tuple:
    """MXFP8 SwiGLU MLP backward pass with activation checkpointing support.

    Public-facing wrapper that handles recomputation of missing checkpoint tensors,
    then delegates to mlp_backward_mxfp8_base_nki.

    Accepts fused gate_up_proj_weight_T [H, 2I] and gate_up_weights [2I, H].
    Internally slices into separate gate [H, I] and up [H, I] for the base kernel.
    Returns fused gate_up_proj_weight_grad [2I, H] and down_proj_weight_grad [H, I].

    Forward recap:
        gate_pre     = hidden @ W_gate.T
        gate_act     = SiLU(gate_pre)
        up           = hidden @ W_up.T
        intermediate = gate_act * up
        output       = intermediate @ W_down.T

    Backward phases:
        Phase 0 (conditional): Recompute any missing checkpointed activations.
                 Dependency chain: gate_pre/up → gate_act → hidden.
        Phase 1-4: Delegated to mlp_backward_mxfp8_base_nki.

    Checkpointing contract:
        - gate_pre, gate_act, up, hidden: if None, will be recomputed into
          the corresponding *_scratch buffer using hidden_states + gate_up_weights.
        - Scratch buffers must always be provided.

    Dimensions:
        S: Sequence length.
        H: Hidden dimension size.
        I: Intermediate dimension size.

    Args:
        output_hidden_states_grad (nl.ndarray): [S, H], incoming gradient dL/d_output.
        hidden_states (nl.ndarray): [S, H], original input (for recompute + phase 3).
        gate_proj_weight_T (nl.ndarray): [H, I], transposed gate projection weights (phase 2).
        up_proj_weight_T (nl.ndarray): [H, I], transposed up projection weights (phase 2).
        down_proj_weight_T (nl.ndarray): [I, H], transposed down projection weights (phase 1).
        gate_up_weights (nl.ndarray): [2I, H], fused gate+up weights (for recompute).
        d_gate_scratch (nl.ndarray): [S, I], scratch: gate gradient.
        d_up_scratch (nl.ndarray): [S, I], scratch: up gradient.
        hidden_states_T (nl.ndarray): [H, S], pre-transposed input hidden states.
        output_grad_T (nl.ndarray): [H, S], pre-transposed output gradient.
        hidden_T (nl.ndarray): [I, S], pre-transposed intermediate activations.
        silu_up_mul_gate_grad_T_scratch (nl.ndarray): [2I, S], scratch: transposed d_gate || d_up.
        gate_pre_scratch (nl.ndarray): [S, I], scratch buffer for gate_pre.
        gate_act_scratch (nl.ndarray): [S, I], scratch buffer for gate_act.
        up_scratch (nl.ndarray): [S, I], scratch buffer for up.
        hidden_scratch (nl.ndarray): [S, I], scratch buffer for hidden.
        gate_pre (nl.ndarray): [S, I], checkpointed gate pre-activation, or None.
        gate_act (nl.ndarray): [S, I], checkpointed SiLU(gate_pre), or None.
        up (nl.ndarray): [S, I], checkpointed up projection, or None.
        hidden (nl.ndarray): [S, I], checkpointed gate_act * up, or None.

    Pre-swizzled/pre-quantized input support:
        Each matmul operand tensor accepts an optional ``*_scales`` (nl.ndarray)
        and ``*_is_swizzled`` (bool) pair. When both are default (None/False),
        the tensor is treated as unswizzled BF16 — identical to prior behavior.

        output_grad_scales, output_grad_is_swizzled: Phase 1 LHS (output_hidden_states_grad).
        down_weight_scales, down_weight_is_swizzled: Phase 1 RHS (down_proj_weight_T).
        gate_weight_scales, gate_weight_is_swizzled: Phase 2 RHS (gate_proj_weight_T).
        up_weight_scales, up_weight_is_swizzled: Phase 2 RHS (up_proj_weight_T).
        hidden_states_T_scales, hidden_states_T_is_swizzled: Phase 3 RHS (hidden_states_T).
        hidden_states_scales, hidden_states_is_swizzled: Recompute LHS (hidden_states).
        gate_up_weights_scales, gate_up_weights_is_swizzled: Recompute RHS (gate_up_weights).
        output_grad_T_scales, output_grad_T_is_swizzled: Phase 4 LHS (output_grad_T).
        hidden_T_scales, hidden_T_is_swizzled: Phase 4 RHS (hidden_T).

    Returns:
        tuple: (hidden_states_grad [S, H], gate_up_weight_grad [2I, H],
                down_proj_weight_grad [H, I]).

    Pseudocode:
        Phase 0: Recompute missing checkpoints (gate_pre, up, gate_act, hidden)
        Phase 1: d_intermediate = output_grad @ W_down.T; SwiGLU bwd -> d_gate, d_up
        Phase 2: hidden_states_grad = d_gate @ W_gate + d_up @ W_up
        Phase 3: [dW_gate; dW_up] = grad_T[2I,S] @ hidden_states[S,H] -> [2I,H]
        Phase 4: dW_down = output_grad.T @ intermediate
    """
    if get_active_sbm() == None:
        create_and_set_active_sbm()

    sbm = get_active_sbm()
    sbm.open_scope(name="MXFP8 MLP BWD ")

    # Build TDs for dimension derivation (these tensors may be pre-swizzled/pre-quantized)
    output_grad_td = TensorDescriptor(
        data=output_hidden_states_grad,
        scales=output_grad_scales,
        is_swizzled=output_grad_is_swizzled,
        is_col_parallel_sharded=run_with_lnc2,
    )
    gate_weight_T_td = TensorDescriptor(
        data=gate_proj_weight_T, scales=gate_weight_scales, is_swizzled=gate_weight_is_swizzled
    )

    # Derive dimensions from logical shapes
    H, S = output_grad_td.logical_shape
    I = gate_weight_T_td.logical_shape[0]
    dtype = nl.bfloat16

    if run_with_lnc2:
        kernel_assert(S % NUM_LNC2_CORES == 0, f"S ({S}) must be even for LNC2")
        kernel_assert(I % NUM_LNC2_CORES == 0, f"I ({I}) must be even for LNC2 (phase 3 I-sharding)")
        kernel_assert(H % NUM_LNC2_CORES == 0, f"H ({H}) must be even for LNC2 (phase 4 H-sharding)")

    # DGT requires K dimension divisible by DGT_MIN_K for unswizzled BF16 inputs
    kernel_assert(
        output_grad_td.is_quantized or output_grad_td.is_swizzled or H % DGT_MIN_K == 0,
        f"H ({H}) must be divisible by {DGT_MIN_K} for DGT when output_grad is unswizzled BF16",
    )
    kernel_assert(
        hidden_states_T_is_swizzled or hidden_states_T_scales != None or S % DGT_MIN_K == 0,
        f"S ({S}) must be divisible by {DGT_MIN_K} for DGT when hidden_states_T is unswizzled BF16",
    )

    # Allocate output buffers
    hidden_states_grad = nl.ndarray((S, H), dtype=dtype, buffer=nl.shared_hbm if run_with_lnc2 else nl.hbm)
    gate_up_weight_grad = nl.ndarray((2 * I, H), dtype=dtype, buffer=nl.shared_hbm if run_with_lnc2 else nl.hbm)
    down_proj_weight_grad = nl.ndarray((H, I), dtype=dtype, buffer=nl.shared_hbm if run_with_lnc2 else nl.hbm)

    _, shard_id = get_program_sharding_info(run_with_lnc2)

    s_base = shard_id * (S // NUM_LNC2_CORES) if run_with_lnc2 else 0

    # Phase 0: Recompute missing checkpointed activations (gate_pre/up → gate_act → hidden).
    # Resolve effective tensors: use checkpoint if provided, else recompute into scratch.
    eff_gate_pre = gate_pre if gate_pre != None else gate_pre_scratch
    eff_up = up if up != None else up_scratch
    eff_gate_act = gate_act if gate_act != None else gate_act_scratch
    eff_hidden = hidden if hidden != None else hidden_scratch

    # Build TDs needed for recompute and phases 1-4
    hidden_states_td = TensorDescriptor(
        data=hidden_states,
        scales=hidden_states_scales,
        is_swizzled=hidden_states_is_swizzled,
        is_col_parallel_sharded=run_with_lnc2,
    )
    gate_up_weights_td = TensorDescriptor(
        data=gate_up_weights,
        scales=gate_up_weights_scales,
        is_swizzled=gate_up_weights_is_swizzled,
        is_col_parallel_sharded=True,
    )
    gate_pre_scratch_td = TensorDescriptor(data=gate_pre_scratch)
    up_scratch_td = TensorDescriptor(data=up_scratch)
    gate_act_scratch_td = TensorDescriptor(data=gate_act_scratch)
    hidden_scratch_td = TensorDescriptor(data=hidden_scratch)

    eff_gate_pre_td = TensorDescriptor(data=eff_gate_pre)
    eff_up_td = TensorDescriptor(data=eff_up)
    eff_gate_act_td = TensorDescriptor(data=eff_gate_act)
    eff_hidden_td = TensorDescriptor(data=eff_hidden)

    # Step 0a: Recompute gate_pre and/or up if not checkpointed
    need_recompute_projections = (gate_pre == None) or (up == None)
    if need_recompute_projections:
        recompute_gate_up_projection(
            hidden_td=hidden_states_td,
            gate_up_td=gate_up_weights_td,
            gate_pre_td=gate_pre_scratch_td,
            up_td=up_scratch_td,
            s_base_offset=s_base,
            dtype=dtype,
            fp8_x4_dtype=fp8_x4_dtype,
            TILES_IN_BLOCK_M=recompute_tiles_m,
            TILES_IN_BLOCK_N=recompute_tiles_n,
            TILES_IN_BLOCK_K=recompute_tiles_k,
            spill_reload=spill_reload,
            use_scale_packing=use_scale_packing,
            run_with_lnc2=run_with_lnc2,
        )

    nisa.core_barrier(gate_pre_scratch_td.data, cores=(0, 1))

    # Step 0b: Recompute gate_act = SiLU(gate_pre) if not checkpointed
    if gate_act == None:
        recompute_gate_act(
            gate_pre_td=eff_gate_pre_td,
            gate_act_td=gate_act_scratch_td,
            s_base_offset=s_base,
            dtype=dtype,
            run_with_lnc2=run_with_lnc2,
        )

    if need_recompute_projections:
        nisa.core_barrier(up_scratch_td.data, cores=(0, 1))

    # Step 0c: Recompute hidden = gate_act * up if not checkpointed
    if hidden == None:
        recompute_hidden(
            gate_act_td=eff_gate_act_td,
            up_td=eff_up_td,
            hidden_td=hidden_scratch_td,
            s_base_offset=s_base,
            dtype=dtype,
            run_with_lnc2=run_with_lnc2,
        )

    # Build remaining TensorDescriptors for phases 1-4
    up_weight_T_td = TensorDescriptor(data=up_proj_weight_T, scales=up_weight_scales, is_swizzled=up_weight_is_swizzled)
    down_weight_T_td = TensorDescriptor(
        data=down_proj_weight_T, scales=down_weight_scales, is_swizzled=down_weight_is_swizzled
    )
    d_gate_td = TensorDescriptor(data=d_gate_scratch, is_f_by_k=True, is_col_parallel_sharded=run_with_lnc2)
    d_up_td = TensorDescriptor(data=d_up_scratch, is_f_by_k=True, is_col_parallel_sharded=run_with_lnc2)
    hidden_states_T_td = TensorDescriptor(
        data=hidden_states_T, scales=hidden_states_T_scales, is_swizzled=hidden_states_T_is_swizzled
    )
    output_grad_T_td = TensorDescriptor(
        data=output_grad_T, scales=output_grad_T_scales, is_swizzled=output_grad_T_is_swizzled
    )
    hidden_T_td = TensorDescriptor(data=hidden_T, scales=hidden_T_scales, is_swizzled=hidden_T_is_swizzled)
    scratch_td = TensorDescriptor(
        data=silu_up_mul_gate_grad_T_scratch, is_f_by_k=True, is_col_parallel_sharded=run_with_lnc2
    )
    hidden_states_grad_td = TensorDescriptor(data=hidden_states_grad)
    weight_grad_td = TensorDescriptor(data=gate_up_weight_grad)
    down_weight_grad_td = TensorDescriptor(data=down_proj_weight_grad)

    # Phases 1-4: Delegate to base kernel.
    mlp_backward_mxfp8_base_nki(
        output_grad_td=output_grad_td,
        hidden_states_td=hidden_states_td,
        gate_pre_td=eff_gate_pre_td,
        gate_act_td=eff_gate_act_td,
        up_td=eff_up_td,
        hidden_td=eff_hidden_td,
        gate_weight_T_td=gate_weight_T_td,
        up_weight_T_td=up_weight_T_td,
        down_weight_T_td=down_weight_T_td,
        d_gate_td=d_gate_td,
        d_up_td=d_up_td,
        hidden_states_T_td=hidden_states_T_td,
        output_grad_T_td=output_grad_T_td,
        hidden_T_td=hidden_T_td,
        scratch_td=scratch_td,
        hidden_states_grad_td=hidden_states_grad_td,
        weight_grad_td=weight_grad_td,
        down_weight_grad_td=down_weight_grad_td,
        run_with_lnc2=run_with_lnc2,
        phase1_tiles_m=phase1_tiles_m,
        phase1_tiles_n=phase1_tiles_n,
        phase1_tiles_k=phase1_tiles_k,
        phase2_tiles_m=phase2_tiles_m,
        phase2_tiles_n=phase2_tiles_n,
        phase2_tiles_k=phase2_tiles_k,
        phase3_tiles_m=phase3_tiles_m,
        phase3_tiles_n=phase3_tiles_n,
        phase3_tiles_k=phase3_tiles_k,
        phase4_tiles_m=phase4_tiles_m,
        phase4_tiles_n=phase4_tiles_n,
        phase4_tiles_k=phase4_tiles_k,
        fp8_x4_dtype=fp8_x4_dtype,
        spill_reload=spill_reload,
        use_scale_packing=use_scale_packing,
    )

    sbm.close_scope()

    return hidden_states_grad, gate_up_weight_grad, down_proj_weight_grad
