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

"""Fused RMSNorm + MX-quantization (+ optional router top-K) in token-major [T, H] layout (prefill).


Memory:
    SBUF is allocated explicitly via ``SbufManager`` to control buffer placement and break
    anti-dependencies. Two principles:
      - Cross-tile: the per-tile scope is opened with ``interleave_degree=4`` + ``increment_section()``
        per iteration, so consecutive tiles use distinct SBUF address sets -- tile N+k's DMA load
        overlaps tile N's compute. (degree 1->4 took the H=3072/E=128 target ~450M to ~326M cycles;
        degree 5 reached ~311M but is the SBUF-max there, so 4 is used to keep headroom for larger E/H.)
      - Intra-tile: the inner k/buf loops (swizzle, spill) ``alloc_stack`` a FRESH address each
        iteration (no per-iteration scope), so no inner buffer is overwritten while still being read
        (zero intra-tile WAR), and all free together at the tile section rotation.
    The spill coalesces all per-k transposed blocks into one staging buffer per region and issues a
    SINGLE wide DMA each (vs num_H512 tiny 512 B stores). Loop-invariants (gamma_bc, biases, router
    weights, expert iota) live in the outer scope. PSUM is allocated directly (SbufManager is SBUF-only).

Flow per token tile (always processes a full 128-partition tile; partial tiles pad to 128, store n_tok):
    1. plain DMA load HBM [T, H] -> [128, H] bf16
    2. fused norm+transpose, split so transpose/router don't wait on the full-H reduction
       (inv_rms[t] is a per-token scalar constant across h, so it factors out of both):
       PASS 1 per H512 block: accumulate sum-of-squares; apply GAMMA ONLY; FP32-packed
         swizzle-transpose -> persistent swizzled bf16 (qmx layout); (optional) router matmul.
       between: inv_rms = rsqrt(ss/unpadded_hidden_size + eps); build inv_rms_swz via a PE transpose-broadcast.
       PASS 2 per H512 block: scale swizzled by inv_rms (bf16-view mul); quantize_mx.
    3. (optional) router top-K (scale logits by inv_rms first) + activation + scatter
    4. transpose quant back to token-major and spill packed [T, H + scale_region] to HBM

Constraints: H a multiple of 512; Trn3+ only (quantize_mx is NeuronCore-v4).
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ..utils.allocator import SbufManager, sizeinbytes
from ..utils.common_types import RouterActFnType
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ..utils.logging import get_logger
from ..utils.stream_shuffle_broadcast import stream_shuffle_broadcast

_H0 = 128  # SBUF partition dim (nl.tile_size.pmax)
_1B_XPOSE_PSUM_STEP = 2  # 1-byte (fp8) transpose requires PSUM output element-step 2 (HW constraint)
_K_BLOCK = 512  # bf16 elements per MX block (-> 128 partitions after swizzle)
_SWIZZLE_STRIDE = 2  # stride-2 interleave in the FP32-packed transpose
_TRANSPOSED_FREE = 256  # _K_BLOCK // _BF16_TO_FP32_PACK
_QUANT_FREE = 512  # _K_BLOCK (bf16 lanes for quantize_mx)
_Q_WIDTH = 4  # MX 4-wide lane (_q_width)
_SCALES_PER_BLOCK = 4  # H512 tiles foldable into one 128-partition scale block
_QUADRANT = 32  # MX scale quadrant size (HW quadrant layout: 4 valid scales per 32 partitions)
_F_MAX = 512  # gemm moving free-dim max (router E <= 512)
_SBUF_SCRATCH_RESERVE = 1024  # bytes reserved below the SbufManager upper bound
_NEG_FLT_MAX = -3.4028235e38  # most negative finite fp32; fills top-K padding so it never wins
_MAX_TILE_INTERLEAVE = 4  # Maximum x Buffering Degree


@nki.jit
def rmsnorm_mx_prefill(
    hidden_states: nl.NkiTensor,
    gamma: nl.NkiTensor,
    router_weights: nl.NkiTensor = None,
    router_bias: nl.NkiTensor = None,
    eps: float = 1e-6,
    top_k: int = 1,
    router_act_fn: RouterActFnType = RouterActFnType.SIGMOID,
    qmx_output_dtype=nl.float8_e4m3fn_x4,
    pack_scales: bool = True,
    pack_affinities: bool = False,
    unpadded_hidden_size: int = None,
    residual: nl.NkiTensor = None,
):
    """Fused RMSNorm [T,H] + MX quantization (+ optional router top-K) for prefill.

    Dimensions:
        B: batch, S: sequence, T = B*S tokens
        H: hidden dim (multiple of 512); E: experts (<= 512); K: top_k (<= 8)
        H0 = 128 partition fold, num_H512 = H/512, q_width = 4

    Args:
        hidden_states (nl.NkiTensor): [B, S, H] bf16 input on HBM. Loaded in NATURAL H order -- the
            kernel applies the swizzle internally during the FP32-packed transpose (see below), so the
            caller must NOT pre-permute the hidden states.
        gamma (nl.NkiTensor): [1, H] or [H] RMSNorm weights on HBM. Natural H order (indexes hidden_states
            directly); not swizzled.
        router_weights (nl.NkiTensor): [H, E] router weights PRE-PERMUTED into the kernel's internal
            swizzle H order on HBM. If None, the router is skipped and only the packed quant tensor is
            returned.

            Why the permute: the router matmul runs against the swizzle-TRANSPOSED activations (the same
            transpose quantize_mx needs), whose partition axis is the swizzled H index
            h_swz = h512*512 + 4*p + q (h512 = H512-block, p = 0..127 partition, q = 0..3 lane) -- NOT
            natural h. To make logits = sum_h norm[t,h] * W[h,e] come out right, row h of the natural
            weight must be placed at swizzle slot h_swz. Equivalently, build the permuted weight as
            W_perm[swizzle_slot] = W[swizzle_h_index[swizzle_slot]].

            Reference swizzle (offline host weight-prep):

                def swizzle_h_index(H):                  # swizzle slot -> original H index
                    num_h512 = H // 512
                    h512 = torch.arange(num_h512).reshape(num_h512, 1, 1)
                    q    = torch.arange(4).reshape(1, 4, 1)
                    p    = torch.arange(128).reshape(1, 1, 128)
                    return (h512 * 512 + 4 * p + q).reshape(-1)   # shape [H]

                router_weights_permuted = natural_router_weights[swizzle_h_index(H)]   # [H, E]

        router_bias (nl.NkiTensor): [1, E] or [E] optional router bias on HBM.
        eps (float): epsilon for numerical stability.
        top_k (int): number of experts to select per token (<= 8). Only used when router_weights set.
        router_act_fn (RouterActFnType): SIGMOID or SOFTMAX. Only used when router_weights set.
        qmx_output_dtype: packed MX output dtype (float8_e4m3fn_x4 default).
        pack_scales (bool): controls how the per-block MX scales are laid out in the scale region of
            each packed output row. quantize_mx emits one uint8 scale per (32-partition quadrant x
            4-lane) group, so each H512 tile produces scales that only occupy 4 of every 32 columns
            after transpose -- 7/8 of an unfolded scale tile is wasted padding.
            - pack_scales=True (default, "folded"): pack 4 consecutive H512 tiles into one 128-wide
              block by shifting tile k into within-quadrant offset (k%4)*4. This reclaims the wasted
              columns, so scale_region = ceil(num_H512/4)*128 (roughly H/16 columns).
            - pack_scales=False ("unfolded"): one 128-wide block per H512 tile, no folding, so
              scale_region = num_H512*128 (= H/4 columns) -- ~4x larger
        pack_affinities (bool): router only. If True, the dense [T, E] expert affinities are
            concatenated into each packed row after the scale region (as bf16 reinterpreted into the
            fp8 row) instead of returned as a separate HBM tensor, so a downstream block gather pulls
            [hidden | scale | affinities] in one indirect DMA. The total row is padded to a multiple
            of 4 fp8 columns (the hidden region's fp32-reinterpret transpose requires it). When False
            (default), expert_affinities is returned as its own [T, E] tensor (legacy layout).
        unpadded_hidden_size (int): actual (unpadded) hidden size for the RMS mean denominator. When the
            input H is zero-padded offline (e.g. up to a multiple of 512), the sum-of-squares is taken
            over the full padded H (the zero pad contributes 0), but the mean divides by unpadded_hidden_size
            so padding does not skew the norm. Defaults to H (no padding).
        residual (nl.NkiTensor): [B, S, H] optional residual on HBM. When set, the kernel adds it to
            hidden_states before RMSNorm (hidden = hidden_states + residual); the norm/quant/router
            all consume the sum. The pre-norm sum is also written out (output_residual) for the next
            layer's residual stream. If None, no residual add is performed.

    Returns:
        list of HBM tensors, always starting with norm_quant_packed (nl.NkiTensor): [T, row_region]
        fp8, where row_region = H + scale_region, or the affinity-padded H_load when pack_affinities
        (router only). Then appended in order when the corresponding input is set: expert_index
        [T,top_k] int32 (when router_weights is set), then expert_affinities [T,E] (when router_weights
        is set AND not pack_affinities -- when pack_affinities the affinities live inside the packed
        row, so no separate tensor is returned), then output_residual [T,H] (when residual is set).

    Pseudocode:
        for each token tile of 128 tokens (padded to 128, store n_tok):
            # 1. load [128, H] bf16 (+ optional residual add, also spilled to output_residual)
            in_tile = hidden_states[tile]  (+ residual[tile])
            # 2. fused norm + swizzle-transpose + (optional) router, inv_rms deferred past transpose:
            #    PASS 1 per H512 block: accumulate sum-of-squares; norm = in * gamma; FP32-packed
            #    stride-2 swizzle-transpose -> swizzled_all; router_logits += swizzled @ W
            #    between: inv_rms = rsqrt(ss / unpadded_hidden_size + eps); broadcast to inv_rms_swz
            #    PASS 2 per H512 block: swizzled_all *= inv_rms_swz; quantize_mx -> quant + scale
            # 3. (optional) router top-K: logits = inv_rms * raw_logits (+ bias); activation; scatter
            # 4. transpose quant + scales back to token-major; spill packed [n_tok, H + scale_region]
    """
    kernel_assert(len(hidden_states.shape) == 3, f"hidden_states must be [B, S, H], got {hidden_states.shape}")
    B, S, H = hidden_states.shape
    T = B * S
    kernel_assert(H % _K_BLOCK == 0, f"H ({H}) must be a multiple of {_K_BLOCK}")

    # Padded-H handling: SS is summed over the full (zero-padded) H, mean divides by unpadded_hidden_size.
    if unpadded_hidden_size == None:
        unpadded_hidden_size = H
    kernel_assert(0 < unpadded_hidden_size <= H, f"unpadded_hidden_size ({unpadded_hidden_size}) must be in (0, H={H}]")

    num_H512 = H // _K_BLOCK
    n_packed = div_ceil(num_H512, _SCALES_PER_BLOCK) if pack_scales else num_H512
    scale_region = n_packed * _H0

    in_dtype = hidden_states.dtype
    has_router = router_weights != None
    compute_dtype = router_weights.dtype if has_router else in_dtype

    """Affinity packing (router only): the dense [T, E] affinities are appended to each packed row as
    bf16 reinterpreted into the fp8 columns (E bf16 == E*_BF16_AS_FP8 fp8 cols). The downstream
    transpose reinterprets the hidden region as fp32, so the TOTAL row must be a multiple of
    _Q_WIDTH fp8 cols; pad the row accordingly (div_ceil(row, 4)*4)."""
    _BF16_AS_FP8 = sizeinbytes(compute_dtype)  # bf16 -> 2 fp8 columns
    pack_affinities = pack_affinities and has_router
    if pack_affinities:
        E = router_weights.shape[1]
        affin_off = H + scale_region  # fp8-column offset of the affinity region in the row
        row_region = div_ceil(affin_off + E * _BF16_AS_FP8, _Q_WIDTH) * _Q_WIDTH
    else:
        affin_off = 0
        row_region = H + scale_region

    in_view = hidden_states.reshape((T, H))
    gamma_view = gamma.reshape((1, H))
    out_packed = nl.ndarray((T, row_region), dtype=nl.float8_e4m3fn, buffer=nl.shared_hbm)

    has_residual = residual != None
    if has_residual:
        kernel_assert(
            tuple(residual.shape) == (B, S, H), f"residual must match hidden_states {(B, S, H)}, got {residual.shape}"
        )
        residual_view = residual.reshape((T, H))
        output_residual = nl.ndarray((T, H), dtype=residual.dtype, buffer=nl.shared_hbm)
    else:
        residual_view = None
        output_residual = None

    if has_router:
        E = router_weights.shape[1]
        kernel_assert(E <= _F_MAX, f"E ({E}) must be <= {_F_MAX}")
        kernel_assert(top_k <= 8, f"top_k ({top_k}) must be <= 8")
        num_h_tiles = num_H512 * _Q_WIDTH
        expert_index = nl.ndarray((T, top_k), dtype=nl.int32, buffer=nl.shared_hbm)
        expert_affinities = None if pack_affinities else nl.ndarray((T, E), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    else:
        E = 0
        expert_index = None
        expert_affinities = None

    # LNC sharding on the token dimension T (tokens are independent: no cross-core exchange).
    _, n_prgs, prg_id = get_verified_program_sharding_info("rmsnorm_mx_prefill", (0, 1))
    nominal_shard = div_ceil(T, n_prgs)
    shard_offset = prg_id * nominal_shard
    shard_size = min(nominal_shard, T - shard_offset) if shard_offset < T else 0

    # ---- SBUF manager + outer (loop-invariant) scope ----
    sb_upper = nl.tile_size.total_available_sbuf_size - _SBUF_SCRATCH_RESERVE
    sbm = SbufManager(0, sb_upper, get_logger("rmsnorm_mx_prefill"))
    sbm.open_scope(name="invariants")

    # gamma broadcast [1,H] -> [128,H], plus rsqrt bias constant (all loop-invariant).
    gamma_bc = sbm.alloc_stack((_H0, H), dtype=gamma.dtype, name="gamma_bc")
    _broadcast_gamma_once(sbm, gamma_view, gamma_bc, H, gamma.dtype)
    eps_bias = sbm.alloc_stack((_H0, 1), dtype=nl.float32, name="eps_bias")
    nisa.memset(eps_bias, value=eps)
    # Loop-invariant identity[128,128] for the inv_rms single-matmul transpose-broadcast
    # stationary = inv_rms broadcast on free (stride-0), moving = identity -> dst[h,t] = inv_rms[t].
    identity_f32 = nl.shared_identity_matrix(_H0, dtype=nl.float32)

    if has_router:
        # Router weights [128, num_h_tiles, E] (partition=H). Source is ht-major so permute axes 0/1.
        router_weights_sb = sbm.alloc_stack((_H0, num_h_tiles, E), dtype=router_weights.dtype, name="router_weights_sb")
        router_weights_src = router_weights.reshape((num_h_tiles, _H0, E)).permute((1, 0, 2))
        nisa.dma_copy(dst=router_weights_sb, src=router_weights_src, dge_mode=nisa.dge_mode.hwdge)

        # Loop-invariant expert-number iota [128, E] for the one-hot scatter.
        # [0, 1, 2, ... E-1] 128 rows
        expert_ids = sbm.alloc_stack((_H0, E), dtype=nl.float32, name="expert_ids")
        nisa.iota(dst=expert_ids[0:_H0, :E], pattern=[[1, E]], offset=0, channel_multiplier=0)

        bias_bcast = None
        if router_bias != None:
            bias_bcast = sbm.alloc_stack((_H0, E), dtype=nl.float32, name="bias_bcast")
            bias_row = sbm.alloc_stack((1, E), dtype=nl.float32, name="bias_row")
            # bf16 router_bias -> f32 bias_row
            nisa.dma_copy(dst=bias_row, src=router_bias.reshape((1, E)))
            stream_shuffle_broadcast(src=bias_row, dst=bias_bcast)
    else:
        router_weights_sb = expert_ids = bias_bcast = None

    # ---- Per-tile scope, interleave degree budgeted against free SBUF (cross-tile pipeline)
    num_token_tiles = div_ceil(shard_size, _H0)  # token tiles of 128 per shard
    # Free space here already excludes the invariants allocated above; divide by the per-tile section
    # footprint to find how many buffer sets fit.

    per_tile_bytes = _tile_section_bytes(H, num_H512, E, top_k, compute_dtype, qmx_output_dtype, in_dtype)
    tile_interleave = max(1, min(_MAX_TILE_INTERLEAVE, sbm.get_free_space() // per_tile_bytes))

    sbm.open_scope(interleave_degree=tile_interleave)
    for tile_idx in nl.range(num_token_tiles):
        tok_off = shard_offset + tile_idx * _H0
        n_tok = min(_H0, shard_offset + shard_size - tok_off)

        """Stage 1: load tile (+ optional residual add, fused on the DMA engine via dma_compute so it
        costs no Vector/Scalar/PE cycles -- the bottleneck engines stay free for quantize_mx). The
        pre-norm sum hidden = input + residual feeds the norm and is also spilled to output_residual."""
        in_tile = sbm.alloc_stack((_H0, H), dtype=in_dtype, name=f"in_tile_t{tile_idx}")
        if has_residual:
            nisa.dma_compute(
                dst=in_tile[0:n_tok, 0:H],
                srcs=[in_view[tok_off : tok_off + n_tok, 0:H], residual_view[tok_off : tok_off + n_tok, 0:H]],
                reduce_op=nl.add,
            )
            nisa.dma_copy(
                dst=output_residual[tok_off : tok_off + n_tok, 0:H],
                src=in_tile[0:n_tok, 0:H],
                dge_mode=nisa.dge_mode.hwdge,
            )
        else:
            nisa.dma_copy(
                dst=in_tile[0:n_tok, 0:H], src=in_view[tok_off : tok_off + n_tok, 0:H], dge_mode=nisa.dge_mode.hwdge
            )

        """Stages 2-4: fused RMSNorm + swizzle-transpose + router, split so the transpose/router path
        does not wait on the full-H sum-of-squares. PASS 1 (per H512 block): accumulate squares,
        apply gamma only, transpose, copy to a persistent swizzled buffer, router matmul. Then rsqrt
        + build inv_rms_swz via a PE transpose-broadcast. PASS 2: scale by inv_rms and quantize_mx."""
        swizzled_all = sbm.alloc_stack(
            (_H0, num_H512, _TRANSPOSED_FREE), dtype=nl.float32, name=f"swizzled_all_t{tile_idx}"
        )
        inv_rms = sbm.alloc_stack(
            (_H0, 2), dtype=nl.float32, name=f"inv_rms_t{tile_idx}"
        )  # col0 = accumulated SS, col1 = rsqrt
        quant_swz_sb = sbm.alloc_stack((_H0, num_H512, _H0), dtype=qmx_output_dtype, name=f"quant_swz_sb_t{tile_idx}")
        scale_swz_sb = sbm.alloc_stack(
            (_H0, num_H512, _H0), dtype=nl.uint8, name=f"scale_swz_sb_t{tile_idx}"
        )  # packed when spilling
        logits_psum = nl.ndarray((_H0, E), dtype=nl.float32, buffer=nl.psum) if has_router else None

        _fused_norm_router_transpose_quantize(
            sbm,
            in_tile,
            gamma_bc,
            eps_bias,
            identity_f32,
            inv_rms,
            swizzled_all,
            quant_swz_sb,
            scale_swz_sb,
            n_tok,
            H,
            num_H512,
            unpadded_hidden_size,
            compute_dtype,
            tile_idx,
            router_weights_sb=router_weights_sb if has_router else None,
            router_logits_psum=logits_psum,
        )

        # Stage 5b: per-tile router top-K + activation + scatter (real n_tok rows only).
        if has_router:
            _router_topk_from_logits(
                sbm,
                logits_psum,
                bias_bcast,
                expert_index,
                expert_affinities,
                tok_off,
                n_tok,
                E,
                top_k,
                router_act_fn,
                expert_ids,
                inv_rms,
                tile_idx,
                out_packed=out_packed if pack_affinities else None,
                affin_off=affin_off,
                row_region=row_region,
            )

        # Stage 6: transpose quant + scales back to token-major and spill packed row.
        _spill_packed(
            sbm, quant_swz_sb, scale_swz_sb, out_packed, tok_off, n_tok, H, num_H512, n_packed, pack_scales, tile_idx
        )

        sbm.increment_section()  # rotate to the other buffer set for the next tile
    sbm.close_scope()
    sbm.close_scope()

    outputs = [out_packed]
    if has_router:
        outputs.append(expert_index)
        # When pack_affinities, the affinities live inside out_packed -> no standalone tensor.
        if expert_affinities != None:
            outputs.append(expert_affinities)
    if has_residual:
        outputs.append(output_residual)
    return outputs


def _broadcast_gamma_once(
    sbm: SbufManager, gamma_view: nl.NkiTensor, gamma_bc: nl.NkiTensor, H: int, gamma_dtype
) -> None:
    """Broadcast gamma [1, H] across all 128 token partitions, once.

    In [T, H] the token axis is on partitions, so gamma (a per-H vector) must be replicated to every
    partition. Partition-axis stride-0 reads are not allowed for the vector engine. PE broadcast:
    ones[1,128] (stationary) x gamma[1,H] (moving) contracts the size-1 partition ->
    out[p,h] = gamma[0,h] for all p. PSUM free-dim caps at 512, so tile H into _K_BLOCK chunks.

    gamma_loaded is raw-DMA-loaded so it must match gamma_dtype; ones matches it as the matmul partner.
    """
    gamma_loaded = sbm.alloc_stack((1, H), dtype=gamma_dtype, name="gamma_loaded")
    nisa.dma_copy(dst=gamma_loaded[0:1, 0:H], src=gamma_view[0:1, 0:H], dge_mode=nisa.dge_mode.hwdge)
    ones = sbm.alloc_stack((1, _H0), dtype=gamma_dtype, name="gamma_bc_ones")
    nisa.memset(ones, value=1.0)
    for chunk_idx in nl.range(div_ceil(H, _K_BLOCK)):
        chunk_off = chunk_idx * _K_BLOCK
        chunk_sz = min(_K_BLOCK, H - chunk_off)
        broadcast_psum = nl.ndarray((_H0, _K_BLOCK), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=broadcast_psum[0:_H0, 0:chunk_sz],
            stationary=ones[0:1, 0:_H0],
            moving=gamma_loaded[0:1, chunk_off : chunk_off + chunk_sz],
        )
        nisa.tensor_copy(dst=gamma_bc[0:_H0, chunk_off : chunk_off + chunk_sz], src=broadcast_psum[0:_H0, 0:chunk_sz])


def _fused_norm_router_transpose_quantize(
    sbm: SbufManager,
    in_tile: nl.NkiTensor,
    gamma_bc: nl.NkiTensor,
    eps_bias: nl.NkiTensor,
    identity: nl.NkiTensor,
    inv_rms: nl.NkiTensor,
    swizzled_all: nl.NkiTensor,
    quant_swz_sb: nl.NkiTensor,
    scale_swz_sb: nl.NkiTensor,
    n_tok: int,
    H: int,
    num_H512: int,
    unpadded_hidden_size: int,
    compute_dtype,
    tile_idx: int,
    router_weights_sb: nl.NkiTensor = None,
    router_logits_psum: nl.NkiTensor = None,
) -> None:
    """Fused RMSNorm + swizzle-transpose + router, with the inv_rms scale deferred past the transpose.

    inv_rms[t] is a per-token scalar constant across the hidden dim, so it factors out of both the
    router matmul (logits = inv_rms[t] * sum_h in*gamma*W) and quantize_mx. That lets PASS 1 apply
    GAMMA ONLY per H512 block -- transpose + router proceed without waiting on the full-H
    sum-of-squares reduction (better pipelining).

    PASS 1 (loop block_idx over num_H512 512-wide blocks):
      a. accumulate sum-of-squares into reduce_regs (read out into inv_rms[:,0] on the last block);
      b. norm_block = in_block * gamma_block (compute_dtype view of an fp32-packed staging buffer);
      c. fp32-packed stride-2 swizzle transpose -> PSUM;
      d. copy PSUM -> persistent swizzled_all[:,block_idx,:] (one block per iter);
      e. router matmul on the swizzled compute_dtype activations (no inv_rms).
    Between passes: rsqrt finalize, then build inv_rms_swz [128,512] via a single-matmul transpose-
    broadcast: stationary = inv_rms[:,1] read with a free stride-0 broadcast -> [n_tok,128],
    moving = identity[n_tok,n_tok], so dst[h,t] = sum_p inv_rms[p]*delta(p,t) = inv_rms[t] (transpose +
    partition-broadcast in one PE op), then an interleaving copy to the 4*t+q free layout.
    PASS 2: scale every block of swizzled_all by inv_rms_swz (bf16-view multiply -- packing-safe),
    then quantize_mx per block.

    Args (all SBUF unless noted; partition axis first):
        in_tile           [128_t, H] in_dtype    : this tile's token-major hidden states (token on part).
        gamma_bc          [128_t, H] gamma_dtype : gamma broadcast across all token partitions.
        eps_bias          [128_t, 1] fp32        : eps constant for the rsqrt bias.
        identity          [128, 128] fp32        : identity matrix (moving operand of the broadcast matmul).
        inv_rms           [128_t, 2] fp32        : scratch; col0 = sum-of-squares, col1 = finalized rsqrt.
        swizzled_all      [128_H, num_H512, 256] fp32 : persistent swizzled output; compute_dtype view is
                                                  [128_H, num_H512, 512] with free = 4*t+q.
        quant_swz_sb      [128_H, num_H512, 128] fp8x4 : quantize_mx output (H on partition).
        scale_swz_sb      [128_H, num_H512, 128] uint8 : per-block MX scales (H on partition).
        n_tok             int                    : real tokens this tile (<= 128; rest are padding).
        H, num_H512       int                    : (padded) hidden dim and H/512 block count.
        unpadded_hidden_size     int                    : unpadded hidden size; mean divides by this, not H.
        compute_dtype     dtype                  : 16-bit dtype of the swizzle/router/quant-feed path
                                                  (matches router_weights so the router matmul is consistent).
        router_weights_sb [128_H, num_H512*4_H, E] compute_dtype : pre-permuted router weights, or None.
        router_logits_psum[128_t, E] fp32 PSUM   : router logits accumulator, or None (no router).

    swizzled_all compute_dtype view has partition stride num_H512*_QUANT_FREE; block block_idx starts at
    compute_dtype free offset block_idx*_QUANT_FREE.
    """
    has_router = router_weights_sb != None
    E = router_weights_sb.shape[2] if has_router else 0

    # ---- PASS 1: per-block square-accumulate, gamma, transpose, copy, router ----
    for block_idx in nl.range(num_H512):
        block_off = block_idx * _QUANT_FREE  # select which H512 tile
        # (a) sum-of-squares accumulation in the Scalar reduce registers across all blocks.
        squared = sbm.alloc_stack((_H0, _QUANT_FREE), dtype=nl.float32, name=f"squared_t{tile_idx}_b{block_idx}")
        is_first_block = block_idx == 0
        is_last_block = block_idx == num_H512 - 1
        nisa.activation(
            dst=squared[0:n_tok, 0:_QUANT_FREE],
            op=nl.square,
            data=in_tile[0:n_tok, block_off : block_off + _QUANT_FREE],
            reduce_op=nl.add,
            reduce_cmd=nisa.reduce_cmd.reset_reduce if is_first_block else nisa.reduce_cmd.reduce,
            reduce_res=inv_rms[0:n_tok, 0:1] if is_last_block else None,
        )

        # (b) gamma-only multiply into the 16-bit view of an fp32-packed per-block staging buffer.
        norm_block = sbm.alloc_stack(
            (_H0, _TRANSPOSED_FREE), dtype=nl.float32, name=f"norm_block_t{tile_idx}_b{block_idx}"
        )
        norm_block_16b = norm_block.view(compute_dtype)
        nisa.tensor_tensor(
            dst=norm_block_16b[0:n_tok, 0:_QUANT_FREE],
            data1=in_tile[0:n_tok, block_off : block_off + _QUANT_FREE],
            data2=gamma_bc[0:n_tok, block_off : block_off + _QUANT_FREE],
            op=nl.multiply,
        )

        # (c) fp32-packed stride-2 swizzle transpose -> PSUM.
        transposed_psum = nl.ndarray((_H0, _TRANSPOSED_FREE), dtype=nl.float32, buffer=nl.psum)  # (128, 256)
        for stride_idx in nl.range(_SWIZZLE_STRIDE):  # 2
            src_ap = norm_block.ap(
                pattern=[[_TRANSPOSED_FREE, _H0], [_SWIZZLE_STRIDE, _H0]],
                offset=stride_idx,
            )  # select every other index
            dst_ap = transposed_psum.ap(
                pattern=[[_TRANSPOSED_FREE, _H0], [_SWIZZLE_STRIDE, _H0]],
                offset=stride_idx,
            )  # transposed write out with a stride of 2, and writing to even or odd indices
            nisa.nc_transpose(dst=dst_ap, data=src_ap)

        # (d) copy PSUM -> persistent swizzled_all block
        nisa.tensor_copy(dst=swizzled_all[0:_H0, block_idx, 0:_TRANSPOSED_FREE], src=transposed_psum)

        # (e) router matmul on the swizzled bf16 (no inv_rms -- it factors out, applied to logits later).
        if has_router:
            for q_lane in nl.range(_Q_WIDTH):
                stationary_q = swizzled_all.ap(
                    pattern=[[num_H512 * _QUANT_FREE, _H0], [_Q_WIDTH, n_tok]],  # [[H, H0], [4, 128]]
                    offset=block_off + q_lane,  # select which q in 4_H to use
                    dtype=compute_dtype,
                )
                # 16-bit matmul and fp32 accumulation
                nisa.nc_matmul(
                    dst=router_logits_psum[0:n_tok, 0:E],
                    stationary=stationary_q[0:_H0, 0:n_tok],
                    moving=router_weights_sb[0:_H0, block_idx * _Q_WIDTH + q_lane, 0:E],
                )

    # ---- Between passes: rsqrt finalize + build inv_rms_swz via PE transpose-broadcast ----
    nisa.activation(
        dst=inv_rms[0:n_tok, 1:2],  # stored in slot index 1
        op=nl.rsqrt,
        data=inv_rms[0:n_tok, 0:1],
        bias=eps_bias[0:n_tok, 0:1],
        scale=1.0 / unpadded_hidden_size,  # SS summed over padded H; mean divides by unpadded_hidden_size
    )

    # Single-matmul transpose-broadcast: stationary = inv_rms[:128_t,1] free-stride-0 broadcast [n_tok,128],
    # moving = identity[n_tok,n_tok] -> broadcast_psum[h,t] = inv_rms[t].
    inv_rms_stationary = inv_rms.ap(pattern=[[2, _H0], [0, _H0]], offset=1)  # [n_tok,128], free replicated
    broadcast_psum = nl.ndarray((_H0, _H0), dtype=nl.float32, buffer=nl.psum)
    # transpose + broadcast
    nisa.nc_matmul(
        dst=broadcast_psum[0:_H0, 0:n_tok],
        stationary=inv_rms_stationary[0:n_tok, 0:_H0],
        moving=identity[0:n_tok, 0:n_tok],
        is_moving_onezero=True,
    )

    # Materialize the 4*t+q free interleave straight out of PSUM (one copy: evict + cast + interleave):
    # inv_rms_swz[h, 4*t+q] = broadcast_psum[h,t] (q stride-0 read replicates across the 4 lanes).
    inv_rms_swz = sbm.alloc_stack(
        (_H0, _QUANT_FREE), dtype=compute_dtype, name=f"inv_rms_swz_t{tile_idx}"
    )  # (128, 512)
    src_ap = broadcast_psum.ap(
        pattern=[[_H0, _H0], [1, n_tok], [0, _Q_WIDTH]]
    )  # reads the same [H, T] value 4x times (every T has same scale across H)
    dst_ap = inv_rms_swz.ap(pattern=[[_QUANT_FREE, _H0], [_Q_WIDTH, n_tok], [1, _Q_WIDTH]])
    nisa.tensor_copy(dst=dst_ap, src=src_ap)

    # ---- PASS 2: scale each block by inv_rms_swz (16-bit-view multiply), then quantize_mx ----
    for block_idx in nl.range(num_H512):
        # 16-bit view of swizzled_all block: [128_H, 512]
        block_16b = swizzled_all.ap(
            pattern=[[num_H512 * _QUANT_FREE, _H0], [1, _QUANT_FREE]],
            offset=block_idx * _QUANT_FREE,
            dtype=compute_dtype,
        )
        # [128, 512] * [128, 512] -> [128, 512] in place (inv_rms_swz[h, 4*t+q] = inv_rms[t]).
        nisa.tensor_tensor(
            dst=block_16b[0:_H0, 0:_QUANT_FREE],
            data1=block_16b[0:_H0, 0:_QUANT_FREE],
            data2=inv_rms_swz[0:_H0, 0:_QUANT_FREE],
            op=nl.multiply,
        )
    for block_idx in nl.range(num_H512):
        # 16-bit view of swizzled_all block: [_H0 (=128 H), _QUANT_FREE (=512, free = 4*t+q)]
        quant_src = swizzled_all.ap(
            pattern=[[num_H512 * _QUANT_FREE, _H0], [1, _QUANT_FREE]],
            offset=block_idx * _QUANT_FREE,
            dtype=compute_dtype,
        )
        """src [128_H, 4*n_tok] -> quant_swz_sb[:, block_idx, :] fp8x4 [128_H, n_tok] + scale [128_H, n_tok].
        quantize_mx folds 4 q-lanes (free = 4*t+q) into 1 output per token, so src free must be
        exactly _Q_WIDTH*n_tok. For full tiles _Q_WIDTH*128 == _QUANT_FREE; for partial tiles (small
        T / uneven last shard) n_tok < 128, so slicing to _Q_WIDTH*n_tok keeps the in/out AP sizes
        matched (dst is already n_tok-bound)."""
        nisa.quantize_mx(
            dst=quant_swz_sb[0:_H0, block_idx : block_idx + 1, 0:n_tok],
            src=quant_src[0:_H0, 0 : _Q_WIDTH * n_tok],
            dst_scale=scale_swz_sb[0:_H0, block_idx : block_idx + 1, 0:n_tok],
        )


def _router_topk_from_logits(
    sbm,
    logits_psum,
    bias_bcast,
    expert_index,
    expert_affinities,
    tok_off,
    n_tok,
    E,
    top_k,
    act_fn,
    expert_ids,
    inv_rms,
    tile_idx,
    out_packed=None,
    affin_off=0,
    row_region=0,
):
    """Top-K + activation + one-hot scatter from accumulated router logits; store this tile's rows.

    logits_psum is [128, E] (full tile; first n_tok rows real). Writes expert_index[tok_off:tok_off+n_tok]
    and the dense affinities for this tile. expert_ids is the loop-invariant [128, E] expert iota.

    Affinity destination: when out_packed is None the affinities are DMA'd to the standalone
    expert_affinities[tok_off:tok_off+n_tok] tensor (legacy). When out_packed is given (pack_affinities),
    the affinities are written into a bf16 view of out_packed's row tail at fp8-column offset affin_off
    (row stride = row_region fp8 cols), so a downstream block gather pulls them with hidden+scale.

    The router matmul ran on gamma-only activations (inv_rms factored out), so logits_psum holds
    raw = sum_h in*gamma*W. The true logit is inv_rms[t]*raw (+ bias). inv_rms is per-token = the
    partition axis here, so it applies as a clean per-partition tensor_scalar. It MUST be applied
    before the bias-add and top-K: bias is per-expert and added after the scale, so scaling post-top-K
    would mis-order experts whenever a bias is present.
    """
    E_padded = max(E, 8)  # max8/find_index8 need >= 8 columns
    logits_sb = sbm.alloc_stack((_H0, E_padded), dtype=nl.float32, name=f"logits_sb_t{tile_idx}")
    if E_padded > E:
        # Pad columns to -FLT_MAX so the fake experts can never win top-K (any real logit beats them).
        nisa.memset(logits_sb, value=_NEG_FLT_MAX)

    # logits_sb = inv_rms * raw (+ bias). With bias, fuse scale+add into one scalar_tensor_tensor;
    # without bias, a per-partition tensor_scalar on the Scalar engine suffices.
    if bias_bcast != None:
        nisa.scalar_tensor_tensor(
            dst=logits_sb[0:n_tok, 0:E],
            data=logits_psum[0:n_tok, 0:E],
            op0=nl.multiply,
            operand0=inv_rms[0:n_tok, 1:2],
            op1=nl.add,
            operand1=bias_bcast[0:n_tok, 0:E],
        )
    else:
        nisa.tensor_scalar(
            dst=logits_sb[0:n_tok, 0:E],
            data=logits_psum[0:n_tok, 0:E],
            op0=nl.multiply,
            operand0=inv_rms[0:n_tok, 1:2],
            engine=nisa.scalar_engine,
        )

    top8 = sbm.alloc_stack((_H0, 8), dtype=nl.float32, name=f"top8_t{tile_idx}")
    nisa.max8(dst=top8[0:_H0, :], src=logits_sb[0:_H0, :])

    # Allocate as int32 (matches expert_index HBM dtype) but view as uint32 for find_index8's required
    # output dtype. Indices are 0..E-1, always positive, so the bit patterns are identical
    idx8_i32 = sbm.alloc_stack((_H0, 8), dtype=nl.int32, name=f"idx8_i32_t{tile_idx}")
    idx8 = idx8_i32.view(nl.uint32)
    nisa.nc_find_index8(dst=idx8[0:_H0, :], data=logits_sb[0:_H0, :], vals=top8[0:_H0, :])
    idx_topk = idx8[0:_H0, :top_k]
    nisa.dma_copy(
        dst=expert_index[tok_off : tok_off + n_tok, :], src=idx8_i32[0:n_tok, :top_k], dge_mode=nisa.dge_mode.hwdge
    )

    affin_topk = sbm.alloc_stack((_H0, 1, top_k), dtype=nl.float32, name=f"affin_topk_t{tile_idx}")
    # Activation scratch on the managed stack (SIGMOID uses none; SOFTMAX/DVE-exp use these).
    act_exp_v = sbm.alloc_stack((_H0, top_k), dtype=nl.float32, name=f"act_exp_v_t{tile_idx}")
    act_exp_sum = sbm.alloc_stack((_H0, 1), dtype=nl.float32, name=f"act_exp_sum_t{tile_idx}")
    act_negmax = sbm.alloc_stack((_H0, 1), dtype=nl.float32, name=f"act_negmax_t{tile_idx}")
    _apply_activation_single_tile(
        act_fn,
        affin_topk,
        top8[0:_H0, :top_k].reshape((_H0, 1, top_k)),
        _H0,
        top_k,
        act_exp_v,
        act_exp_sum,
        act_negmax,
    )

    # affin_full holds the dense [128, E] scatter result. Affinities are always bf16
    affin_full = sbm.alloc_stack((_H0, E), dtype=nl.bfloat16, name=f"affin_full_t{tile_idx}")
    # Scatter scratch on the managed stack: one idx_fp32, plus a fresh contrib buffer per top-K slot
    # (fresh address per kk -> slot kk+1's mask+scale overlaps slot kk's accumulate).
    scatter_idx_fp32 = sbm.alloc_stack((_H0, top_k), dtype=nl.float32, name=f"scatter_idx_fp32_t{tile_idx}")
    scatter_contrib_bufs = []
    for kk in range(top_k):
        scatter_contrib_bufs.append(
            sbm.alloc_stack((_H0, E), dtype=nl.bfloat16, name=f"scatter_contrib_t{tile_idx}_k{kk}")
        )
    _scatter_one_hot(
        affin_full, affin_topk, idx_topk, _H0, E, top_k, scatter_idx_fp32, scatter_contrib_bufs, expert_ids
    )
    if out_packed != None:
        """Packed: write affinities into the bf16-viewed tail of the fp8 row. The row stride is
        row_region fp8 cols == row_region/_BF16_AS_FP8 bf16 cols; the affinity region starts at fp8
        col affin_off == affin_off/_BF16_AS_FP8 bf16 cols (affin_off is even by construction)."""
        _bf16_as_fp8 = sizeinbytes(nl.bfloat16)  # 2 fp8 cols per bf16
        row_bf16 = row_region // _bf16_as_fp8  # bf16 cols per row
        affin_tail = out_packed.ap(
            pattern=[[row_bf16, n_tok], [1, E]],
            offset=tok_off * row_bf16 + affin_off // _bf16_as_fp8,
            dtype=nl.bfloat16,
        )
        nisa.dma_copy(
            dst=affin_tail[0:n_tok, 0:E],
            src=affin_full[0:n_tok, 0:E],
            dge_mode=nisa.dge_mode.hwdge,
        )
    else:
        nisa.dma_copy(
            dst=expert_affinities[tok_off : tok_off + n_tok, 0:E],
            src=affin_full[0:n_tok, 0:E],
            dge_mode=nisa.dge_mode.hwdge,
        )


def _spill_packed(
    sbm: SbufManager,
    quant_swz_sb: nl.NkiTensor,
    scale_swz_sb: nl.NkiTensor,
    out_packed: nl.NkiTensor,
    tok_off: int,
    n_tok: int,
    H: int,
    num_H512: int,
    n_packed: int,
    pack_scales: bool,
    tile_idx: int,
) -> None:
    """Transpose quant (and scales) back to token-major and spill packed [n_tok, H + scale_region].

    quant_swz_sb is [H0, num_H512, n_tok] fp8x4 in swizzled (H-on-partition) order. fp8x4 cannot be
    PE-transposed directly, so reinterpret as fp32 (x4 -> 1 lane) for the transpose, then back to fp8.

    Scale fold (pack_scales): after transposing a scale tile to token-major [n_tok, 128], the valid MX
    scales live on the FREE axis at quadrant columns {0-3,32-35,64-67,96-99}. To pack 4 H512 tiles
    into one 128-wide block, tile (block % 4) shifts by tile*q_width within each quadrant. Each block
    stages into a memset-0 buffer (a partial last block leaves harmless zero holes).
    """
    """Coalesce all per-block transposed pieces into ONE token-major staging buffer per region, then do
    a SINGLE wide DMA to HBM instead of num_H512 tiny 512 B stores. quant_tok_major [n_tok,
    H] holds the transposed-back quant; scale_tok_major [n_tok, scale_region] holds the (folded)
    scales. No inner scope: each block allocates a fresh transpose-scratch address -> no intra-tile WAR."""
    scale_region = n_packed * _H0
    # quant_tok_major is fp8 [n_tok, H]; view its block's 512-wide slice as fp32 [n_tok, 128] so the
    # transpose result (fp32 in PSUM) copies straight in -- one PSUM->SBUF copy per block.
    quant_tok_major = sbm.alloc_stack((_H0, H), dtype=nl.float8_e4m3fn, name=f"quant_tok_major_t{tile_idx}")
    quant_tok_major_f32 = quant_tok_major.view(nl.float32)  # [n_tok, H/4] fp32
    quant_swz_f32 = quant_swz_sb.view(nl.float32)
    for block_idx in nl.range(num_H512):
        quant_t_psum = nl.ndarray((_H0, _H0), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=quant_t_psum[0:n_tok, 0:_H0], data=quant_swz_f32[0:_H0, block_idx, 0:n_tok])
        # PSUM->SBUF on the Scalar engine: by spill time this tile's RMSNorm (Scalar) is done, so
        # this keeps the bottleneck Vector engine free for the next tile's quantize_mx.
        engine = nisa.scalar_engine
        nisa.tensor_copy(
            dst=quant_tok_major_f32[0:n_tok, block_idx * _H0 : (block_idx + 1) * _H0],
            src=quant_t_psum[0:n_tok, 0:_H0],
            engine=engine,
        )
    nisa.dma_copy(
        dst=out_packed[tok_off : tok_off + n_tok, 0:H], src=quant_tok_major[0:n_tok, 0:H], dge_mode=nisa.dge_mode.hwdge
    )

    scale_swz_f8 = scale_swz_sb.view(nl.float8_e5m2)
    """Allocate the token-major scale buffer as int32 so the (unpacked-path) memset is ~4x cheaper per
    element on a 4-byte dtype (scale_region/4 int32 cols vs scale_region fp8 cols). 1 int32 == 4 fp8;
    all fp8 writes address this base via .ap(dtype=fp8). Packed path needs NO memset --
    reads only the quadrant-valid columns the fold writes, so the holes are never consumed."""
    scale_tok_major_i32 = sbm.alloc_stack(
        (_H0, scale_region // 4), dtype=nl.int32, align=_QUADRANT, name=f"scale_tok_major_i32_t{tile_idx}"
    )

    if not pack_scales:
        # Unpacked: each block writes a contiguous 128-wide slice; fill first so any partial last
        # block / unwritten column is an OOB sentinel (all-0xFF bytes) for skip indices.
        nisa.memset(scale_tok_major_i32, value=0xFFFFFFFF)
        for block_idx in nl.range(num_H512):
            scale_t_psum = nl.ndarray((_H0, _H0, _1B_XPOSE_PSUM_STEP), dtype=nl.float8_e5m2, buffer=nl.psum)
            nisa.nc_transpose(dst=scale_t_psum[0:n_tok, 0:_H0, 0], data=scale_swz_f8[0:_H0, block_idx, 0:n_tok])
            dst_ap = scale_tok_major_i32.ap(
                pattern=[[scale_region, n_tok], [1, _H0]],
                offset=block_idx * _H0,
                dtype=nl.float8_e5m2,
            )
            nisa.tensor_copy(dst=dst_ap, src=scale_t_psum[0:n_tok, 0:_H0, 0])
    else:
        n_quadrants = _H0 // _QUADRANT  # 4
        for pack_idx in nl.range(n_packed):
            n_tiles_here = min(_SCALES_PER_BLOCK, num_H512 - pack_idx * _SCALES_PER_BLOCK)
            for tile_in_block in nl.range(n_tiles_here):
                block_idx = pack_idx * _SCALES_PER_BLOCK + tile_in_block
                scale_t_psum = nl.ndarray((_H0, _H0, _1B_XPOSE_PSUM_STEP), dtype=nl.float8_e5m2, buffer=nl.psum)
                nisa.nc_transpose(dst=scale_t_psum[0:n_tok, 0:_H0, 0], data=scale_swz_f8[0:_H0, block_idx, 0:n_tok])
                src_ap = scale_t_psum.ap(
                    pattern=[
                        [_H0 * _1B_XPOSE_PSUM_STEP, n_tok],
                        [_QUADRANT * _1B_XPOSE_PSUM_STEP, n_quadrants],
                        [_1B_XPOSE_PSUM_STEP, _Q_WIDTH],
                    ],
                    offset=0,
                )
                # dst into the int32 base (viewed as fp8) pack pack_idx, quadrant-strided, shifted by
                # tile within quadrant.
                dst_ap = scale_tok_major_i32.ap(
                    pattern=[[scale_region, n_tok], [_QUADRANT, n_quadrants], [1, _Q_WIDTH]],
                    offset=pack_idx * _H0 + tile_in_block * _Q_WIDTH,
                    dtype=nl.float8_e5m2,
                )
                nisa.tensor_copy(dst=dst_ap, src=src_ap)
    scale_out = scale_tok_major_i32.view(nl.float8_e4m3fn)
    nisa.dma_copy(
        dst=out_packed[tok_off : tok_off + n_tok, H : H + scale_region],
        src=scale_out[0:n_tok, 0:scale_region],
        dge_mode=nisa.dge_mode.hwdge,
    )


def _apply_activation_single_tile(act_fn, dst_3d, src_3d, T, k, exp_v, exp_sum, negmax):
    """Apply sigmoid/softmax to the [T, 1, k] top-K values.

    On Trn3 B0+ the SOFTMAX path uses nisa.exponential (Vector/DVE engine), which fuses
    exp(src - max) and the sum-of-exp denominator into one op -- replacing the separate max
    tensor_reduce + exp activation + sum tensor_reduce. The top-K values come from max8 sorted
    descending, so element 0 is the per-token max used as the numerical-stability shift.

    The fused reduction only validates when the reduced element count supports the 2x/4x perf
    mode (an even count), so it is gated on even k -- odd k (incl. top_k=1) falls back to the
    plain max-reduce + exp + sum-reduce path.

    Scratch buffers (exp_v [T,k], exp_sum [T,1], negmax [T,1]) are allocated by the caller on the
    managed SBUF stack and passed in, so this routine does no allocation. SIGMOID needs none; the
    DVE-exp path uses exp_v+exp_sum; the plain SOFTMAX path uses negmax+exp_v+exp_sum.
    """
    # nisa.exponential is available on Trn3 B0+ (gen4 sub-version 1, or newer NeuronCore gens).
    has_dve_exp = (nisa.get_nc_version() > nisa.nc_version.gen4) or (
        nisa.get_nc_version() == nisa.nc_version.gen4 and nisa.get_nc_sub_version() == 1
    )
    if act_fn == RouterActFnType.SIGMOID:
        nisa.activation(dst=dst_3d[0:T, 0, 0:k], op=nl.sigmoid, data=src_3d[0:T, 0, 0:k])
    elif has_dve_exp and k % 2 == 0:  # SOFTMAX over the k values, fused on the DVE engine (even k only)
        nisa.exponential(
            dst=exp_v[0:T, :],
            src=src_3d[0:T, 0, 0:k],
            max_value=src_3d[0:T, 0, 0:1],
            reduce_res=exp_sum[0:T, :],
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )
        nisa.reciprocal(dst=exp_sum[0:T, :], data=exp_sum[0:T, :])
        nisa.tensor_scalar(dst=dst_3d[0:T, 0, 0:k], data=exp_v[0:T, :], op0=nl.multiply, operand0=exp_sum[0:T, :])
    else:  # SOFTMAX over the k values
        nisa.tensor_reduce(
            dst=negmax[0:T, :], op=nl.maximum, data=src_3d[0:T, 0, 0:k], axis=1, negate=True, keepdims=True
        )
        nisa.activation(dst=exp_v[0:T, :], op=nl.exp, data=src_3d[0:T, 0, 0:k], bias=negmax[0:T, :])
        nisa.tensor_reduce(dst=exp_sum[0:T, :], op=nl.add, data=exp_v[0:T, :], axis=1, keepdims=True)
        nisa.reciprocal(dst=exp_sum[0:T, :], data=exp_sum[0:T, :])
        nisa.tensor_scalar(dst=dst_3d[0:T, 0, 0:k], data=exp_v[0:T, :], op0=nl.multiply, operand0=exp_sum[0:T, :])


def _scatter_one_hot(expert_affinities, affin_topk, idx_topk, T, E, k, idx_fp32, contrib_bufs, expert_ids):
    """Scatter the compact top-K result into the dense [T, E] affinity table (zero elsewhere).

    Turns the per-token top-K (which expert + what score) into the dense output: each token's k
    scores land at the columns of its k chosen experts, zeros everywhere else. The hardware cannot
    index-write, so the scatter is done with k compute passes (one per top-K SLOT, not per expert):

    Inputs (per token tile):
        idx_topk    [T, k]      : the k chosen expert ids per token (different experts per token ok)
        affin_topk  [T, 1, k]   : their k affinity scores per token
        expert_ids  [T, E]      : iota ruler [0, 1, ..., E-1] on every row (loop-invariant; built
                                  once by the caller and passed in, or built here for one-shot use)

    Per slot kk (one vectorized op over ALL T tokens at once):
        contrib[T, E] = (expert_ids == idx_fp32[:, kk]) * affin_topk[:, 0, kk]
        i.e. for each token, a one-hot row at THAT token's kk-th expert column, scaled by its score
        (zeros elsewhere). operand0/operand1 are per-row scalars ([T] columns), so token t's nonzero
        lands at its own chosen column -- different column per token in the same instruction.
        Then expert_affinities += contrib accumulates it (the add merges the disjoint nonzeros; the
        zeros add nothing, so earlier slots survive -- tensor_scalar overwrites its dst, so a direct
        write would clobber prior slots, hence the separate accumulate).

    After k slots, expert_affinities holds each token's k scores at its k expert columns, 0 elsewhere.
    Cost is O(k) passes, independent of how many distinct experts appear across the tile.

    Scratch is caller-allocated on the managed stack: idx_fp32 [T,k] fp32, and contrib_bufs (a list of
    k [T,E] bf16 buffers, one fresh address per slot so slot kk+1's mask+scale overlaps slot kk's
    accumulate -- the per-kk fresh-address scheduling preserved from the original inline allocation).
    expert_ids is the loop-invariant [T,E] iota (always passed by this kernel's caller).
    """
    nisa.memset(expert_affinities[0:T, 0:E], value=0.0)
    # Tensor scalar requires fp32 operands
    nisa.tensor_copy(dst=idx_fp32[0:T, :], src=idx_topk[0:T, :])
    for kk in nl.range(k):
        # Fuse mask+scale into ONE tensor_scalar (op0=equal -> one-hot, op1=multiply -> scale by the
        # affinity scalar) writing directly to bf16, then accumulate.
        contrib = contrib_bufs[kk]
        nisa.tensor_scalar(
            dst=contrib[0:T, :],
            data=expert_ids[0:T, :],
            op0=nl.equal,
            operand0=idx_fp32[0:T, kk],
            op1=nl.multiply,
            operand1=affin_topk[0:T, 0, kk],
        )
        nisa.tensor_tensor(
            dst=expert_affinities[0:T, 0:E], data1=expert_affinities[0:T, 0:E], data2=contrib[0:T, :], op=nl.add
        )


def _tile_section_bytes(H: int, num_H512: int, E: int, top_k: int, compute_dtype, qmx_output_dtype, in_dtype) -> int:
    """Per-partition SBUF bytes one per-tile section consumes (one buffer set the interleave rotates).

    Used to budget the tile interleave degree against free SBUF: degree = free_space // this. The
    per-block PASS-1 scratch (squared, norm_block) allocates a FRESH address each of num_H512 blocks
    with no inner scope, so all num_H512 copies are live simultaneously within a tile -- hence the
    *num_H512 terms. The H-linear part (~12.25*H) dominates; the router/spill terms are small. Rounded
    up generously so the estimate never undercounts (an over-estimate just lowers the degree safely).
    """
    cd = sizeinbytes(compute_dtype)
    qb = sizeinbytes(qmx_output_dtype)
    fp32 = sizeinbytes(nl.float32)
    bf16 = sizeinbytes(nl.bfloat16)
    f8 = sizeinbytes(nl.float8_e4m3fn)
    E_pad = max(E, 8)
    total = 0
    total += sizeinbytes(in_dtype) * H  # in_tile
    total += fp32 * num_H512 * _TRANSPOSED_FREE  # swizzled_all
    total += fp32 * 2  # inv_rms
    total += qb * num_H512 * _H0  # quant_swz_sb
    total += num_H512 * _H0  # scale_swz_sb (uint8)
    total += fp32 * _QUANT_FREE * num_H512  # squared (one per block, all live)
    total += fp32 * _TRANSPOSED_FREE * num_H512  # norm_block (one per block, all live)
    total += cd * _QUANT_FREE  # inv_rms_swz
    # router top-K + scatter scratch
    total += fp32 * E_pad  # logits_sb
    total += fp32 * 8 + fp32 * 8  # top8 + idx8_i32
    total += fp32 * top_k  # affin_topk
    total += fp32 * (top_k + 1 + 1)  # activation scratch: exp_v[k] + exp_sum[1] + negmax[1]
    total += cd * E  # affin_full
    total += fp32 * top_k  # scatter idx_fp32
    total += bf16 * E * top_k  # scatter contrib (one per top-K slot)
    # spill staging
    total += f8 * H  # quant_tok_major
    total += fp32 * (div_ceil(num_H512, _SCALES_PER_BLOCK) * _H0 // 4)  # scale_tok_major_i32
    return total
