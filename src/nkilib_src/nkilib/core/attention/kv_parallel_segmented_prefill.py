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
"""KV-parallel segmented prefill attention kernel.

This kernel enables context parallelism by distributing the KV cache across multiple
ranks. Each rank computes attention over its local KV shard, then results are merged
using online softmax.

See README.md for detailed documentation.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil
from ..utils.modular_allocator import ModularAllocator
from .attention_segmented_cte import attention_segmented_cte

P_MAX = nl.tile_size.pmax


@nki.jit
def kv_parallel_segmented_prefill(
    q: nl.ndarray,
    k_cache: nl.ndarray,
    v_cache: nl.ndarray,
    block_tables: nl.ndarray,
    kvp_offset: nl.ndarray,
    replica_groups: ReplicaGroup,
    group_size: int,
    block_size: int,
    seg_size: int,
    scale: float = 1.0,
    global_q_offset: int = 0,
    tp_out: bool = False,
) -> nl.ndarray:
    """
    KV-parallel segmented prefill attention.

    Distributes attention computation across ranks, where each rank holds a shard
    of the KV cache. Uses online softmax to merge partial results.

    TODO: Specify intended usage range (e.g., sequence length, group size, head dimension).

    Dimensions:
        BS: Batch size (lnc_degree = Q heads per physical rank)
        S: Sequence length
        D: Head dimension
        G: Group size (number of ranks per replica group)
        R: Number of physical ranks per group (group_size // lnc_degree)

    Args:
        q (nl.ndarray): [BS, S, D], This rank's Q heads (BS = lnc_degree).
        k_cache (nl.ndarray): [num_blocks, num_kv_heads, block_size, D], Local KV cache (K).
        v_cache (nl.ndarray): [num_blocks, num_kv_heads, block_size, D], Local KV cache (V).
        block_tables (nl.ndarray): [1, max_blocks] int32, Block indices for paged KV.
        kvp_offset (nl.ndarray): [1, 1] int32, Causal mask offset =
            -rank_id * local_kv_len + global_q_offset.
        replica_groups (ReplicaGroup): ReplicaGroup for collective operations.
        group_size (int): Number of ranks in the replica group.
        block_size (int): KV cache block size.
        seg_size (int): Segment size for attention iteration.
        scale (float): Attention scale factor (default 1.0).
        global_q_offset (int): Global token position of Q token 0 (default 0). Used to compute
            how many prior KV tokens exist within this rank's shard for each Q chunk.

    Returns:
        out (nl.ndarray): [BS, S, D], Merged attention output for this rank's Q heads.

    Pseudocode:
        # Step 1: All-gather Q across ranks
        q_full = all_gather(q)  # [group_size, S, D]

        # Step 2: For each Q chunk, compute local attention
        for q_chunk_idx in range(num_q_chunks):
            q_chunk = q_full[shard_id, q_start:q_end, :]
            chunk_out, chunk_neg_max, chunk_sum_recip = attention_segmented_cte(q_chunk, k_cache, v_cache)
            partial_out[shard_id, q_start:q_end] = chunk_out

        # Step 3: Pack softmax stats + partial outputs, exchange via all-to-all
        send_packed = pack(partial_out, neg_max, sum_recip)
        recv_packed = all_to_all(send_packed)

        # Step 4: Merge partials using online softmax
        for tile_idx in range(num_tiles):
            global_neg_max = min(recv_packed[:, neg_max_channel])
            factors = exp(global_neg_max - neg_max_per_rank) / sum_recip_per_rank
            factors = factors / sum(factors)
            out[shard_id, tile] = sum(factors * recv_packed[:, out_channel])
    """
    bs, seq_len, head_dim = q.shape
    num_q_chunks = seq_len // seg_size

    shard_id = nl.program_id(0)

    # Input validation
    kernel_assert(seq_len % seg_size == 0, f"seq_len ({seq_len}) must be divisible by seg_size ({seg_size})")
    kernel_assert(group_size % bs == 0, f"group_size ({group_size}) must be divisible by lnc_degree ({bs})")
    kernel_assert(
        q.shape[2] == k_cache.shape[3],
        f"head_dim mismatch: q has {q.shape[2]}, k_cache has {k_cache.shape[3]}",
    )
    kernel_assert(
        k_cache.shape[2] == block_size,
        f"k_cache block_size dim ({k_cache.shape[2]}) must match block_size ({block_size})",
    )

    # bs = lnc_degree = Q heads per physical rank = NCs per physical rank
    lnc_degree = bs
    num_physical_ranks = group_size // lnc_degree
    heads_per_nc = group_size // lnc_degree
    q_heads_per_rank = lnc_degree

    # All-gather Q across ranks.
    # Collectives cannot read/write I/O tensors directly, so each NC DMAs its slice into shared_hbm first.
    q_src = nl.ndarray((lnc_degree, seq_len, head_dim), dtype=q.dtype, buffer=nl.shared_hbm, name="q_src")
    for nc_idx in range(lnc_degree):
        if shard_id == nc_idx:
            nisa.dma_copy(dst=q_src[nc_idx, :, :], src=q[nc_idx, :, :])

    q_full = nl.ndarray(
        (group_size, seq_len, head_dim),
        dtype=q.dtype,
        buffer=nl.shared_hbm,
        name="q_full",
    )
    ncc.all_gather(dsts=[q_full], srcs=[q_src], replica_group=replica_groups, collective_dim=0)

    partial_out = nl.ndarray(
        (group_size, seq_len, head_dim), dtype=nl.float32, buffer=nl.shared_hbm, name="partial_out"
    )
    neg_max = nl.ndarray((group_size, seq_len), dtype=nl.float32, buffer=nl.shared_hbm, name="neg_max")
    sum_recip = nl.ndarray((group_size, seq_len), dtype=nl.float32, buffer=nl.shared_hbm, name="sum_recip")

    chunk_allocator = ModularAllocator()
    kvp_offset_chunk_sbuf = chunk_allocator.alloc_sbuf_tensor((1, 1), nl.int32)
    kvp_offset_chunk_hbm = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.private_hbm, name="kvp_offset_chunk_hbm")
    prior_tokens_chunk_sbuf = chunk_allocator.alloc_sbuf_tensor((1, 1), nl.int32)
    prior_tokens_chunk_hbm = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.private_hbm, name="prior_tokens_chunk_hbm")

    num_kv_blocks = k_cache.shape[0]
    local_kv_len = num_kv_blocks * block_size
    max_prior_tokens = local_kv_len - seg_size

    # Compute local attention for each Q chunk.
    # Each NC processes heads_per_nc Q heads starting at shard_id * heads_per_nc.
    for q_chunk_idx in range(num_q_chunks):
        q_start = q_chunk_idx * seg_size
        q_end = q_start + seg_size

        q_chunk = nl.ndarray(
            (heads_per_nc, seg_size, head_dim), dtype=q.dtype, buffer=nl.private_hbm, name=f"q_chunk_{q_chunk_idx}"
        )
        nisa.dma_copy(
            dst=q_chunk,
            src=q_full[nl.ds(shard_id * heads_per_nc, heads_per_nc), q_start:q_end, :],
        )

        prior_tokens_for_chunk = min(q_start + global_q_offset, max_prior_tokens)

        nisa.dma_copy(dst=kvp_offset_chunk_sbuf, src=kvp_offset)
        nisa.tensor_scalar(dst=kvp_offset_chunk_sbuf, data=kvp_offset_chunk_sbuf, op0=nl.add, operand0=q_start)
        nisa.dma_copy(dst=kvp_offset_chunk_hbm, src=kvp_offset_chunk_sbuf)
        nisa.memset(prior_tokens_chunk_sbuf[...], value=prior_tokens_for_chunk)
        nisa.dma_copy(dst=prior_tokens_chunk_hbm, src=prior_tokens_chunk_sbuf)

        chunk_out, chunk_neg_max, chunk_sum_recip = attention_segmented_cte(
            q=q_chunk,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            prior_tokens=prior_tokens_chunk_hbm,
            block_size=block_size,
            prior_seg_size=seg_size,
            scale=scale,
            tp_q=True,
            tp_out=False,  # Always False internally; tp_out handled at final output write
            kvp_offset=kvp_offset_chunk_hbm,
        )

        nisa.dma_copy(
            dst=partial_out[nl.ds(shard_id * heads_per_nc, heads_per_nc), q_start:q_end, :],
            src=chunk_out,
        )
        nisa.dma_copy(
            dst=neg_max[nl.ds(shard_id * heads_per_nc, heads_per_nc), q_start:q_end],
            src=chunk_neg_max,
        )
        nisa.dma_copy(
            dst=sum_recip[nl.ds(shard_id * heads_per_nc, heads_per_nc), q_start:q_end],
            src=chunk_sum_recip,
        )
    # Pack partial outputs and softmax stats into send_packed[ranks, heads, seq, head_dim+2] and exchange via all-to-all.
    # TODO: Replace with coalesced all-to-all (3 separate tensors) once that support has been added.
    packed_dim = head_dim + 2
    send_packed = nl.ndarray(
        (num_physical_ranks, q_heads_per_rank, seq_len, packed_dim),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
        name="send_packed",
    )
    dest_rank_start = shard_id * heads_per_nc // q_heads_per_rank
    num_dest_ranks = heads_per_nc // q_heads_per_rank
    nisa.dma_copy(
        dst=send_packed[nl.ds(dest_rank_start, num_dest_ranks), :, :, :head_dim],
        src=partial_out[nl.ds(shard_id * heads_per_nc, heads_per_nc), :, :],
    )
    nisa.dma_copy(
        dst=send_packed[nl.ds(dest_rank_start, num_dest_ranks), :, :, head_dim : head_dim + 1],
        src=neg_max[nl.ds(shard_id * heads_per_nc, heads_per_nc), :],
    )
    nisa.dma_copy(
        dst=send_packed[nl.ds(dest_rank_start, num_dest_ranks), :, :, head_dim + 1 : head_dim + 2],
        src=sum_recip[nl.ds(shard_id * heads_per_nc, heads_per_nc), :],
    )

    recv_packed = nl.ndarray(
        (num_physical_ranks, q_heads_per_rank, seq_len, packed_dim),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
        name="recv_packed",
    )
    ncc.all_to_all(dsts=[recv_packed], srcs=[send_packed], replica_group=replica_groups, collective_dim=0)

    # Merge partial attention outputs using online softmax.
    allocator = ModularAllocator()
    out = nl.ndarray(
        (bs, head_dim, seq_len) if tp_out else (bs, seq_len, head_dim), dtype=q.dtype, buffer=nl.shared_hbm, name="out"
    )
    _merge_partial_attention_outputs(
        recv_packed, out, shard_id, num_physical_ranks, q_heads_per_rank, seq_len, head_dim, q.dtype, allocator, tp_out
    )
    return out


def _merge_partial_attention_outputs(
    recv_packed: nl.ndarray,
    out: nl.ndarray,
    shard_id: int,
    num_physical_ranks: int,
    q_heads_per_rank: int,
    seq_len: int,
    head_dim: int,
    out_dtype,
    allocator: ModularAllocator,
    tp_out: bool = False,
) -> None:
    """
    Merge partial attention outputs from all ranks using online softmax rescaling.

    recv_packed has shape [num_physical_ranks, q_heads_per_rank, seq_len, head_dim+2] where
    the last dim interleaves partial output ([:head_dim]), neg_max ([head_dim]), and
    sum_recip ([head_dim+1]) from each rank.

    Tiling strategy: seq_len is tiled in P_MAX-sized tiles. All SBUF buffers are allocated
    once at P_MAX size and reused across tiles, keeping SBUF pressure constant regardless
    of seq_len.
    """
    neg_max_sbuf = allocator.alloc_sbuf_tensor((P_MAX, num_physical_ranks), nl.float32)
    sum_recip_sbuf = allocator.alloc_sbuf_tensor((P_MAX, num_physical_ranks), nl.float32)
    global_neg_max = allocator.alloc_sbuf_tensor((P_MAX, 1), nl.float32)
    factors = allocator.alloc_sbuf_tensor((P_MAX, num_physical_ranks), nl.float32)
    exp_term = allocator.alloc_sbuf_tensor((P_MAX, num_physical_ranks), nl.float32)
    recip = allocator.alloc_sbuf_tensor((P_MAX, num_physical_ranks), nl.float32)
    factor_sum = allocator.alloc_sbuf_tensor((P_MAX, 1), nl.float32)
    factor_sum_recip = allocator.alloc_sbuf_tensor((P_MAX, 1), nl.float32)
    out_tile = allocator.alloc_sbuf_tensor((P_MAX, head_dim), nl.float32)
    partial_tile = allocator.alloc_sbuf_tensor((P_MAX, head_dim), nl.float32)
    scaled = allocator.alloc_sbuf_tensor((P_MAX, head_dim), nl.float32)
    out_tile_cast = allocator.alloc_sbuf_tensor((P_MAX, head_dim), out_dtype)
    # tp_out: extra SBUF buffers for transposing merged tile (P_MAX, head_dim) → (head_dim, P_MAX).
    out_tile_tp_psum = nl.ndarray((head_dim, P_MAX), dtype=nl.float32, buffer=nl.psum) if tp_out else None
    out_tile_tp_sbuf = allocator.alloc_sbuf_tensor((head_dim, P_MAX), out_dtype) if tp_out else None

    pos = shard_id  # this NC's position within q_heads_per_rank (0 or 1 for LNC=2)
    packed_dim = head_dim + 2
    rank_stride = q_heads_per_rank * seq_len * packed_dim

    num_tiles = div_ceil(seq_len, P_MAX)
    for tile_idx in range(num_tiles):
        tile_start = tile_idx * P_MAX
        tile_end = min(tile_start + P_MAX, seq_len)
        tile_size = tile_end - tile_start

        # Load neg_max and sum_recip for all ranks in one DMA each via AP.
        # Partition stride = packed_dim (tokens within a rank), free stride = rank_stride (across ranks).
        stats_base_offset = pos * seq_len * packed_dim + tile_start * packed_dim
        stats_ap = [[packed_dim, P_MAX], [rank_stride, num_physical_ranks]]
        nisa.dma_copy(dst=neg_max_sbuf, src=recv_packed.ap(pattern=stats_ap, offset=stats_base_offset + head_dim))
        nisa.dma_copy(dst=sum_recip_sbuf, src=recv_packed.ap(pattern=stats_ap, offset=stats_base_offset + head_dim + 1))

        # Compute per-rank rescaling factors using online softmax, then normalize.
        # exp(global_neg_max - neg_max_sbuf) = exp(-neg_max_sbuf + global_neg_max)
        nisa.tensor_reduce(
            dst=global_neg_max[:tile_size, :], op=nl.minimum, data=neg_max_sbuf[:tile_size, :], axis=[1], keepdims=True
        )
        nisa.activation(
            dst=exp_term[:tile_size, :],
            op=nl.exp,
            data=neg_max_sbuf[:tile_size, :],
            scale=-1.0,
            bias=global_neg_max[:tile_size, :],
        )
        nisa.activation(dst=recip[:tile_size, :], op=nl.reciprocal, data=sum_recip_sbuf[:tile_size, :])
        nisa.tensor_tensor(
            dst=factors[:tile_size, :], data1=exp_term[:tile_size, :], data2=recip[:tile_size, :], op=nl.multiply
        )
        nisa.tensor_reduce(
            dst=factor_sum[:tile_size, :], op=nl.add, data=factors[:tile_size, :], axis=[1], keepdims=True
        )
        nisa.activation(dst=factor_sum_recip[:tile_size, :], op=nl.reciprocal, data=factor_sum[:tile_size, :])
        nisa.activation(
            dst=factors[:tile_size, :], op=nl.copy, data=factors[:tile_size, :], scale=factor_sum_recip[:tile_size, :]
        )

        # Weighted sum: out = sum_r(factors[:, r] * partial[:, r, :]).
        # TODO: Vectorize via matmul (factors[tile, ranks] @ partial[ranks, head_dim]).
        # Blocked because partial data in recv_packed is interleaved with stats (stride=packed_dim=head_dim+2
        # between tokens), so loading all ranks' partial tiles requires a 3D AP — unsupported by the compiler.
        # Fix: switch to coalesced all-to-all with a separate contiguous recv_out tensor.
        nisa.memset(out_tile, 0)
        for rank_idx in range(num_physical_ranks):
            nisa.dma_copy(
                dst=partial_tile[:tile_size, :], src=recv_packed[rank_idx, pos, tile_start:tile_end, :head_dim]
            )
            nisa.tensor_scalar(
                dst=scaled[:tile_size, :],
                data=partial_tile[:tile_size, :],
                op0=nl.multiply,
                operand0=factors[:tile_size, rank_idx : rank_idx + 1],
            )
            nisa.tensor_tensor(
                dst=out_tile[:tile_size, :],
                data1=out_tile[:tile_size, :],
                data2=scaled[:tile_size, :],
                op=nl.add,
            )

        nisa.tensor_copy(dst=out_tile_cast[:tile_size, :], src=out_tile[:tile_size, :])
        if tp_out:
            # Transpose (P_MAX, head_dim) → (head_dim, P_MAX) for tp_out HBM layout. Perf cost: one
            # nc_transpose per tile + extra SBUF. Root cause: partial output and stats share a single
            # packed tensor (seq_len, head_dim+2), forcing merge to work in (seq_len, head_dim) space.
            # Coalesced all-to-all (multiple src/dst pairs in one call) would allow separate tensors
            # per type, enabling a zero-transpose end-to-end (head_dim, seq_len) tp_out path.
            nisa.nc_transpose(out_tile_tp_psum[:, :tile_size], out_tile[:tile_size, :])
            nisa.tensor_copy(dst=out_tile_tp_sbuf[:, :tile_size], src=out_tile_tp_psum[:, :tile_size])
            nisa.dma_copy(dst=out[shard_id, :, tile_start:tile_end], src=out_tile_tp_sbuf[:, :tile_size])
        else:
            nisa.dma_copy(dst=out[shard_id, tile_start:tile_end, :], src=out_tile_cast[:tile_size, :])
