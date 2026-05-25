# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from dataclasses import replace
from functools import lru_cache
from inspect import signature
from typing import Any, Optional, final

import neuron_dtypes as dt
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nki.collectives import ReplicaGroup
from typing_extensions import override

from nkilib_src.nkilib.core.attention.attention_tkg import INACTIVE_BLOCK_IDX
from nkilib_src.nkilib.core.utils.allocator import SbufManager
from nkilib_src.nkilib.core.utils.common_types import QuantizationType
from nkilib_src.nkilib.core.utils.kernel_helpers import (
    get_max_positive_value_for_dtype,
    get_program_sharding_info,
    is_hbm_buffer,
)
from nkilib_src.nkilib.core.utils.tensor_view import TensorView
from nkilib_src.nkilib.experimental.transformer.attention_block_tkg import (
    attention_block_tkg,
)
from nkilib_src.nkilib.experimental.transformer.attention_block_tkg_sharding import KVDPCollectiveMode
from nkilib_src.nkilib.experimental.transformer.attention_block_tkg_torch import (
    AttentionBlockTkgTorchRef,
)

try:
    from test.integration.nkilib.experimental.transformer.test_attention_block_tkg_model_config import (
        attention_block_tkg_model_configs,
    )
except ImportError:
    attention_block_tkg_model_configs = {}

from test.integration.nkilib.experimental.transformer.test_attention_block_tkg_utils import (
    AttnBlkTestConfig,
    KVScaleTest,
)
from test.integration.nkilib.utils.tensor_generators import (
    generate_stabilized_mx_data,
    np_random_sample_static_quantize_inp,
)
from test.utils.common_dataclasses import (
    TKG_INFERENCE_ARGS,
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    ModelTestType,
    PerRankLazyInputGenerator,
    Platforms,
    ValidationArgs,
)
from test.utils.comparators import maxAllClose
from test.utils.metadata_loader import load_model_configs
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import CollectiveUnitTestFramework, UnitTestFramework, torch_ref_wrapper

# Maximum memory (GB) for test tensor allocation. Tests exceeding this are skipped.
# Override via environment variable TEST_ATTN_BLK_TKG_MAX_MEMORY_GB.
_DEFAULT_MAX_MEMORY_GB = 20
_MAX_MEMORY_BYTES = int(float(os.environ.get("TEST_ATTN_BLK_TKG_MAX_MEMORY_GB", _DEFAULT_MAX_MEMORY_GB)) * 1024**3)

_P_MAX = 128  # Partition dimension size (nl.tile_size.pmax)
_FP8_FN_MAX = 448.0  # max representable value for float8_e4m3fn (used by MX on TRN3)


def _generate_mx_weights_and_scales(
    quantization_type, weight_shape, mx_scale_reshape, w_scale_shape, static_mx_in_shape, rng
):
    """Generate MX-quantized weights and dequantization scales.

    Args:
        quantization_type: One of MX, STATIC_MX, ROW_MX.
        weight_shape: Shape for generate_stabilized_mx_data (already divided by q_width).
        mx_scale_reshape: Reshape dims for MX block weight scales.
        w_scale_shape: Shape for STATIC_MX/ROW_MX jittered weight scales.
        static_mx_in_shape: Shape for STATIC_MX input scale jitter, or None to skip.
        rng: numpy random state.

    Returns:
        (weights, weight_scale, input_scale) where input_scale may be None.

    For STATIC_MX and ROW_MX, scales are jittered with a small random
    perturbation around the base value so that tests exercise non-uniform
    scale values and don't accidentally pass with a degenerate constant scale.
    """
    _, weights, weight_scale = generate_stabilized_mx_data(nl.float8_e4m3fn_x4, weight_shape, val_range=5)
    input_scale = None
    if quantization_type == QuantizationType.MX:
        weight_scale = weight_scale.reshape(mx_scale_reshape)
    elif quantization_type == QuantizationType.STATIC_MX:
        base_scale = 1.0 / _FP8_FN_MAX
        weight_scale = (base_scale * rng.uniform(0.995, 1.005, w_scale_shape)).astype(np.float32)
        if static_mx_in_shape is not None:
            input_scale = (base_scale * rng.uniform(0.995, 1.005, static_mx_in_shape)).astype(np.float32)
    elif quantization_type == QuantizationType.ROW_MX:
        base_scale = 1.0 / _FP8_FN_MAX
        weight_scale = (base_scale * rng.uniform(0.995, 1.005, w_scale_shape)).astype(np.float32)
    return weights, weight_scale, input_scale


def _slice_block_kv_for_all_ranks(
    full_K_cache: np.ndarray,
    full_V_cache: np.ndarray,
    full_active_blocks_table: np.ndarray,
    full_kv_cache_update_idx: np.ndarray,
    KVDP: int,
    B_attn: int,
    S_ctx: int,
    block_len: int,
    d_head: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Slice block KV cache for all ranks in KV data parallelism.

    vLLM treats each DP rank as an independent inference endpoint with its own KV cache.
    Each rank has a local block pool with indices local to that rank's cache, not global indices.

    The test generates golden outputs using a single global KV cache (no KV data parallelism),
    but the kernel under test runs per-rank with local caches. This function converts the global
    KV cache into the per-rank KV cache that vLLM provides.

    For each rank, this function:
    1. Slices active_blocks_table[B, num_active_blocks] on B to get this rank's batches
    2. Finds which global block indices this rank uses
    3. Creates a compact local cache with only this rank's blocks
    4. Remaps active_blocks_table from global to local indices

    Example with B=8, KVDP=4, B_attn=B/KVDP=2, S_ctx=1024, block_len=32:
        Global cache: K_cache[256, 32, d_head]  (256 = B * S_ctx // block_len = 8 * 1024 // 32)
        Rank 0 uses batches [0:2], references global blocks [100, 147, 212, 199, ...]
        After slicing: K_cache_0[64, 32, d_head] with local indices [0, 1, 2, 3, ..., 62, 63]
            (64 = B_attn * S_ctx // block_len = 2 * 1024 // 32)
        active_blocks_table remapped: [100, 147, 212, 199, ...] -> [0, 1, 2, 3, ...]

    Args:
        full_K_cache (np.ndarray): Global K cache [num_blocks, block_len, d_head]
        full_V_cache (np.ndarray): Global V cache [num_blocks, block_len, d_head]
        full_active_blocks_table (np.ndarray): Global block table [B, num_active_blocks]
        full_kv_cache_update_idx (np.ndarray): Global KV update indices [B, S_tkg]
        KVDP (int): KV data parallelism (number of ranks)
        B_attn (int): Batch size of KV cache per rank
        S_ctx (int): Context sequence length
        block_len (int): Block length
        d_head (int): Head dimension

    Returns:
        K_cache (list[np.ndarray]): Per-rank K caches, each [num_blocks/KVDP, block_len, d_head]
        V_cache (list[np.ndarray]): Per-rank V caches, each [num_blocks/KVDP, block_len, d_head]
        active_blocks_table (list[np.ndarray]): Remapped active_blocks_tables, each [B_attn, num_active_blocks]
        kv_cache_update_idx (list[np.ndarray]): Remapped KV update indices, each [B_attn, S_tkg]
    """
    # Validate input shapes
    assert full_K_cache.ndim == 3 and full_K_cache.shape[1:] == (block_len, d_head)
    assert full_V_cache.shape == full_K_cache.shape
    assert full_active_blocks_table.ndim == 2 and full_active_blocks_table.shape[0] == KVDP * B_attn
    assert full_kv_cache_update_idx.ndim == 2 and full_kv_cache_update_idx.shape[0] == KVDP * B_attn

    blocks_per_rank = B_attn * S_ctx // block_len
    K_cache, V_cache, active_blocks_table, kv_cache_update_idx = [], [], [], []

    for rank_idx in range(KVDP):
        # Slice active_blocks_table for this rank's batches
        rank_active_blocks_table = full_active_blocks_table[rank_idx * B_attn : (rank_idx + 1) * B_attn]

        # Find which global block indices this rank uses (excluding inactive sentinel)
        used_blocks = np.unique(rank_active_blocks_table[rank_active_blocks_table != INACTIVE_BLOCK_IDX])

        # Create compact local cache with only this rank's blocks
        new_k = np.zeros((blocks_per_rank, block_len, d_head), dtype=full_K_cache.dtype)
        new_v = np.zeros((blocks_per_rank, block_len, d_head), dtype=full_V_cache.dtype)

        # Copy block data and build global->local index mapping
        block_map = {}
        for new_idx, old_idx in enumerate(used_blocks):
            new_k[new_idx], new_v[new_idx] = full_K_cache[old_idx], full_V_cache[old_idx]
            block_map[old_idx] = new_idx

        # Remap active_blocks_table from global to local indices
        new_abt = rank_active_blocks_table.copy()
        for batch_idx in range(B_attn):
            for col_idx in range(rank_active_blocks_table.shape[1]):
                if rank_active_blocks_table[batch_idx, col_idx] != INACTIVE_BLOCK_IDX:
                    new_abt[batch_idx, col_idx] = block_map[rank_active_blocks_table[batch_idx, col_idx]]

        # Remap kv_cache_update_idx (block_idx * block_len + offset) for all S_tkg columns
        rank_kv_idx = full_kv_cache_update_idx[rank_idx * B_attn : (rank_idx + 1) * B_attn].copy()
        S_tkg = rank_kv_idx.shape[1]
        for batch_idx in range(B_attn):
            for token_idx in range(S_tkg):
                if rank_kv_idx[batch_idx, token_idx] != np.iinfo(np.uint32).max:
                    old_block, offset = divmod(int(rank_kv_idx[batch_idx, token_idx]), block_len)
                    rank_kv_idx[batch_idx, token_idx] = block_map.get(old_block, old_block) * block_len + offset

        K_cache.append(new_k)
        V_cache.append(new_v)
        active_blocks_table.append(new_abt)
        kv_cache_update_idx.append(rank_kv_idx)

    return K_cache, V_cache, active_blocks_table, kv_cache_update_idx


def _create_per_rank_inputs(
    base_input: dict, KVDP: int, q_heads: int, d_head: int, B_attn: int, S_tkg: int, S_ctx: int, block_len: int
) -> tuple:
    """Create per-rank kernel inputs for KV data parallelism tests.

    Slices kernel inputs per rank for multi-rank execution.
    Each rank gets a slice of the batch dimension for K/V cache and mask,
    while Q heads are distributed across ranks.

    Args:
        base_input (dict): Full kernel input dictionary with all tensors
        KVDP (int): KV data parallelism (number of ranks)
        q_heads (int): Number of query heads per rank
        d_head (int): Head dimension
        B_attn (int): Batch size of KV cache per rank (total_batch / KVDP)
        S_tkg (int): Token generation sequence length
        S_ctx (int): Context sequence length
        block_len (int): Block length for block KV cache (0 for flat cache)

    Returns:
        per_rank_input (PerRankLazyInputGenerator): Generator that creates inputs for each rank
        per_rank_cache (dict): Pre-sliced tensors for golden reference computation
    """
    total_q_heads = KVDP * q_heads

    replica_group = ReplicaGroup([list(range(KVDP))])

    # Slice W_qkv per rank: Q portion sliced (each rank gets q_heads), K/V replicated (GQA shares 1 KV head)
    full_W_qkv = base_input['W_qkv']
    q_end = total_q_heads * d_head
    k_end = q_end + d_head
    v_end = k_end + d_head
    W_q_full, W_k, W_v = full_W_qkv[:, :q_end], full_W_qkv[:, q_end:k_end], full_W_qkv[:, k_end:v_end]
    w_qkV_cache = [
        np.concatenate([W_q_full[:, rank_idx * q_heads * d_head : (rank_idx + 1) * q_heads * d_head], W_k, W_v], axis=1)
        for rank_idx in range(KVDP)
    ]

    # Slice bias_qkv if present
    full_bias_qkv = base_input.get('bias_qkv')
    if full_bias_qkv is not None:
        bias_q_full, bias_k, bias_v = (
            full_bias_qkv[:, :q_end],
            full_bias_qkv[:, q_end:k_end],
            full_bias_qkv[:, k_end:v_end],
        )
        bias_qkV_cache = [
            np.concatenate(
                [bias_q_full[:, rank_idx * q_heads * d_head : (rank_idx + 1) * q_heads * d_head], bias_k, bias_v],
                axis=1,
            )
            for rank_idx in range(KVDP)
        ]
    else:
        bias_qkV_cache = [None] * KVDP

    # Slice W_out per rank
    full_W_out = base_input.get('W_out')
    w_out_slices = (
        [
            full_W_out[rank_idx * q_heads * d_head : (rank_idx + 1) * q_heads * d_head, :].copy()
            for rank_idx in range(KVDP)
        ]
        if full_W_out is not None
        else [None] * KVDP
    )

    # Slice mask per rank
    # full_mask shape: (S_ctx, B, total_q_heads, S_tkg)
    # Kernel needs: (S_ctx, B_attn, total_q_heads, S_tkg) - sliced batch, all heads (after gather)
    # Golden needs: (S_ctx, B, q_heads, S_tkg) - full batch, 1 Q head
    full_mask = base_input['attention_mask']
    mask_slices = [full_mask[:, rank_idx * B_attn : (rank_idx + 1) * B_attn, :, :] for rank_idx in range(KVDP)]
    golden_mask = full_mask[:, :, :q_heads, :]

    # Slice pos_ids and swa_start_pos_ids per rank (batch dimension)
    # full shape: (B, S_tkg) -> per-rank: (B_attn, S_tkg)
    full_pos_ids = base_input.get('pos_ids')
    if full_pos_ids is not None:
        pos_ids_slices = [full_pos_ids[rank_idx * B_attn : (rank_idx + 1) * B_attn] for rank_idx in range(KVDP)]
    else:
        pos_ids_slices = [None] * KVDP
    full_swa_start_pos_ids = base_input.get('swa_start_pos_ids')
    if full_swa_start_pos_ids is not None:
        swa_start_pos_ids_slices = [
            full_swa_start_pos_ids[rank_idx * B_attn : (rank_idx + 1) * B_attn] for rank_idx in range(KVDP)
        ]
    else:
        swa_start_pos_ids_slices = [None] * KVDP

    # Slice K/V cache per rank
    full_K_cache, full_V_cache = base_input['K_cache'], base_input['V_cache']
    full_active_blocks_table = base_input.get('active_blocks_table')
    full_kv_cache_update_idx = base_input.get('kv_cache_update_idx')

    if block_len > 0:
        K_cache, V_cache, active_blocks_table, kv_cache_update_idx = _slice_block_kv_for_all_ranks(
            full_K_cache,
            full_V_cache,
            full_active_blocks_table,
            full_kv_cache_update_idx,
            KVDP,
            B_attn,
            S_ctx,
            block_len,
            d_head,
        )
    else:
        K_cache = [full_K_cache[rank_idx * B_attn : (rank_idx + 1) * B_attn] for rank_idx in range(KVDP)]
        V_cache = [full_V_cache[rank_idx * B_attn : (rank_idx + 1) * B_attn] for rank_idx in range(KVDP)]
        active_blocks_table = [None] * KVDP
        kv_cache_update_idx = [
            full_kv_cache_update_idx[rank_idx * B_attn : (rank_idx + 1) * B_attn] for rank_idx in range(KVDP)
        ]

    def create_per_rank_input(rank_id: int) -> dict:
        result = base_input.copy()
        result['W_qkv'] = w_qkV_cache[rank_id]
        result['bias_qkv'] = bias_qkV_cache[rank_id]
        result['W_out'] = w_out_slices[rank_id]
        result['K_cache'] = K_cache[rank_id]
        result['V_cache'] = V_cache[rank_id]
        result['attention_mask'] = mask_slices[rank_id]
        result['pos_ids'] = pos_ids_slices[rank_id]
        result['swa_start_pos_ids'] = swa_start_pos_ids_slices[rank_id]
        result['active_blocks_table'] = active_blocks_table[rank_id]
        result['kv_cache_update_idx'] = kv_cache_update_idx[rank_id]
        result['KVDP'] = KVDP
        result['KVDP_replica_group'] = replica_group
        return result

    per_rank_input = PerRankLazyInputGenerator(generator=create_per_rank_input)
    per_rank_input.base_input = base_input  # Store for golden reference

    per_rank_cache = {
        'w_qkv': w_qkV_cache,
        'bias_qkv': bias_qkV_cache,
        'w_out': w_out_slices,
        'golden_mask': golden_mask,
        'full_active_blocks_table': full_active_blocks_table,
    }
    return per_rank_input, per_rank_cache


def _slice_golden_KV_cache_for_rank(
    rank_golden: dict,
    rank_id: int,
    B_attn: int,
    block_len: int,
    d_head: int,
    S_ctx: int,
    per_rank_cache: dict,
    update_cache: bool,
) -> dict:
    """Slice golden K/V cache output for a specific rank.

    Golden computes with full batch B, so K/V outputs have shape [B, ...].
    This function slices to [B/KVDP, ...] to match the kernel's per-rank output.
    X_out is returned unchanged (already has correct shape from golden).

    When update_cache is False, the torch ref returns K_tkg/V_tkg (raw projected
    K/V) instead of K_cache_updated/V_cache_updated

    Args:
        rank_golden (dict): Full golden output with 'X_out' and either
            'K_cache_updated'/'V_cache_updated' (update_cache=True) or
            'K_tkg'/'V_tkg' (update_cache=False)
        rank_id (int): Rank index to slice for
        B_attn (int): Batch size of KV cache per rank (B/KVDP)
        block_len (int): Block length for block KV cache (0 for flat cache)
        d_head (int): Head dimension
        S_ctx (int): Context sequence length
        per_rank_cache (dict): Pre-computed slicing info including 'full_active_blocks_table'
        update_cache (bool): Whether cache update is enabled

    Returns:
        dict: Golden output with X_out unchanged, K/V sliced to per-rank batch size
    """
    if not update_cache:
        # K_tkg shape: [D, B, S_tkg], V_tkg shape: [B, S_tkg, D]
        golden_k = rank_golden['K_tkg'][:, rank_id * B_attn : (rank_id + 1) * B_attn, :]
        golden_v = rank_golden['V_tkg'][rank_id * B_attn : (rank_id + 1) * B_attn, :, :]
        return {'X_out': rank_golden['X_out'], 'K_tkg': golden_k, 'V_tkg': golden_v}

    if block_len > 0:
        blocks_per_rank = B_attn * S_ctx // block_len
        full_active_blocks_table = per_rank_cache['full_active_blocks_table']
        rank_active_blocks_table = full_active_blocks_table[rank_id * B_attn : (rank_id + 1) * B_attn]
        used_blocks = np.unique(rank_active_blocks_table[rank_active_blocks_table != INACTIVE_BLOCK_IDX])

        golden_k = np.zeros((blocks_per_rank, block_len, d_head), dtype=rank_golden['K_cache_updated'].dtype)
        golden_v = np.zeros((blocks_per_rank, block_len, d_head), dtype=rank_golden['V_cache_updated'].dtype)
        for new_idx, old_idx in enumerate(used_blocks):
            golden_k[new_idx] = rank_golden['K_cache_updated'][old_idx]
            golden_v[new_idx] = rank_golden['V_cache_updated'][old_idx]
    else:
        golden_k = rank_golden['K_cache_updated'][rank_id * B_attn : (rank_id + 1) * B_attn]
        golden_v = rank_golden['V_cache_updated'][rank_id * B_attn : (rank_id + 1) * B_attn]

    return {'X_out': rank_golden['X_out'], 'K_cache_updated': golden_k, 'V_cache_updated': golden_v}


def estimate_test_memory_bytes(cfg: AttnBlkTestConfig) -> int:
    """Estimate total host memory (bytes) for a test config without allocating tensors.

    Computes the sum of all tensor sizes created during the test.
    """
    batch = cfg.batch
    num_heads = cfg.q_heads
    d_head = cfg.d_head
    H = cfg.H
    S_ctx = cfg.S_ctx
    S_max_ctx = cfg.S_max_ctx
    S_tkg = cfg.S_tkg
    block_len = cfg.block_len
    kv_quant = cfg.kv_quant
    KVDP = cfg.KVDP
    quantization_type = cfg.quantization_type
    skip_output_projection = cfg.skip_output_projection

    is_quantized = quantization_type != QuantizationType.NONE
    elem = 1 if is_quantized else 2  # fp8=1, bf16=2
    kv_elem = 1 if kv_quant else elem
    is_block_kv = block_len > 0

    # KVDP inflates num_heads for input generation (mirrors _run_attention_block_test)
    effective_heads = KVDP * num_heads if KVDP > 1 else num_heads
    I = d_head * (effective_heads + 2)  # num_kv_heads=1 always

    total = 0

    # --- generate_kernel_inputs ---
    total += batch * S_tkg * H * elem  # X
    total += H * I * elem  # W_qkv
    if cfg.rmsnorm_X:
        total += H * elem  # rmsnorm_X_gamma
    if cfg.test_bias:
        total += I * elem  # bias_qkv
        total += H * elem  # bias_out
    if not cfg.skip_rope:
        total += 2 * (d_head // 2) * batch * S_tkg * elem  # cos + sin
    if cfg.qk_norm_pre_rope_gamma:
        total += 2 * d_head * elem  # W_rmsnorm_Q/K_pre_rope
    if cfg.qk_norm_post_rope_gamma:
        total += 2 * d_head * elem  # W_rmsnorm_Q/K_post_rope

    # KV cache
    if is_block_kv:
        num_blocks = batch * S_ctx // block_len
        total += 2 * num_blocks * block_len * d_head * kv_elem  # K + V cache
        total += batch * (S_ctx // block_len) * 4  # active_blocks_table (int32)
    else:
        total += 2 * batch * S_max_ctx * d_head * kv_elem  # K + V cache

    total += S_ctx * batch * effective_heads * S_tkg  # attention_mask (uint8)
    total += batch * 4  # kv_cache_update_idx (uint32)
    total += batch * 8  # cache_len (int64)

    if not skip_output_projection:
        total += effective_heads * d_head * H * elem  # W_out
    if kv_quant:
        total += 4 + 128 * 4  # k_scale + v_scale (float32)
    if is_quantized:
        total += 128 * 3 * 4 + 128 * 4  # weight/input dequant scales qkv
        if not skip_output_projection:
            total += 2 * 128 * 4  # weight/input dequant scales out

    # --- KVDP per-rank copies (_create_per_rank_inputs) ---
    if KVDP > 1:
        B_attn = batch // KVDP
        if is_block_kv:
            rank_blocks = B_attn * S_ctx // block_len
            total += KVDP * 2 * rank_blocks * block_len * d_head * kv_elem
            total += KVDP * B_attn * (S_ctx // block_len) * 4
        else:
            total += KVDP * 2 * B_attn * S_max_ctx * d_head * kv_elem
        total += KVDP * S_ctx * B_attn * effective_heads * S_tkg  # per-rank masks
        total += KVDP * H * d_head * (num_heads + 2) * elem  # per-rank W_qkv
        if not skip_output_projection:
            total += KVDP * num_heads * d_head * H * elem  # per-rank W_out

    # --- Golden reference overhead (float32 intermediates) ---
    total += batch * S_tkg * H * 4  # X as f32
    total += H * I * 4  # W_qkv as f32
    total += batch * effective_heads * S_tkg * d_head * 4  # QKV output
    total += 2 * batch * effective_heads * S_tkg * S_ctx * 4  # attn scores + softmax
    total += batch * effective_heads * S_tkg * d_head * 4  # attn output
    if not skip_output_projection:
        total += batch * S_tkg * H * 4  # output projection

    # --- output_placeholder (zeros_like golden) ---
    total += batch * S_tkg * H * elem  # X_out
    if not is_block_kv:
        total += 2 * batch * S_max_ctx * d_head * kv_elem  # K/V out placeholders

    return total


def _generate_kv_quant_inputs(
    cfg,
    _rng,
    H,
    num_q_heads,
    num_kv_heads,
    d_head,
    weight_dequant_scale_qkv,
    input_dequant_scale_qkv,
    K_cache_shape,
    V_cache_shape,
    kv_quant_dtype,
):
    """Derive kv_scale from std(K_active) and generate quantized KV cache.

    See attention_block_tkg_input_distributions_design_spec.md for full analysis.
    """
    _QUANT_DTYPE_MAX = get_max_positive_value_for_dtype(kv_quant_dtype)
    _COVERAGE = 4.0  # 4σ coverage: ~0.006% Gaussian clipping rate

    if cfg.kv_scale is KVScaleTest.DEFAULT:
        # Derive kv_scale from std(K_active) after QKV projection.
        # K = X_norm @ W where W has variance 1/H (fan-in scaled).
        # Var(K) = H × Var(X_norm) × Var(W) = H × Var(X_norm) × (1/H) = Var(X_norm)
        # Without RMSNorm: Var(X) = Var(Uniform[-1,1]) = (1-(-1))²/12 = 1/3, std ≈ 0.577
        # With RMSNorm: Var(X_norm) ≈ 1.0, std ≈ 1.0
        # kv_scale = FP8_max / (coverage × std(K_active))
        # e.g. without RMSNorm: max ≈ 4×std ≈ 4×0.577 ≈ 2.3, scale = 240/2.3 ≈ 104
        # MX micro-scaling preserves the input distribution (no per-tensor clipping),
        # so MX output variance matches the unquantized (NONE) path.
        if cfg.quantization_type == QuantizationType.NONE or cfg.quantization_type.is_mx():
            x_var = 1.0 if cfg.rmsnorm_X else 1.0 / 3.0
            k_active_std = np.sqrt(x_var)
            v_active_std = k_active_std
        elif cfg.quantization_type == QuantizationType.ROW:
            x_var = 1.0 if cfg.rmsnorm_X else 1.0 / 3.0
            k_active_std = np.sqrt(x_var)
            v_active_std = k_active_std
            if weight_dequant_scale_qkv is not None:
                num_q = num_q_heads * d_head
                k_active_std *= float(weight_dequant_scale_qkv[0, num_q : num_q + d_head].max()) / float(
                    weight_dequant_scale_qkv[0, num_q : num_q + d_head].mean()
                )
                v_active_std *= float(weight_dequant_scale_qkv[0, num_q + d_head : num_q + 2 * d_head].max()) / float(
                    weight_dequant_scale_qkv[0, num_q + d_head : num_q + 2 * d_head].mean()
                )
        elif cfg.quantization_type == QuantizationType.STATIC:
            # With calibrated in_scale, X_fp8 is Gaussian.
            # K_active = (X_fp8 @ W_fp8) × w_scale × in_scale.
            # With fan-in-calibrated w_scale and coverage-calibrated in_scale:
            # std(K_active) ≈ std(X_norm) ≈ 1.0 (with RMSNorm) or 0.577 (without).
            x_var = 1.0 if cfg.rmsnorm_X else 1.0 / 3.0
            k_active_std = np.sqrt(x_var)
            v_active_std = k_active_std
        else:
            raise ValueError(f"Unsupported quantization type: {cfg.quantization_type}")
        # QK norm (pre or post RoPE) normalizes K to unit RMS over d_head,
        # overriding the projection std. After norm: std(K) ≈ mean(gamma) ≈ 1.0
        # V is not affected by QK norm.
        if cfg.qk_norm_pre_rope or cfg.qk_norm_post_rope:
            k_active_std = 1.0
        k_scale_scalar = _QUANT_DTYPE_MAX / (_COVERAGE * k_active_std)
        v_scale_scalar = _QUANT_DTYPE_MAX / (_COVERAGE * v_active_std)
    else:
        assert isinstance(cfg.kv_scale, float), f"kv_scale must be KVScaleTest.DEFAULT or a float, got {cfg.kv_scale}"
        k_scale_scalar = cfg.kv_scale
        v_scale_scalar = cfg.kv_scale
        k_active_std = _QUANT_DTYPE_MAX / (_COVERAGE * k_scale_scalar)
        v_active_std = k_active_std

    # Use different shapes to test both broadcast (1,1) and per-partition (PMAX,1) paths
    k_scale = np.full((1, 1), k_scale_scalar, dtype=np.float32)
    v_scale = np.full((128, 1), v_scale_scalar, dtype=np.float32)

    # Generate KV cache matching the distribution of scaled K/V:
    # Cache stores K*scale, so std in cache = std(K) × scale = dtype_max/coverage
    np.random.seed(42)
    k_cache_std = k_active_std * k_scale_scalar
    v_cache_std = v_active_std * v_scale_scalar
    K_cache_f32 = np.random.normal(0, k_cache_std, K_cache_shape).astype(np.float32)
    V_cache_f32 = np.random.normal(0, v_cache_std, V_cache_shape).astype(np.float32)
    K_cache = dt.static_cast(np.clip(K_cache_f32, -_QUANT_DTYPE_MAX, _QUANT_DTYPE_MAX), kv_quant_dtype)
    V_cache = dt.static_cast(np.clip(V_cache_f32, -_QUANT_DTYPE_MAX, _QUANT_DTYPE_MAX), kv_quant_dtype)

    return k_scale, v_scale, K_cache, V_cache


def generate_kernel_inputs(cfg: AttnBlkTestConfig):
    from nkilib_src.nkilib.core.attention.gen_mask_tkg_torch import build_full_attention_mask
    from test.integration.nkilib.core.attention.test_attention_tkg_utils import (
        build_active_attention_mask,
        build_swa_positions,
        gen_deterministic_active_block_table,
        generate_cache_lens,
    )

    # Short aliases for dimensions used repeatedly in shape expressions
    dtype = cfg.dtype
    batch, d_head, H = cfg.batch, cfg.d_head, cfg.H
    S_ctx, S_max_ctx, S_tkg = cfg.S_ctx, cfg.S_max_ctx, cfg.S_tkg
    H_actual = cfg.H_actual if cfg.H_actual is not None else H
    num_q_heads = cfg.q_heads
    num_kv_heads = 1

    eps = 1e-5 if dtype == np.float32 else 1e-3

    # FP8 constants used across quantization paths
    _COVERAGE = 4.0  # 4σ coverage: ~0.006% Gaussian clipping rate
    _weight_fp8_dtype = nl.float8_e4m3  # Weight quantization dtype (independent of KV cache dtype)

    # ── Tensor generators ──────────────────────────────────────────────────────
    #
    # bf16 has 7 mantissa bits → 1 ULP = 2⁻⁷ relative to the significand.
    # With round-to-nearest, max error is ½ ULP = 2⁻⁸. Worst case occurs at the
    # bottom of each exponent bucket (significand=1.0): 2⁻⁸/1.0 = 1/256 ≈ 0.4%.
    # This is the worst-case relative error when quantizing f32 to bf16.
    #
    # With Gaussian(0,1) weights, QKV projections have std ≈ √H, so attention
    # scores (QK^T) have std ≈ H (thousands). When two positions score almost
    # identically, 0.4% noise can flip which one wins:
    #
    #   f32 scores:  [2001.0, 1998.5, 1950.0, 1870.0]
    #   bf16 noise:  [  -8.0,   +6.0,   -3.0,   +1.0]   (~0.4% of 2000)
    #   bf16 scores: [1993.0, 2004.5, 1947.0, 1871.0]   ← position 1 now wins
    #
    #   After softmax subtracts max:
    #     f32:  [  0.0,  -2.5, ...]  → exp → position 0 gets 92%
    #     bf16: [-11.5,   0.0, ...]  → exp → position 1 gets 99.99%
    #
    # The fundamental problem: softmax exponentiates the *differences* (after
    # subtracting max), but bf16 noise scales with the *magnitudes*. When noise
    # exceeds the difference signal, the ranking can flip. At large scale,
    # softmax behaves as argmax and picks a completely wrong V row. This causes
    # random heads to fail — it's statistical, depending on which positions
    # happen to score nearly identically.
    #
    # At small scale (scores ≈ 1), centered values are near zero where exp() is
    # flat, so even if a flip occurs, the probabilities barely change.
    #
    # To keep scores O(1), we use W ~ N(0, σ=1/√fan_in). This scaling is "unitary"
    # in the sense that it preserves variance through matmuls. Gammas, cos/sin,
    # and other tensors are chosen similarly to keep std ≈ 0.5–1.0 up to softmax.
    #
    # Notes:
    # - Not all test configs use normalization, so we can't rely on it alone.
    # - Large biases can mask attention bugs; small biases can be masked by them.
    # - Configs vary (RMSNorm, QK-norm, RoPE, bias on/off), but std ≈ 0.5–1.0
    #   with reasonable biases works across all of them.
    # - This is close to typical weight initialization in training.
    # - After softmax, larger weights are fine — the peaky regime is past.
    #
    # See attention_block_tkg_input_distributions_design_spec.md for full analysis.
    _rng = np.random.default_rng(0)

    def uniform_activation(shape, dtype):
        """Uniform[-1, 1]"""
        return np.ascontiguousarray(dt.static_cast(_rng.uniform(-1.0, 1.0, shape).astype(np.float32), dtype))

    def gaussian(shape, dtype, std):
        """N(0, std)"""
        return np.ascontiguousarray(dt.static_cast(_rng.normal(0.0, std, shape).astype(np.float32), dtype))

    def fan_in_projection(shape, dtype, fan_in):
        """N(0, σ=1/√fan_in). Keeps matmul output variance ≈ input variance."""
        return gaussian(shape, dtype, std=1.0 / np.sqrt(fan_in))

    def near_unity(shape, dtype):
        """Uniform[0.5, 1.5]. RMSNorm gammas are ~1.0 in trained models."""
        return np.ascontiguousarray(dt.static_cast(_rng.uniform(0.5, 1.5, shape).astype(np.float32), dtype))

    def small_bias(shape, dtype):
        """Uniform[-0.1, 0.1]. Biases are small in trained models."""
        return np.ascontiguousarray(dt.static_cast(_rng.uniform(-0.1, 0.1, shape).astype(np.float32), dtype))

    generate_quant_tensor = np_random_sample_static_quantize_inp()

    # -- input: post-layernorm activations are O(1)
    X = uniform_activation((batch, S_tkg, H), dtype)
    X[:, :, H_actual:] = 0.0

    # If transposed_in, convert X from [B, S, H] to [H0, n_prgs, H1_shard, BxS]
    # using (lnc, h0, h1) decomposition: h = lnc * H_per_shard + h0 * H1_shard + h1
    # numpy: flat.reshape(BxS, n_prgs, H0, H1_shard).transpose(2, 1, 3, 0)
    if cfg.transposed_in:
        n_prgs = cfg.lnc
        H0 = nl.tile_size.pmax
        H1_shard = H // n_prgs // H0
        BxS = batch * S_tkg
        X_transposed = X.reshape(BxS, n_prgs, H0, H1_shard).transpose(2, 1, 3, 0)
        X_transposed = np.ascontiguousarray(X_transposed)
        X = X_transposed

    # -- rmsnorm X: gamma weights are ~1.0 in trained models
    rmsnorm_X_gamma = near_unity((1, H), dtype) if cfg.rmsnorm_X else None

    # -- qkv projections, optional bias
    dim_I_qkv = (num_q_heads + 2 * num_kv_heads) * d_head
    if cfg.quantization_type == QuantizationType.NONE:
        # W_qkv: projection from hidden dim H → QKV, fan-in-scaled by fan_in=H
        W_qkv = fan_in_projection((H, dim_I_qkv), dtype, fan_in=H)
        weight_dequant_scale_qkv = None
        input_dequant_scale_qkv = None
    elif cfg.quantization_type == QuantizationType.ROW:
        # fan_in=H calibrates per-row w_scale for variance-preserving projection.
        # See attention_block_tkg_input_distributions_design_spec.md for full analysis.
        W_qkv, weight_dequant_scale_qkv, _ = generate_quant_tensor(
            shape=(H, dim_I_qkv), dtype=_weight_fp8_dtype, granularity="row", fan_in=H
        )
        weight_dequant_scale_qkv = np.broadcast_to(weight_dequant_scale_qkv, (128, dim_I_qkv))
        input_dequant_scale_qkv = None
    elif cfg.quantization_type == QuantizationType.STATIC:
        # fan_in=H calibrates w_scale so W_fp8 × w_scale has variance 1/H (variance-preserving).
        # in_scale is calibrated so X_fp8 is Gaussian: in_scale = coverage × std(X) / FP8_MAX.
        # With rmsnorm_X: std(X_norm) ≈ 1.0. Without: std(X) = std(Uniform[-1,1]) = 1/√3 ≈ 0.577.
        W_q, w_scale_q, _ = generate_quant_tensor(shape=(H, num_q_heads * d_head), dtype=_weight_fp8_dtype, fan_in=H)
        W_k, w_scale_k, _ = generate_quant_tensor(shape=(H, num_kv_heads * d_head), dtype=_weight_fp8_dtype, fan_in=H)
        W_v, w_scale_v, _ = generate_quant_tensor(shape=(H, num_kv_heads * d_head), dtype=_weight_fp8_dtype, fan_in=H)
        W_qkv = np.concatenate([W_q, W_k, W_v], axis=1)
        weight_dequant_scale_qkv = np.array([[w_scale_q, w_scale_k, w_scale_v]])
        weight_dequant_scale_qkv = np.broadcast_to(weight_dequant_scale_qkv, (128, 3))
        _x_std = 1.0 if cfg.rmsnorm_X else np.sqrt(1.0 / 3.0)
        _qkv_quant_max = get_max_positive_value_for_dtype(
            _weight_fp8_dtype
        )  # matches dtype in generate_quant_tensor calls above
        _in_scale = np.float32(_COVERAGE * _x_std / _qkv_quant_max * _rng.uniform(0.8, 1.2))
        input_dequant_scale_qkv = np.broadcast_to(_in_scale.reshape(1, 1), (128, 1))
    elif cfg.quantization_type.is_mx():
        _q_width = 4
        W_qkv, weight_dequant_scale_qkv, input_dequant_scale_qkv = _generate_mx_weights_and_scales(
            cfg.quantization_type,
            weight_shape=(H // _q_width, dim_I_qkv * _q_width),
            mx_scale_reshape=(H // 32, dim_I_qkv),
            w_scale_shape=(1, 3) if cfg.quantization_type == QuantizationType.STATIC_MX else (1, dim_I_qkv),
            static_mx_in_shape=(1, 1),
            rng=_rng,
        )
        # Pre-shuffle X along H for MXFP hardware layout: [B,S,H//512,128,4] → [B,S,4,H//512,128]
        X = np.ascontiguousarray(
            X.reshape(batch, S_tkg, H // (_P_MAX * _q_width), _P_MAX, _q_width)
            .transpose(0, 1, 4, 2, 3)
            .reshape(batch, S_tkg, H)
        )
        # Pre-shuffle RMSNorm gamma to match shuffled H layout
        if rmsnorm_X_gamma is not None:
            rmsnorm_X_gamma = np.ascontiguousarray(
                rmsnorm_X_gamma.reshape(1, H // (_P_MAX * _q_width), _P_MAX, _q_width)
                .transpose(0, 3, 1, 2)
                .reshape(1, H)
            )
    else:
        raise ValueError(f"Unsupported quantization type: {cfg.quantization_type}")
    bias_qkv = small_bias((1, (num_q_heads + 2 * num_kv_heads) * d_head), dtype) if cfg.test_bias else None

    # -- rmsnorm QK pre RoPE gamma weights
    W_rmsnorm_Q_pre_rope = near_unity((1, d_head), dtype) if cfg.qk_norm_pre_rope_gamma else None
    W_rmsnorm_K_pre_rope = near_unity((1, d_head), dtype) if cfg.qk_norm_pre_rope_gamma else None
    # -- RoPE: cos/sin are bounded to [-1, 1] by definition
    cos = None if cfg.skip_rope else uniform_activation((d_head // 2, batch, S_tkg), dtype)
    sin = None if cfg.skip_rope else uniform_activation((d_head // 2, batch, S_tkg), dtype)

    # -- rmsnorm QK post RoPE
    W_rmsnorm_Q_post_rope = near_unity((1, d_head), dtype) if cfg.qk_norm_post_rope_gamma else None
    W_rmsnorm_K_post_rope = near_unity((1, d_head), dtype) if cfg.qk_norm_post_rope_gamma else None

    # -- Attention (and KV cache)
    is_block_kv = cfg.block_len > 0

    # Determine cache shapes
    if is_block_kv:
        assert not cfg.K_cache_transposed
        assert S_ctx % cfg.block_len == 0
        assumed_num_cache_blocks = batch * S_ctx // cfg.block_len
        K_cache_shape = V_cache_shape = (assumed_num_cache_blocks, cfg.block_len, d_head)
    else:
        assumed_num_cache_blocks = 0
        K_cache_shape = (batch, 1, d_head, S_max_ctx) if cfg.K_cache_transposed else (batch, 1, S_max_ctx, d_head)
        V_cache_shape = (batch, 1, S_max_ctx, d_head)

    # Generate KV cache in FP8 when kv_quant=True
    kv_cache_dtype = cfg.kv_quant_dtype if cfg.kv_quant else dtype
    _CACHE_DTYPE_MAX = get_max_positive_value_for_dtype(kv_cache_dtype)
    if cfg.kv_quant:
        # Determine KV scale and generate FP8 cache
        k_scale, v_scale, K_cache, V_cache = _generate_kv_quant_inputs(
            cfg,
            _rng,
            H,
            num_q_heads,
            num_kv_heads,
            d_head,
            weight_dequant_scale_qkv,
            input_dequant_scale_qkv,
            K_cache_shape,
            V_cache_shape,
            kv_cache_dtype,
        )
    else:
        # KV cache stores projected K/V values which are O(1) after fan-in-scaled projection
        K_cache = uniform_activation(K_cache_shape, kv_cache_dtype)
        V_cache = uniform_activation(V_cache_shape, kv_cache_dtype)
        k_scale = None
        v_scale = None

    # pos_id (shape=(batch, 1)) defines the first position to append new KV to cache, per batch element
    cache_len_kwargs = {}
    if cfg.cache_lens_mean is not None:
        cache_len_kwargs["mean_frac"] = cfg.cache_lens_mean
    if cfg.cache_lens_stddev is not None:
        cache_len_kwargs["stddev_frac"] = cfg.cache_lens_stddev
    cache_len = generate_cache_lens(batch, S_ctx, S_tkg, **cache_len_kwargs)
    assert cache_len.max() <= (S_ctx - S_tkg)
    import torch

    cache_lens_torch = torch.from_numpy(cache_len.flatten()).to(torch.float32)

    if cfg.use_pos_id:
        # In-kernel mask generation: pass pos_ids instead of attention_mask
        if cfg.sliding_window > 0:
            swa_start_pos_ids, pos_ids = build_swa_positions(
                pos_id=cache_len,
                bs=batch,
                s_active=S_tkg,
                sliding_window=cfg.sliding_window,
                cache_len=S_ctx,
                block_len=cfg.block_len,
            )
        else:
            pos_ids = np.broadcast_to(cache_len, (batch, S_tkg)).astype(np.float32)
            if S_tkg > 1:
                pos_ids = pos_ids + np.arange(S_tkg, dtype=np.float32)[np.newaxis, :]
            swa_start_pos_ids = None

        attention_mask = (
            build_active_attention_mask(
                batch=batch,
                num_heads=cfg.q_heads,
                s_active=S_tkg,
                transposed=True,
            )
            .numpy()
            .astype(np.uint8)
        )
    else:
        pos_ids = None
        swa_start_pos_ids = None
        attention_mask = build_full_attention_mask(
            cache_lens=cache_lens_torch,
            batch=batch,
            num_heads=cfg.q_heads,
            s_active=S_tkg,
            s_ctx=S_ctx,
            lnc=cfg.lnc,
            block_len=cfg.block_len,
            include_active_mask=True,
            transposed=True,
            enable_fa_s_prior_tiling=cfg.enable_fa_s_prior_tiling,
            KVDP=cfg.KVDP,
        ).numpy()  # mask: (S_ctx, batch, num_heads, S_tkg)
        attention_mask = dt.static_cast(np.ascontiguousarray(attention_mask), dtype=np.uint8)

    active_blocks_table = (
        gen_deterministic_active_block_table(
            batch, S_ctx, S_tkg, cache_len, cfg.block_len, batch * S_ctx // cfg.block_len
        ).astype(np.int32)
        if is_block_kv
        else None
    )  # (B, S_ctx // block_len)

    # kv_cache_update_idx: (B, S_tkg) for block KV with per-token physical positions,
    #                      (B, 1) for flat KV with start position (consecutive tokens assumed)
    def generate_kv_cache_update_idx():
        if cfg.block_len == 0:
            # Flat KV: only start position needed, consecutive tokens assumed
            return cache_len.astype(np.uint32)

        # Block KV: translate each token's logical position to physical slot_mapping
        # Expand cache_len (B, 1) to per-token logical positions (B, S_tkg)
        logical_positions = cache_len + np.arange(S_tkg)  # (B, S_tkg)
        logical_blks = logical_positions // cfg.block_len
        offset_in_blk = logical_positions % cfg.block_len
        physical_blks = active_blocks_table[np.arange(batch)[:, None], logical_blks]
        physical_kv_cache_update_idx = physical_blks * cfg.block_len + offset_in_blk
        # Mask update for last batch element to test scenario when it is just padding
        if batch > 1:
            physical_kv_cache_update_idx[-1, :] = -1
        return physical_kv_cache_update_idx.astype(np.uint32)

    kv_cache_update_idx = generate_kv_cache_update_idx()

    # Output projection
    weight_dequant_scale_out = None
    input_dequant_scale_out = None
    if cfg.skip_output_projection:
        W_out = None
    elif cfg.quantization_type == QuantizationType.NONE:
        # W_out: projection from attention output → H. After softmax, probs@V has low std.
        # Use std=0.5 to scale output back up so bias is meaningful (not at noise level).
        W_out = gaussian((cfg.q_heads * d_head, H), dtype, std=0.5)
    elif cfg.quantization_type == QuantizationType.ROW:
        # fan_in for variance-preserving per-row w_scale.
        W_out, weight_dequant_scale_out, _ = generate_quant_tensor(
            shape=(cfg.q_heads * d_head, H), dtype=_weight_fp8_dtype, granularity="row", fan_in=cfg.q_heads * d_head
        )
        weight_dequant_scale_out = np.broadcast_to(weight_dequant_scale_out, (128, H))
        input_dequant_scale_out = None
    elif cfg.quantization_type == QuantizationType.STATIC:
        _fan_in_out = cfg.q_heads * d_head
        _out_proj_quant_dtype = _weight_fp8_dtype
        W_out, weight_dequant_scale_out, _ = generate_quant_tensor(
            shape=(_fan_in_out, H), dtype=_out_proj_quant_dtype, fan_in=_fan_in_out
        )
        # Calibrate input_dequant_scale_out to attention output magnitude.
        # With kv_quant: attn_out ≈ softmax @ V_cache_fp8, std ≈ FP8_MAX/_COVERAGE ≈ 60
        # Without kv_quant: attn_out ≈ softmax @ V_cache_bf16, std ≈ 0.577
        # Jitter (0.8-1.2×) tests robustness to imperfect calibration.
        _out_proj_quant_max = get_max_positive_value_for_dtype(_out_proj_quant_dtype)
        _attn_out_std = _CACHE_DTYPE_MAX / _COVERAGE if cfg.kv_quant else np.sqrt(1.0 / 3.0)
        _jitter = _rng.uniform(0.8, 1.2)
        input_dequant_scale_out = np.float32(_COVERAGE * _attn_out_std / _out_proj_quant_max * _jitter)
        weight_dequant_scale_out = np.broadcast_to(weight_dequant_scale_out.reshape(1, 1), (128, 1))
        input_dequant_scale_out = np.broadcast_to(input_dequant_scale_out.reshape(1, 1), (128, 1))
    elif cfg.quantization_type.is_mx():
        _q_width = 4
        N_D = cfg.q_heads * d_head
        W_out, weight_dequant_scale_out, input_dequant_scale_out = _generate_mx_weights_and_scales(
            cfg.quantization_type,
            weight_shape=(N_D // _q_width, H * _q_width),
            mx_scale_reshape=(N_D // 32, H),
            w_scale_shape=(1, 1) if cfg.quantization_type == QuantizationType.STATIC_MX else (1, H),
            static_mx_in_shape=(1, 1),
            rng=_rng,
        )
        # Output projection STATIC_MX scales need broadcast to (128, 1) for kernel interface
        if cfg.quantization_type == QuantizationType.STATIC_MX:
            weight_dequant_scale_out = np.broadcast_to(weight_dequant_scale_out, (128, 1)).copy()
            input_dequant_scale_out = np.broadcast_to(input_dequant_scale_out, (128, 1)).copy()
    else:
        raise ValueError(f"Unsupported quantization type: {cfg.quantization_type}")

    # bias_out: match the output projection scale (~0.1) so bias doesn't dominate.
    bias_out = small_bias((1, H), dtype) if cfg.test_bias else None

    # ── FP8 KV cache scale fusion ──────────────────────────────────────────
    # The kernel operates on raw FP8 KV values (K*k_scale, V*v_scale) without
    # dequantizing. When softmax_scale is None, the kernel automatically fuses
    # k_scale. When softmax_scale is explicit, the caller must fuse k_scale.
    # The caller must always fuse v_scale into W_out.
    softmax_scale_adjusted = cfg.softmax_scale
    if cfg.kv_quant:
        _k_scale_scalar = float(k_scale.flat[0])
        _v_scale_scalar = float(v_scale.flat[0])

        # Only fuse k_scale into softmax_scale when explicitly provided;
        # the kernel handles the None case automatically.
        if cfg.softmax_scale is not None:
            softmax_scale_adjusted = cfg.softmax_scale / _k_scale_scalar

        if W_out is not None:
            if cfg.quantization_type == QuantizationType.NONE:
                # bf16 weights: divide directly
                W_out = dt.static_cast(W_out.astype(np.float32) / _v_scale_scalar, dtype)
            elif cfg.quantization_type in (QuantizationType.ROW, QuantizationType.STATIC):
                # FP8 weights with dequant scale: absorb v_scale into the scale
                weight_dequant_scale_out = (weight_dequant_scale_out.astype(np.float32) / _v_scale_scalar).astype(
                    np.float32
                )

    return {
        # -- input
        "X": X,
        "X_in_sb": cfg.input_in_sb,
        "X_hidden_dim_actual": H_actual,
        # -- rmsnorm X
        "rmsnorm_X_enabled": cfg.rmsnorm_X,
        "rmsnorm_X_eps": eps,
        "rmsnorm_X_gamma": rmsnorm_X_gamma,
        # -- qkv projections
        "W_qkv": W_qkv,
        "bias_qkv": bias_qkv,
        "quantization_type_qkv": cfg.quantization_type,
        "weight_dequant_scale_qkv": weight_dequant_scale_qkv,
        "input_dequant_scale_qkv": input_dequant_scale_qkv,
        # -- QK rmsnorm pre RoPE
        "rmsnorm_QK_enabled": cfg.qk_norm_pre_rope,
        "rmsnorm_QK_eps": eps,
        "W_rmsnorm_Q_pre_rope": W_rmsnorm_Q_pre_rope,
        "W_rmsnorm_K_pre_rope": W_rmsnorm_K_pre_rope,
        # -- RoPE
        "cos": cos,
        "sin": sin,
        "rope_contiguous_layout": cfg.rope_contiguous_layout,
        # -- QK rmsnorm post RoPE
        "rmsnorm_QK_post_rope_enabled": cfg.qk_norm_post_rope,
        "rmsnorm_QK_post_rope_eps": eps,
        "W_rmsnorm_Q_post_rope": W_rmsnorm_Q_post_rope,
        "W_rmsnorm_K_post_rope": W_rmsnorm_K_post_rope,
        # -- attention
        "skip_attention": cfg.skip_attention,
        "K_cache_transposed": cfg.K_cache_transposed,
        "active_blocks_table": active_blocks_table,
        "K_cache": K_cache,
        "V_cache": V_cache,
        "attention_mask": attention_mask,
        "sink": None,
        "softmax_scale": softmax_scale_adjusted,
        "enable_fa_s_prior_tiling": cfg.enable_fa_s_prior_tiling,
        # -- FP8 KV cache quantization
        "k_scale": k_scale,
        "v_scale": v_scale,
        # -- KV cache update
        "update_cache": cfg.update_cache,
        "kv_cache_update_idx": kv_cache_update_idx,
        # -- output projection
        "W_out": W_out,
        "bias_out": bias_out,
        "quantization_type_out": cfg.quantization_type,
        "weight_dequant_scale_out": weight_dequant_scale_out,
        "input_dequant_scale_out": input_dequant_scale_out,
        # -- output
        "transposed_out": cfg.transposed_out,
        "transposed_in": cfg.transposed_in,
        "out_in_sb": cfg.output_in_sb,
        # -- KV data parallelism
        "KVDP": cfg.KVDP,
        "KVDP_replica_group": None,
        "KVDP_collective_mode": cfg.KVDP_collective_mode,
        # -- in-kernel mask generation
        "pos_ids": pos_ids,
        "swa_start_pos_ids": swa_start_pos_ids,
        "S_ctx": S_ctx if (cfg.use_pos_id and cfg.block_len == 0) else None,
        # -- MXFP quantization
        "is_h_transposed_by_4": cfg.quantization_type.is_mx(),
    }


# wrapper to test SBUF IO
def attention_block_tkg_kernel_test_wrapper(
    # -- input
    X: nl.ndarray,
    X_in_sb: bool,
    X_hidden_dim_actual: Optional[int],
    # -- rmsnorm X
    rmsnorm_X_enabled: bool,
    rmsnorm_X_eps: Optional[float],
    rmsnorm_X_gamma: Optional[nl.ndarray],
    # -- qkv projections
    W_qkv: nl.ndarray,
    bias_qkv: Optional[nl.ndarray],
    quantization_type_qkv: QuantizationType,
    weight_dequant_scale_qkv: Optional[nl.ndarray],
    input_dequant_scale_qkv: Optional[nl.ndarray],
    # -- QK rmsnorm pre RoPE
    rmsnorm_QK_enabled: bool,
    rmsnorm_QK_eps: Optional[float],
    W_rmsnorm_Q_pre_rope: Optional[nl.ndarray],
    W_rmsnorm_K_pre_rope: Optional[nl.ndarray],
    # -- RoPE embeddings
    cos: Optional[nl.ndarray],
    sin: Optional[nl.ndarray],
    rope_contiguous_layout: bool,
    # -- QK rmsnorm post RoPE
    rmsnorm_QK_post_rope_enabled: bool,
    rmsnorm_QK_post_rope_eps: float,
    W_rmsnorm_Q_post_rope: Optional[nl.ndarray],
    W_rmsnorm_K_post_rope: Optional[nl.ndarray],
    # -- attention
    skip_attention: bool,
    K_cache_transposed: bool,
    active_blocks_table: Optional[nl.ndarray],
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    attention_mask: nl.ndarray,
    sink: Optional[nl.ndarray],
    softmax_scale: Optional[float],
    enable_fa_s_prior_tiling: bool,
    # -- FP8 KV cache quantization
    k_scale: Optional[nl.ndarray],
    v_scale: Optional[nl.ndarray],
    # -- KV cache update
    update_cache: bool,
    kv_cache_update_idx: nl.ndarray,
    # -- output projection
    W_out: Optional[nl.ndarray],
    bias_out: Optional[nl.ndarray],
    quantization_type_out: QuantizationType,
    weight_dequant_scale_out: Optional[nl.ndarray],
    input_dequant_scale_out: Optional[nl.ndarray],
    # -- output
    transposed_out: bool,
    transposed_in: bool,
    out_in_sb: bool,
    sbm: Optional[SbufManager] = None,
    # -- KV data parallelism
    KVDP: int = 1,
    KVDP_replica_group=None,
    KVDP_collective_mode=None,
    # -- in-kernel mask generation
    pos_ids: Optional[nl.ndarray] = None,
    swa_start_pos_ids: Optional[nl.ndarray] = None,
    S_ctx: Optional[int] = None,
    is_h_transposed_by_4: bool = False,
):
    if transposed_in:
        # X is already in transposed layout [H0, n_prgs, H1_shard, BxS] from generate_kernel_inputs
        H0, n_prg, H1, _ = X.shape
        H = H0 * n_prg * H1
        _, B, _, S_tkg = attention_mask.shape
    else:
        B, S_tkg, H = X.shape
    if X_in_sb:
        # QKV_tkg requires the input shape to be (pmax, B*S, H // pmax)
        assert H % 128 == 0, "H must be divisible by 128"
        H0 = nl.tile_size.pmax
        H1 = H // 128
        BxS = B * S_tkg

        # Check program dimensionality
        _, lnc, _ = get_program_sharding_info()
        assert H1 % lnc == 0

        X_sb = nl.ndarray((H0, BxS, H1), X.dtype, nl.sbuf, name="X_sb")
        X_hbm = X.reshape((BxS, lnc, H0, H1 // lnc))

        """
        Note how X@HBM is read to SBUF: The full H dimension is divided into (lnc, H0=128, H1//lnc).
        Per SBUF partition (the H0=128 dim), we read H1//lnc values from each of the lnc chunks,
        interleaving them to reconstruct the full H1 dimension in SBUF while transposing the layout
        from (BxS, lnc, H0, H1//lnc) to (H0, BxS, H1). This matches how qkv_tkg() kernel expects
        SBUF input and constrains attention_block_tkg() SBUF input layout.
        """
        nisa.dma_copy(
            dst=TensorView(X_sb).reshape_dim(2, (lnc, -1)).get_view(),
            src=TensorView(X_hbm)
            .rearrange(('BS', 'lnc', 'H0', 'H1 // lnc'), ('H0', 'BS', 'lnc', 'H1 // lnc'))
            .get_view(),
        )

        X = X_sb

    kernel_output, K_hbm_out, V_hbm_out = attention_block_tkg(
        X=X,
        X_hidden_dim_actual=X_hidden_dim_actual,
        rmsnorm_X_enabled=rmsnorm_X_enabled,
        rmsnorm_X_eps=rmsnorm_X_eps,
        rmsnorm_X_gamma=rmsnorm_X_gamma,
        W_qkv=W_qkv,
        bias_qkv=bias_qkv,
        quantization_type_qkv=quantization_type_qkv,
        weight_dequant_scale_qkv=weight_dequant_scale_qkv,
        input_dequant_scale_qkv=input_dequant_scale_qkv,
        rmsnorm_QK_pre_rope_enabled=rmsnorm_QK_enabled,
        rmsnorm_QK_pre_rope_eps=rmsnorm_QK_eps if rmsnorm_QK_eps else 1e-5,
        rmsnorm_QK_pre_rope_W_Q=W_rmsnorm_Q_pre_rope,
        rmsnorm_QK_pre_rope_W_K=W_rmsnorm_K_pre_rope,
        cos=cos,
        sin=sin,
        rope_contiguous_layout=rope_contiguous_layout,
        rmsnorm_QK_post_rope_enabled=rmsnorm_QK_post_rope_enabled,
        rmsnorm_QK_post_rope_eps=rmsnorm_QK_post_rope_eps,
        rmsnorm_QK_post_rope_W_Q=W_rmsnorm_Q_post_rope,
        rmsnorm_QK_post_rope_W_K=W_rmsnorm_K_post_rope,
        skip_attention=skip_attention,
        K_cache_transposed=K_cache_transposed,
        active_blocks_table=active_blocks_table,
        K_cache=K_cache,
        V_cache=V_cache,
        attention_mask=attention_mask,
        sink=sink,
        softmax_scale=softmax_scale,
        enable_fa_s_prior_tiling=enable_fa_s_prior_tiling,
        update_cache=update_cache,
        kv_cache_update_idx=kv_cache_update_idx,
        k_scale=k_scale,
        v_scale=v_scale,
        W_out=W_out,
        bias_out=bias_out,
        quantization_type_out=quantization_type_out,
        weight_dequant_scale_out=weight_dequant_scale_out,
        input_dequant_scale_out=input_dequant_scale_out,
        transposed_out=transposed_out,
        out_in_sb=out_in_sb,
        transposed_in=transposed_in,
        sbm=sbm,
        KVDP=KVDP,
        KVDP_replica_group=KVDP_replica_group,
        KVDP_collective_mode=KVDP_collective_mode,
        pos_ids=pos_ids,
        swa_start_pos_ids=swa_start_pos_ids,
        S_ctx=S_ctx,
        is_h_transposed_by_4=is_h_transposed_by_4,
    )

    assert is_hbm_buffer(K_hbm_out)
    assert is_hbm_buffer(V_hbm_out)

    if not out_in_sb:
        return kernel_output, K_hbm_out, V_hbm_out

    assert kernel_output.buffer == nl.sbuf, "Expecting output on SBUF"

    # copy output to HBM
    skip_output_projection = W_out is None
    if skip_output_projection:
        kernel_output_hbm = nl.ndarray(kernel_output.shape, kernel_output.dtype, nl.hbm, name="kernel_output_hbm")
        nisa.dma_copy(kernel_output_hbm, kernel_output)
    else:
        kernel_output_hbm = relayout_sbuf_to_hbm_for_output_projection(kernel_output, transposed_out, B, S_tkg, H)

    return kernel_output_hbm, K_hbm_out, V_hbm_out


def relayout_sbuf_to_hbm_for_output_projection(kernel_output, transposed_out, B, S_tkg, H):
    # if transposed: SBUF.layout=(PMAX, H // lnc // PMAX, B*S_tkg) and HBM.layout=(PMAX, lnc, H // lnc // PMAX, B*S_tkg)
    # else: SBUF.layout=(B*S_tkg, H // lnc) and HBM.layout=(B*S_tkg, H)

    # Note: this code is based on the output_projection_tkg() logic
    _, n_prgs, prg_id = get_program_sharding_info()
    if transposed_out:
        H0, H1, H2 = n_prgs, nl.tile_size.pmax, H // n_prgs // nl.tile_size.pmax
        kernel_output_hbm = nl.ndarray(
            (H1, H0, H2, B * S_tkg), kernel_output.dtype, nl.shared_hbm, name="kernel_output_hbm"
        )
        nisa.dma_copy(
            dst=kernel_output_hbm.ap(
                pattern=[
                    [H0 * H2 * B * S_tkg, H1],
                    [B * S_tkg, H2],
                    [1, B * S_tkg],
                ],
                offset=prg_id * H2 * B * S_tkg,
            ),
            src=kernel_output,
        )
        return kernel_output_hbm

    # Else, not transposed out
    kernel_output_hbm = nl.ndarray((B * S_tkg, H), kernel_output.dtype, nl.shared_hbm, name="kernel_output_hbm")
    H_sharded = H // n_prgs
    nisa.dma_copy(kernel_output_hbm[:, nl.ds(prg_id * H_sharded, H_sharded)], kernel_output)
    return kernel_output_hbm


# FP8 KV cache validation: cosine similarity catches directional drift from mixed-precision
# (fp8/bf16 kernel vs fp32 golden), while allclose with min_pass_rate catches per-element errors.
# Both are needed because cosine similarity alone misses uniform scaling errors, and allclose
# alone is too strict for the accumulated rounding from FP8 quantization boundaries.
def make_cosine_similarity_validator(
    golden: npt.NDArray[Any], rtol: float, atol: float, min_cosine_similarity: float, min_pass_rate: float, name: str
) -> type[CustomValidator]:
    """Create a validator that checks cosine similarity and allclose with a minimum pass rate."""
    _golden = golden
    _rtol = rtol
    _atol = atol
    _min_cos = min_cosine_similarity
    _min_pass_rate = min_pass_rate
    _name = name
    _shape = golden.shape
    _dtype = golden.dtype

    class CosineValidator(CustomValidator):
        @override
        def validate(self, inference_output: npt.NDArray[Any]) -> bool:
            actual = inference_output.view(_dtype).reshape(_shape).astype(np.float32)
            expected = _golden.astype(np.float32)

            # Cosine similarity on flattened vectors
            a, b = actual.flatten(), expected.flatten()
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

            # Allclose with min_pass_rate
            allclose_pass = maxAllClose(
                actual, expected, rtol=_rtol, atol=_atol, verbose=1, logfile=self.logfile, min_pass_rate=_min_pass_rate
            )

            self._print_with_log(
                f"Validating {_name}: cosine_similarity={cos_sim:.6f} (min={_min_cos}), "
                f"allclose(pass_rate>={_min_pass_rate})={allclose_pass}"
            )
            return cos_sim >= _min_cos and allclose_pass

    return CosineValidator


def _golden_ref_via_torch(kernel_input: dict, lnc: int) -> dict:
    """Compute golden reference using the torch ref, returning numpy arrays in kernel dtypes.

    torch_ref_wrapper upcasts bf16/fp8→f32 for CPU compatibility. We cast back
    to the actual kernel IO dtypes (bf16/fp8)
    """
    kv_dtype = kernel_input['K_cache'].dtype
    torch_ref = AttentionBlockTkgTorchRef(lnc, kv_quant_dtype=str(kv_dtype))
    ignored = set(kernel_input) - set(signature(torch_ref).parameters)
    assert not ignored, f"kernel_input keys not consumed by torch ref: {ignored}"
    ref_output = torch_ref_wrapper(torch_ref)(**kernel_input)
    x_dtype = kernel_input['X'].dtype
    output_dtypes = {
        "X_out": x_dtype,
        "K_tkg": kv_dtype,
        "V_tkg": kv_dtype,
        "K_cache_updated": kv_dtype,
        "V_cache_updated": kv_dtype,
    }
    return {k: v.astype(output_dtypes[k]) for k, v in ref_output.items()}


def _infer_output_shapes_and_dtypes(kernel_input: dict, lnc: int) -> dict:
    """Infer output tensor shapes and dtypes by running the torch ref.

    The kernel has many output-shape variants (update_cache, K_cache_transposed,
    out_in_sb, transposed_out, block KV, …). Rather than duplicating that logic
    here — which is brittle and has caused shape mismatches — we run the torch
    ref once and mirror its output shapes.
    """
    golden = _golden_ref_via_torch(kernel_input, lnc)
    return {k: np.zeros(v.shape, dtype=v.dtype) for k, v in golden.items()}


def _get_tolerances(kv_quant: bool, quantization_type: QuantizationType, kv_quant_dtype_max: float = 240.0):
    """Return per-output (rtol, atol) dict based on quantization mode.

    With k_scale/v_scale fusion, X_out is O(1) for all kv_quant configs.
    K/V cache/tkg values remain in FP8 range (std ≈ FP8_MAX/4), so their
    tolerances are set differently:

    - atol = FP8_MAX/32: the FP8 step size at 1σ of the cache distribution.
      Covers rounding differences at small values where rtol contributes little.
    - rtol = 6%: FP8 has 3 mantissa bits (~12.5% worst-case quantization error).
      At high magnitudes (2-3σ), the step size can reach FP8_MAX/8 to FP8_MAX/16.
      The combined tolerance (atol + rtol × |value|) ensures coverage across the
      full range. 6% is used instead of 5% because kernel and golden may round
      to different adjacent FP8 values, and the relative error at 2-3σ values
      can slightly exceed 5%.

    X_out uses tighter rtol=5% because output values are O(1) after v_scale
    fusion, so FP8 quantization noise is not the dominant error source.
    """
    if kv_quant or quantization_type != QuantizationType.NONE:
        _kv_cache_atol = kv_quant_dtype_max / 32.0  # ULP at 1σ of cache distribution
        return {
            "X_out": (0.05, 1.0),
            "K_cache": (0.06, _kv_cache_atol),
            "V_cache": (0.06, _kv_cache_atol),
            "K_tkg": (0.06, _kv_cache_atol),
            "V_tkg": (0.06, _kv_cache_atol),
        }
    return {
        "X_out": (0.015, 1e-5),
        "K_cache": (0.015, 1e-5),
        "V_cache": (0.015, 1e-5),
        "K_tkg": (0.015, 1e-5),
        "V_tkg": (0.015, 1e-5),
    }


def _make_cosine_validation(
    golden_outputs: dict,
    tolerances: dict,
    kv_quant: bool,
    quantization_type: QuantizationType = QuantizationType.NONE,
    name_prefix: str = "",
) -> dict:
    """Build per-output cosine similarity + allclose validators.

    Pass rate:
      - Non-quantized (bf16): 100% of elements within rtol.
      - FP8 kv_quant X_out: 99% pass rate. FP8 has coarser quantization.
    """
    min_pass_rate_x_out = 0.99 if kv_quant else 1.0
    _cosine_threshold = 0.995 if (kv_quant or quantization_type != QuantizationType.NONE) else 0.99
    return {
        name: CustomValidatorWithOutputTensorData(
            validator=make_cosine_similarity_validator(
                golden,
                rtol=tolerances[name][0],
                atol=tolerances[name][1],
                min_cosine_similarity=_cosine_threshold,
                min_pass_rate=min_pass_rate_x_out if name == 'X_out' else 1.0,
                name=f"{name_prefix}{name}",
            ),
            output_ndarray=golden,
        )
        for name, golden in golden_outputs.items()
    }


def _run_attention_block_test(
    test_manager: Orchestrator,
    platform_target: Platforms,
    cfg: AttnBlkTestConfig,
):
    """Shared test execution logic for attention block TKG kernel.

    Single-rank case (KVDP=1):
        Uses UnitTestFramework with torch reference (AttentionBlockTkgTorchRef).

        INPUTS ──┬──> KERNEL ──> X_out, K/V_out ──┐
                 │                                ├─> compare
                 └──> GOLDEN ──> X_out, K/V_out ──┘

    Multi-rank case (KVDP>1):
        Uses manual orchestration (UnitTestFramework doesn't support PerRankLazy*).
        Generate inputs once with total_q_heads = KVDP * q_heads, then slice per-rank.
        This ensures:
          - Shared data is identical across ranks: X, W_k, W_v (GQA has 1 KV head shared by all Q heads)
          - Per-rank data is different: W_q, W_out (by head), K/V cache, mask (by batch)

        For each rank:
                                          ┌─slice K/V[B/KVDP]──> KERNEL ──> X_out, K/V[B/KVDP] ──────┐
                                          │                                    │                     │
        INPUTS ──slice W_q/W_out[q_heads]─┤                                    ├─> compare X_out     ├─> compare K/V
        K/V[B],                           │                                    │                     │
        W_q/W_out[q_heads*KVDP]           └────────────────────> GOLDEN ──> X_out, K/V[B] ──slice──> K/V[B/KVDP]

    Shape changes with KVDP>1 (per rank_id):

        Tensor                  Kernel                              Golden                       Description
        ──────                  ──────                              ──────                       ───────────
        X                       [B, S_tkg, H]                       [B, S_tkg, H]                same, shared across ranks
        W_qkv                   [H, d*(q_heads+2)]                  [H, d*(q_heads+2)]           same, sliced Q per rank, W_k/W_v replicated (GQA)
        W_out                   [q_heads*d, H]                      [q_heads*d, H]               same, sliced Q per rank
        K_cache (flat)          [B/KVDP, 1, S_ctx, d_head]          [B, 1, S_ctx, d_head]        kernel sliced B, golden full
        K_cache (block)         [num_blocks/KVDP, block_len, d]     [num_blocks, block_len, d]   kernel sliced B, golden full
        attention_mask          [S_ctx, B/KVDP, q_heads*KVDP, S]    [S_ctx, B, q_heads, S]       kernel: sliced B, gathered heads

        Note: For block KV indices (active_blocks_table and kv_cache_update_idx)
        we remap global block indices to per-rank local indices in _slice_block_kv_for_all_ranks()
    """
    estimated_bytes = estimate_test_memory_bytes(cfg)
    if estimated_bytes > _MAX_MEMORY_BYTES:
        pytest.skip(
            f"Estimated memory {estimated_bytes / 1024**3:.1f} GiB exceeds "
            f"limit {_MAX_MEMORY_BYTES / 1024**3:.1f} GiB "
            f"(set TEST_ATTN_BLK_TKG_MAX_MEMORY_GB to override)"
        )

    tolerances = _get_tolerances(
        cfg.kv_quant, cfg.quantization_type, get_max_positive_value_for_dtype(cfg.kv_quant_dtype)
    )

    if cfg.KVDP > 1:
        _run_kvdp_test(test_manager, platform_target, cfg, tolerances)
    else:
        _run_single_rank_test(test_manager, platform_target, cfg, tolerances)


def _run_single_rank_test(
    test_manager: Orchestrator,
    platform_target: Platforms,
    cfg: AttnBlkTestConfig,
    tolerances: dict,
):
    """Run single-rank test using UnitTestFramework with cosine similarity validation."""

    kernel_input = generate_kernel_inputs(cfg)
    golden_outputs = _golden_ref_via_torch(kernel_input, cfg.lnc)

    # When update_cache=True, the kernel returns K_cache/V_cache in-place, causing:
    # 1. The NKI compiler renames these inputs to K_cache.must_alias_input / V_cache.must_alias_input
    #    in the NEFF. Rename input keys so the neuron-profile command uses the NEFF names.
    # 2. The NEFF output files are named K_cache/V_cache (matching the aliased inputs), not
    #    K_cache_updated/V_cache_updated (the torch ref names). Rename golden keys to match.
    # The test framework handles the .must_alias_input suffix throughout
    # (kernel_tracer strips it for compilation, unit_test_framework for validation).
    if cfg.update_cache:
        kernel_input['K_cache.must_alias_input'] = kernel_input.pop('K_cache')
        kernel_input['V_cache.must_alias_input'] = kernel_input.pop('V_cache')
        golden_outputs['K_cache'] = golden_outputs.pop('K_cache_updated')
        golden_outputs['V_cache'] = golden_outputs.pop('V_cache_updated')

    def input_generator(test_config):
        return kernel_input

    custom_validation = ValidationArgs(
        golden_output=_make_cosine_validation(golden_outputs, tolerances, cfg.kv_quant, cfg.quantization_type)
    )

    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=nki.jit(attention_block_tkg_kernel_test_wrapper),
        torch_ref=torch_ref_wrapper(AttentionBlockTkgTorchRef(cfg.lnc, kv_quant_dtype=cfg.kv_quant_dtype)),
        kernel_input_generator=input_generator,
        output_tensor_descriptor=lambda ki: _infer_output_shapes_and_dtypes(
            {k.removesuffix(".must_alias_input"): v for k, v in ki.items()}, cfg.lnc
        ),
    )
    framework.run_test(
        test_config=None,
        compiler_args=CompilerArgs(logical_nc_config=cfg.lnc, enable_birsim=False, platform_target=platform_target),
        inference_args=replace(
            TKG_INFERENCE_ARGS,
            collective_ranks=1,
            enable_determinism_check=False,
        ),
        custom_validation_args=custom_validation,
    )


def _run_kvdp_test(
    test_manager: Orchestrator,
    platform_target: Platforms,
    cfg: AttnBlkTestConfig,
    tolerances: dict,
):
    """Run multi-rank KVDP test using CollectiveUnitTestFramework.

    Uses per_rank_torch_ref_input_override to transform kernel inputs for golden
    computation (slice W_qkv/W_out per rank, set KVDP=1 for torch ref), and
    custom_comparator to post-process the torch_ref golden (slice K/V cache,
    rename keys, wrap in cosine similarity validators).
    """

    # Block KV with S_tkg > 1 and KVDP: _slice_block_kv_for_all_ranks remaps kv_cache_update_idx
    # per starting block only. When S_tkg > 1 crosses a block boundary, the kernel writes to
    # consecutive local blocks, but the remapping doesn't guarantee global block N and N+1 map
    # to adjacent local blocks. This is an issue with the kernel interface but KVDP exposes
    # the issue (in other cases golden and kernel behave in the same way).
    kvdp_cfg = replace(cfg, q_heads=cfg.KVDP * cfg.q_heads)
    base_input = generate_kernel_inputs(kvdp_cfg)

    # For KV data parallelism with KVDP ranks, each rank has q_heads, so total = KVDP * q_heads.
    # Generate inputs once with total heads, then slice per-rank.
    B_attn = cfg.batch // cfg.KVDP
    per_rank_input_gen, per_rank_cache = _create_per_rank_inputs(
        base_input, cfg.KVDP, cfg.q_heads, cfg.d_head, B_attn, cfg.S_tkg, cfg.S_ctx, cfg.block_len
    )

    # Build per-rank input generator with .must_alias_input renaming
    def create_per_rank_input(rank_id):
        result = per_rank_input_gen.for_rank(rank_id).copy()
        if cfg.update_cache:
            result['K_cache.must_alias_input'] = result.pop('K_cache')
            result['V_cache.must_alias_input'] = result.pop('V_cache')
        return result

    # Override torch_ref inputs per rank: slice W_qkv/W_out to this rank's q_heads,
    # use full-batch mask, set KVDP=1, and mask kv_cache_update_idx for block KV.
    def ref_input_override(rank_id, kernel_input_for_rank):
        golden_input_copy = per_rank_input_gen.base_input.copy()
        golden_input_copy['W_qkv'] = per_rank_cache['w_qkv'][rank_id]
        golden_input_copy['bias_qkv'] = per_rank_cache['bias_qkv'][rank_id]
        golden_input_copy['W_out'] = per_rank_cache['w_out'][rank_id]
        golden_input_copy['attention_mask'] = per_rank_cache['golden_mask']
        golden_input_copy['KVDP'] = 1
        if cfg.block_len > 0:
            masked_idx = golden_input_copy['kv_cache_update_idx'].copy()
            masked_idx[: rank_id * B_attn] = np.iinfo(np.uint32).max
            masked_idx[(rank_id + 1) * B_attn :] = np.iinfo(np.uint32).max
            golden_input_copy['kv_cache_update_idx'] = masked_idx
        return golden_input_copy

    # Post-process torch_ref golden: cast dtypes, slice K/V per rank, rename keys,
    # wrap in cosine similarity validators.
    x_dtype = base_input['X'].dtype
    kv_dtype = base_input['K_cache'].dtype
    output_dtypes = {
        "X_out": x_dtype,
        "K_tkg": kv_dtype,
        "V_tkg": kv_dtype,
        "K_cache_updated": kv_dtype,
        "V_cache_updated": kv_dtype,
    }

    def comparator(rank_id, golden_dict):
        # Cast from float32 back to kernel IO dtypes (same as _golden_ref_via_torch)
        rank_golden = {k: v.astype(output_dtypes[k]) for k, v in golden_dict.items()}
        rank_golden = _slice_golden_KV_cache_for_rank(
            rank_golden,
            rank_id,
            B_attn,
            cfg.block_len,
            cfg.d_head,
            cfg.S_ctx,
            per_rank_cache,
            cfg.update_cache,
        )
        if cfg.update_cache:
            rank_golden['K_cache'] = rank_golden.pop('K_cache_updated')
            rank_golden['V_cache'] = rank_golden.pop('V_cache_updated')
        return _make_cosine_validation(
            rank_golden, tolerances, cfg.kv_quant, cfg.quantization_type, name_prefix=f"rank{rank_id}:"
        )

    framework = CollectiveUnitTestFramework(
        test_manager=test_manager,
        kernel_entry=nki.jit(attention_block_tkg_kernel_test_wrapper),
        torch_ref=torch_ref_wrapper(AttentionBlockTkgTorchRef(cfg.lnc, kv_quant_dtype=cfg.kv_quant_dtype)),
        per_rank_input_generator=create_per_rank_input,
        collective_ranks=cfg.KVDP,
        per_rank_torch_ref_input_override=ref_input_override,
    )
    framework.run_test(
        test_config=None,
        compiler_args=CompilerArgs(logical_nc_config=cfg.lnc, enable_birsim=False, platform_target=platform_target),
        inference_args=replace(TKG_INFERENCE_ARGS, collective_ranks=cfg.KVDP, enable_determinism_check=False),
        custom_comparator=comparator,
    )


@lru_cache(maxsize=1)
def _get_attention_block_metadata():
    return load_model_configs("test_attention_block")


# fmt: off
RANGE_ATTN_BLK_CFGS = [
    # SBUF IO
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=6144, H_actual=2880, S_ctx=11264, S_max_ctx=11264, S_tkg=1,
                        output_in_sb=True),
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=6144, H_actual=2880, S_ctx=11264, S_max_ctx=11264, S_tkg=1,
                        transposed_out=True, output_in_sb=True),
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=6144, H_actual=2880, S_ctx=11264, S_max_ctx=11264, S_tkg=2,
                        update_cache=False, rmsnorm_X=False, skip_rope=True, input_in_sb=True, output_in_sb=True),
    # HBM IO
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=6144, H_actual=2880, S_ctx=11264, S_max_ctx=11264, S_tkg=1),
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=128, H=6144, H_actual=2880, S_ctx=11264, S_max_ctx=11264, S_tkg=1,
                        qk_norm_pre_rope=True, qk_norm_post_rope=True, qk_norm_post_rope_gamma=True, test_bias=True),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=6144, H_actual=2880, S_ctx=10240, S_max_ctx=10240, S_tkg=5,
                        update_cache=False),
    # GPT OSS RIV'25
    AttnBlkTestConfig(batch=8, q_heads=8, d_head=64, H=6144, H_actual=2880, S_ctx=11264, S_max_ctx=11264, S_tkg=1,
                        test_bias=True),
    AttnBlkTestConfig(batch=8, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=11264, S_max_ctx=11264, S_tkg=5,
                        test_bias=True),
    AttnBlkTestConfig(batch=8, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=10240, S_max_ctx=10240, S_tkg=5,
                        test_bias=True),
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=10240, S_max_ctx=10240, S_tkg=4,
                        K_cache_transposed=True, test_bias=True),
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=10240, S_max_ctx=10240, S_tkg=4,
                        K_cache_transposed=True, test_bias=True),
    # Qwen3
    AttnBlkTestConfig(batch=16, q_heads=1, d_head=128, H=4096, H_actual=None, S_ctx=10240, S_max_ctx=10240, S_tkg=1,
                        qk_norm_pre_rope=True),
    # Qwen3 with pre-rope gamma weights
    AttnBlkTestConfig(batch=16, q_heads=1, d_head=128, H=4096, H_actual=None, S_ctx=10240, S_max_ctx=10240, S_tkg=1,
                        update_cache=False, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True),
    # Gemma3 with pre-rope gamma weights
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=5376, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        update_cache=False, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True),
    AttnBlkTestConfig(batch=8, q_heads=4, d_head=128, H=5376, H_actual=None, S_ctx=10240, S_max_ctx=10240, S_tkg=3,
                        update_cache=False, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True, test_bias=True),
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=5376, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=16, update_cache=False, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True),
    # New model, 2025-Jul
    AttnBlkTestConfig(batch=32, q_heads=1, d_head=64, H=3072, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        rmsnorm_X=False, test_bias=True),
    AttnBlkTestConfig(batch=32, q_heads=1, d_head=64, H=3072, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, test_bias=True),
    AttnBlkTestConfig(batch=64, q_heads=1, d_head=64, H=3072, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, test_bias=True),
    AttnBlkTestConfig(batch=32, q_heads=1, d_head=64, H=3072, H_actual=None, S_ctx=128, S_max_ctx=128, S_tkg=1,
                        rmsnorm_X=False, test_bias=True),
    AttnBlkTestConfig(batch=32, q_heads=1, d_head=64, H=3072, H_actual=None, S_ctx=128, S_max_ctx=128, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, test_bias=True),
    AttnBlkTestConfig(batch=64, q_heads=1, d_head=64, H=3072, H_actual=None, S_ctx=128, S_max_ctx=128, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, test_bias=True),
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=64, H=3072, H_actual=None, S_ctx=128, S_max_ctx=128, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, test_bias=True),
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, test_bias=True),
    AttnBlkTestConfig(batch=8, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, test_bias=True),
    AttnBlkTestConfig(batch=16, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, test_bias=True),
    AttnBlkTestConfig(batch=32, q_heads=1, d_head=64, H=3072, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=3,
                        K_cache_transposed=True, rmsnorm_X=False, transposed_out=True, test_bias=True),
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=2,
                        K_cache_transposed=True, rmsnorm_X=False, transposed_out=True, test_bias=True),
    # secret text
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=7168, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=1,
                        K_cache_transposed=True, qk_norm_post_rope=True, qk_norm_post_rope_gamma=True),
    # llama
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, qk_norm_post_rope=True),
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, skip_rope=True),
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False),
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, rope_contiguous_layout=False),
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=5120, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rope_contiguous_layout=False),
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=5120, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, skip_rope=True, rope_contiguous_layout=False),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=10240, S_max_ctx=16384, S_tkg=5,
                        K_cache_transposed=True),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=10240, S_max_ctx=10240, S_tkg=5,
                        K_cache_transposed=True),
    AttnBlkTestConfig(batch=8, q_heads=2, d_head=128, H=16384, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=7,
                        K_cache_transposed=True),
    AttnBlkTestConfig(batch=1, q_heads=16, d_head=128, H=16384, H_actual=None, S_ctx=4096, S_max_ctx=8192, S_tkg=7,
                        rmsnorm_X=False),
    AttnBlkTestConfig(batch=1, q_heads=16, d_head=128, H=16384, H_actual=None, S_ctx=4096, S_max_ctx=8192, S_tkg=7,
                        rmsnorm_X=False, rope_contiguous_layout=False),
    AttnBlkTestConfig(batch=8, q_heads=2, d_head=128, H=16384, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=7,
                        K_cache_transposed=True, transposed_out=True),
    # Test vectors for block KV
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=5,
                        block_len=16),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=5,
                        block_len=16),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=12288, S_max_ctx=12288, S_tkg=5,
                        block_len=16),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=10240, S_max_ctx=10240, S_tkg=5,
                        block_len=16),
    # Block boundary crossing: S_tkg=17 > block_len=16 guarantees tokens span
    # at least two blocks regardless of starting position within a block.
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=128, S_max_ctx=128, S_tkg=17,
                        block_len=16),
    # BxS > pmax (128): exercises multi-iteration tiling loop in _update_block_cache_scalar
    AttnBlkTestConfig(batch=32, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=5,
                        block_len=16),
    AttnBlkTestConfig(batch=64, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=128, S_max_ctx=128, S_tkg=3,
                        block_len=16),
    # Test vectors to verify functionality of different q_heads, d_head and H dimensions
    AttnBlkTestConfig(batch=2, q_heads=1, d_head=128, H=2048, H_actual=None, S_ctx=10240, S_max_ctx=16384, S_tkg=5,
                        K_cache_transposed=True),
    AttnBlkTestConfig(batch=2, q_heads=1, d_head=64, H=2048, H_actual=None, S_ctx=10240, S_max_ctx=16384, S_tkg=5,
                        K_cache_transposed=True),
    AttnBlkTestConfig(batch=2, q_heads=2, d_head=64, H=3072, H_actual=None, S_ctx=10240, S_max_ctx=16384, S_tkg=5),
    AttnBlkTestConfig(batch=2, q_heads=3, d_head=64, H=4096, H_actual=None, S_ctx=10240, S_max_ctx=16384, S_tkg=5,
                        update_cache=False),
    AttnBlkTestConfig(batch=2, q_heads=4, d_head=128, H=6144, H_actual=None, S_ctx=10240, S_max_ctx=16384, S_tkg=5,
                        update_cache=False, K_cache_transposed=True),
    AttnBlkTestConfig(batch=2, q_heads=3, d_head=128, H=20480, H_actual=None, S_ctx=10240, S_max_ctx=16384, S_tkg=5,
                        K_cache_transposed=True),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=10240, S_max_ctx=10240, S_tkg=5,
                        K_cache_transposed=True),
    # static quantization tests
    # TODO: random input causing numerical instability for quantized weights, more tests will be added
    # after better fp8 random generator is implemented
    # E2E inference tests shows good accuracy
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=5,
                        K_cache_transposed=True, quantization_type=QuantizationType.STATIC),
    # row-wise quantization tests
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=5,
                        K_cache_transposed=True, quantization_type=QuantizationType.ROW),
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=5,
	                    K_cache_transposed=True, quantization_type=QuantizationType.ROW, kv_quant=True),
    # MX quantization tests
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        K_cache_transposed=True, quantization_type=QuantizationType.MX, supported_platforms={Platforms.TRN3, Platforms.TRN3_A0}),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        K_cache_transposed=True, quantization_type=QuantizationType.STATIC_MX, supported_platforms={Platforms.TRN3, Platforms.TRN3_A0}),
    # ROW_MX not yet supported, requires support in output_projection_tkg.
    # AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
    #                     K_cache_transposed=True, quantization_type=QuantizationType.ROW_MX, supported_platforms={Platforms.TRN3, Platforms.TRN3_A0}),
    # softmax_scale tests (Gemma model support)
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=10240, S_max_ctx=10240, S_tkg=4,
                        K_cache_transposed=True, test_bias=True, softmax_scale=0.05),
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=7168, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=1,
                        K_cache_transposed=True, qk_norm_post_rope=True, qk_norm_post_rope_gamma=True, softmax_scale=0.09),
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=5120, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, skip_rope=True, rope_contiguous_layout=False, softmax_scale=0.13),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=5,
                        block_len=16, softmax_scale=0.17),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=10240, S_max_ctx=10240, S_tkg=5,
                        K_cache_transposed=True, softmax_scale=0.21),
    # llama FP8 KV Cache Tests
    AttnBlkTestConfig(batch=2, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        rmsnorm_X=False, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    AttnBlkTestConfig(batch=37, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    AttnBlkTestConfig(batch=96, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # llama FP8 KV Cache Tests - batched cache update
    AttnBlkTestConfig(batch=32, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=4096, S_max_ctx=4096, S_tkg=1,
                        rmsnorm_X=False, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    AttnBlkTestConfig(batch=128, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        rmsnorm_X=False, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # llama FP8 KV Cache Tests - block KV cache
    AttnBlkTestConfig(batch=16, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=16, rmsnorm_X=False, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    AttnBlkTestConfig(batch=32, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=16, rmsnorm_X=False, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    AttnBlkTestConfig(batch=64, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=16, rmsnorm_X=False, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # FP8 KV cache direct cast (kv_scale=1.0)
    AttnBlkTestConfig(batch=1, q_heads=2, d_head=128, H=8192, H_actual=None, S_ctx=26624, S_max_ctx=36896, S_tkg=5,
                        block_len=32, kv_quant=True, kv_scale=1.0, enable_fa_s_prior_tiling=False),
    # FP8 KV cache - finite (float8_e4m3fn)
    AttnBlkTestConfig(batch=16, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=16, rmsnorm_X=False, kv_quant=True, kv_quant_dtype=nl.float8_e4m3fn,
                        kv_scale=104.0, supported_platforms={Platforms.TRN3}),
    AttnBlkTestConfig(batch=1, q_heads=2, d_head=128, H=8192, H_actual=None, S_ctx=26624, S_max_ctx=36896, S_tkg=5,
                        block_len=32, kv_quant=True, kv_quant_dtype=nl.float8_e4m3fn, kv_scale=1.0,
                        enable_fa_s_prior_tiling=False, supported_platforms={Platforms.TRN3}),
    # Long context tests (S_ctx >= 128k, slower)
    # flat KV S_ctx=128k
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                        rmsnorm_X=False, test_bias=True),
    # flat KV S_ctx=512k
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=524288, S_max_ctx=524288, S_tkg=1,
                        rmsnorm_X=False, test_bias=True),
    # block KV S_ctx=128k, block_len=32
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True),
    # block KV S_ctx=128k, block_len=32, cache_lens tightly concentrated around 20%
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True, cache_lens_mean=0.2, cache_lens_stddev=0.01),
    # block KV S_ctx=512k, block_len=32
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=524288, S_max_ctx=524288, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True),
    # KVDP tests (KVDP=4, GPT-OSS-like)
    # q_heads=1 means each rank has 1 q_head, total KVDP * q_heads across ranks
    # flat KV S_ctx=1k B=8
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        test_bias=True, KVDP=4, kv_quant=True),
    # update_cache=False for complete API coverage
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        test_bias=True, KVDP=4, kv_quant=True, update_cache=False),
    # q_heads=2 tests the general transpose path (q_heads>1)
    AttnBlkTestConfig(batch=8, q_heads=2, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        test_bias=True, KVDP=4, kv_quant=True),
    # block KV S_ctx=1k B=8, block_len=32
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True, KVDP=4, kv_quant=True),
    # block KV S_ctx=1k B=32, block_len=32
    AttnBlkTestConfig(batch=32, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True, KVDP=4, kv_quant=True),
    # KVDP + block KV + S_tkg > 1: exercises per-token KVDP slicing
    # S_tkg=17 > block_len=16 guarantees block boundary crossing for any seed.
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=17,
                        block_len=16, rmsnorm_X=False, test_bias=True, KVDP=4, kv_quant=True),
    # KVDP long context tests (S_ctx >= 128k, slower)
    # flat KV S_ctx=512k
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=524288, S_max_ctx=524288, S_tkg=1,
                        rmsnorm_X=False, test_bias=True, KVDP=4, kv_quant=True),
    # block KV S_ctx=512k, block_len=32
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=524288, S_max_ctx=524288, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True, KVDP=4, kv_quant=True),
    # flat KV S_ctx=1M
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1048576, S_max_ctx=1048576, S_tkg=1,
                        rmsnorm_X=False, test_bias=True, KVDP=4, kv_quant=True),
    # block KV S_ctx=1M, block_len=32
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1048576, S_max_ctx=1048576, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True, KVDP=4, kv_quant=True),
    # KVDP B=64 tests
    # block KV S_ctx=1k B=64, block_len=32
    AttnBlkTestConfig(batch=64, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True, KVDP=4, kv_quant=True),
    # block KV S_ctx=128k B=64, block_len=32
    AttnBlkTestConfig(batch=64, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True, KVDP=4, kv_quant=True),
    # flat KV S_ctx=128k B=64
    AttnBlkTestConfig(batch=64, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                        rmsnorm_X=False, test_bias=True, KVDP=4, kv_quant=True),
    # Small S_ctx=128: sprior_n_prgs=1 (not sharded)
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=5120, H_actual=None, S_ctx=128, S_max_ctx=128, S_tkg=1,
                        block_len=32, rope_contiguous_layout=False, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True),
    # Large batch + small S_ctx: batch-sharded so sprior_n_prgs=1
    AttnBlkTestConfig(batch=256, q_heads=1, d_head=128, H=5120, H_actual=None, S_ctx=128, S_max_ctx=128, S_tkg=1,
                        block_len=32, rope_contiguous_layout=False, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True),
    # B=1 large block_len with FA tiling
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=5120, H_actual=None, S_ctx=32768, S_max_ctx=32768, S_tkg=1,
                        block_len=128, rope_contiguous_layout=False, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True),

    # ===== Large batch coverage (B*S_tkg > 128) =====
    ## Flat KV (block_len=0)
    AttnBlkTestConfig(batch=255, q_heads=2, d_head=128, H=8192, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=3,
                        qk_norm_pre_rope=True, qk_norm_post_rope=True, qk_norm_post_rope_gamma=True, test_bias=True),
    AttnBlkTestConfig(batch=255, q_heads=2, d_head=64, H=8192, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=3,
                        K_cache_transposed=True),
    AttnBlkTestConfig(batch=384, q_heads=3, d_head=128, H=5120, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        qk_norm_post_rope=True, qk_norm_post_rope_gamma=True, transposed_out=True),
    AttnBlkTestConfig(batch=384, q_heads=3, d_head=64, H=5120, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    ## Block KV
    # FP8 KV quantization
    AttnBlkTestConfig(batch=255, q_heads=3, d_head=128, H=5120, H_actual=None, S_ctx=26624, S_max_ctx=36896, S_tkg=2,
                        block_len=32, kv_quant=True, kv_scale=1.0),
    # FP8 weight quantization
    AttnBlkTestConfig(batch=255, q_heads=2, d_head=128, H=8192, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, quantization_type=QuantizationType.STATIC),
    # d_head=64, q_heads>1
    AttnBlkTestConfig(batch=255, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=32),
    # No RoPE, update_cache=False, transposed_out, test_bias, softmax_scale
    AttnBlkTestConfig(batch=255, q_heads=3, d_head=128, H=5120, H_actual=None, S_ctx=10240, S_max_ctx=12288, S_tkg=1,
                        block_len=32, skip_rope=True, update_cache=False, transposed_out=True, test_bias=True, softmax_scale=0.05),
    # B*S_tkg > 256 (multi-tile with S_tkg>1)
    AttnBlkTestConfig(batch=128, q_heads=2, d_head=128, H=5120, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=3,
                        block_len=32),
    # QK norm post-RoPE, rmsnorm_X=False (no input norm)
    AttnBlkTestConfig(batch=255, q_heads=1, d_head=128, H=5120, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, qk_norm_post_rope=True, qk_norm_post_rope_gamma=True, rmsnorm_X=False),
    # Large B*q_heads
    AttnBlkTestConfig(batch=129, q_heads=9, d_head=128, H=5120, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True, qk_norm_post_rope=True,
                        qk_norm_post_rope_gamma=True),
    # rope_contiguous_layout=False with large tile_B * q_heads (exercises gemm_moving_fmax in RoPE)
    AttnBlkTestConfig(batch=255, q_heads=8, d_head=64, H=8192, H_actual=None, S_ctx=32768, S_max_ctx=32768, S_tkg=5,
                        block_len=128, rope_contiguous_layout=False),
    # B=1024
    AttnBlkTestConfig(batch=1024, q_heads=2, d_head=128, H=5120, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True),
    AttnBlkTestConfig(batch=1024, q_heads=2, d_head=128, H=5120, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                       block_len=32, rmsnorm_X=False, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True,
                       quantization_type=QuantizationType.STATIC),
    # Large batch + KVDP
    AttnBlkTestConfig(batch=256, q_heads=2, d_head=64, H=3072, H_actual=2880, S_ctx=10240, S_max_ctx=10240, S_tkg=1,
                        block_len=32, test_bias=True, KVDP=4),
    AttnBlkTestConfig(batch=288, q_heads=2, d_head=128, H=5120, H_actual=4096, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, test_bias=True, KVDP=16),
    AttnBlkTestConfig(batch=512, q_heads=4, d_head=128, H=5120, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True, KVDP=4),
    AttnBlkTestConfig(batch=2048, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=10240, S_max_ctx=10240, S_tkg=1,
                        block_len=128, KVDP=8, kv_quant=True),
    # KV-DP ALL_GATHER_SLICE regression tests (KVDP=4, S_ctx=1k)
    # flat KV q_heads=1
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        rmsnorm_X=False, test_bias=True, KVDP=4, KVDP_collective_mode=KVDPCollectiveMode.ALL_GATHER_SLICE),
    # flat KV q_heads=2
    AttnBlkTestConfig(batch=8, q_heads=2, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        rmsnorm_X=False, test_bias=True, KVDP=4, KVDP_collective_mode=KVDPCollectiveMode.ALL_GATHER_SLICE),
    # block KV q_heads=1
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True, KVDP=4, KVDP_collective_mode=KVDPCollectiveMode.ALL_GATHER_SLICE),
    # block KV q_heads=2
    AttnBlkTestConfig(batch=8, q_heads=2, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True, KVDP=4, KVDP_collective_mode=KVDPCollectiveMode.ALL_GATHER_SLICE),
    # flat KV S_ctx=131k
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                        rmsnorm_X=False, test_bias=True, KVDP=4, KVDP_collective_mode=KVDPCollectiveMode.ALL_GATHER_SLICE),
    # block KV S_ctx=131k, block_len=32
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True, KVDP=4, KVDP_collective_mode=KVDPCollectiveMode.ALL_GATHER_SLICE),
    # large B q_heads=2: exercises tiled transpose (B*q_heads=512 > pmax)
    AttnBlkTestConfig(batch=256, q_heads=2, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, test_bias=True, KVDP=4, KVDP_collective_mode=KVDPCollectiveMode.ALL_GATHER_SLICE),

    # ===== Transposed in+out tests =====
    # Tests the [H0, n_prgs, H1_shard, BxS] HBM input layout with transposed output.
    # Covers: multiple models, d_head sizes, batch sizes, quant, qk_norm, softmax_scale, flat/block KV.
    # llama3_70b: B=1, block KV
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=32, transposed_in=True, transposed_out=True),
    # llama3_70b: B=16, block KV
    AttnBlkTestConfig(batch=16, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=32, transposed_in=True, transposed_out=True),
    # qwen3_32b: B=1, block KV, qk_norm_pre_rope, rmsnorm_X=False
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=5120, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=32, rmsnorm_X=False, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True,
                        transposed_in=True, transposed_out=True),
    # gptoss_120b: B=1, d_head=64, H_actual padding, block KV
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=32, transposed_in=True, transposed_out=True),
    # qwen3_235b: B=1, flat KV, qk_norm_pre_rope
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=4096, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True,
                        transposed_in=True, transposed_out=True),
    # gemma3_27b: B=16, softmax_scale, rmsnorm_X=False, qk_norm_pre_rope, block KV
    AttnBlkTestConfig(batch=16, q_heads=1, d_head=128, H=5376, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=32, rmsnorm_X=False, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True,
                        softmax_scale=0.07715167498, transposed_in=True, transposed_out=True),
    # llama3_70b: B=8, S_tkg=5, STATIC weight quant, K_cache_transposed
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=5,
                        K_cache_transposed=True, quantization_type=QuantizationType.STATIC,
                        transposed_in=True, transposed_out=True),
    # llama3_70b TP=16: B=16, q_heads=4, block KV (block_len=32, S_ctx=1024)
    # First layer: BSH input, transposed output
    AttnBlkTestConfig(batch=16, q_heads=4, d_head=128, H=8192, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, transposed_out=True),
    # Last layer: transposed input, BSH output
    AttnBlkTestConfig(batch=16, q_heads=4, d_head=128, H=8192, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, transposed_in=True),

    # ===== In-kernel mask generation (use_pos_id=True) =====
    # Block KV, basic causal mask
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=5376, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True, use_pos_id=True),
    # Flat KV, basic causal mask
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=5376, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=0, use_pos_id=True),
    # Block KV, SWA mask
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=5376, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True, use_pos_id=True, sliding_window=256),
    # KVDP
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=5376, H_actual=None, S_ctx=16384, S_max_ctx=16384, S_tkg=1,
                        block_len=32, KVDP=4, use_pos_id=True),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=5376, H_actual=None, S_ctx=16384, S_max_ctx=16384, S_tkg=3,
                        block_len=32, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True, KVDP=4, use_pos_id=True, sliding_window=256),
]
# fmt: on


def _low_rank_cfgs(cfgs: list[AttnBlkTestConfig]) -> list[AttnBlkTestConfig]:
    """Filter configs that need at most 4 NeuronCores (non-high-rank)."""
    return [c for c in cfgs if not c.is_high_rank()]


@pytest_test_metadata(name="Attention Block TKG", tags=["model"])
@pytest_marks(["attention", "tkg", "experimental", "mx"])
@final
class TestRangeAttnBlk:
    # fmt: off
    FAST_ATTN_BLK_CFGS = [
        # Basic flat KV (core path: RMSNorm + QKV + RoPE + attention + output projection)
        AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            K_cache_transposed=True),
        # Block KV cache
        AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=1,
                            block_len=32),
        # FP8 static weight quantization
        AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            block_len=32, quantization_type=QuantizationType.STATIC),
        # FP8 KV cache quantization
        AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            block_len=32, kv_quant=True),
        # QK norm pre-rope with gamma (Qwen3/Gemma3 path)
        AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=5120, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            block_len=32, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True),
        # Transposed in+out layout
        AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            block_len=32, transposed_in=True, transposed_out=True),
        # Multi-token generation (S_tkg > 1)
        AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=5,
                            block_len=32),
        # In-kernel mask generation (use_pos_id=True)
        AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=5376, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            block_len=32, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True, use_pos_id=True),
        # H_actual padding (GPT-OSS style)
        AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            test_bias=True),
    ]
    # fmt: on

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "attn_blk_cfg",
        FAST_ATTN_BLK_CFGS,
        ids=lambda p: p.test_id(),
    )
    def test_attn_blk_fast(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        attn_blk_cfg: AttnBlkTestConfig,
    ):
        _run_attention_block_test(
            test_manager=test_manager,
            platform_target=platform_target,
            cfg=attn_blk_cfg,
        )

    # fmt: off
    @pytest.mark.parametrize("attn_blk_cfg",
        _low_rank_cfgs(RANGE_ATTN_BLK_CFGS),
        ids=lambda p: p.test_id(),
    # fmt: on
    )
    def test_attn_blk_megakernel(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        attn_blk_cfg: AttnBlkTestConfig
    ):
        assert not attn_blk_cfg.is_high_rank(), \
            f"High-rank config (KVDP={attn_blk_cfg.KVDP}, DCP={attn_blk_cfg.DCP}) belongs in test_attention_block_tkg_high_rank.py"
        _run_attention_block_test(
            test_manager=test_manager,
            platform_target=platform_target,
            cfg=attn_blk_cfg,
        )


@pytest_marks(["attention", "tkg", "experimental", "mx", "model"])
@final
class TestAttnBlkModel:
    """Model regression tests for Attention Block TKG kernel.

    Separate test methods per tier for cleaner pytest discovery:
    - test_tier0: Critical model configs (high priority)
    - test_optimal: Optimal performance configs
    - test_generality: Generality/coverage configs

    High-rank model configs (KVDP/DCP > 4) are in test_attention_block_tkg_high_rank.py.
    """

    def _run_model_test(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        cfg: AttnBlkTestConfig,
    ):
        """Common test logic for all model tiers."""
        assert not cfg.is_high_rank(), (
            f"High-rank config (KVDP={cfg.KVDP}, DCP={cfg.DCP}) belongs in test_attention_block_tkg_high_rank.py"
        )
        attn_blk_metadata_list = _get_attention_block_metadata()
        test_metadata_key = {
            "batch": cfg.batch,
            "q_heads": cfg.q_heads,
            "d_head": cfg.d_head,
            "H": cfg.H,
            "S_ctx": cfg.S_ctx,
            "S_tkg": cfg.S_tkg,
            "kv_quant": cfg.kv_quant,
            "KVDP": cfg.KVDP,
            "transposed_in": cfg.transposed_in,
            "DCP": cfg.DCP,
        }
        collector.match_and_add_metadata_dimensions(test_metadata_key, attn_blk_metadata_list)
        _run_attention_block_test(
            test_manager=test_manager,
            platform_target=platform_target,
            cfg=cfg,
        )

    @pytest.mark.tier0
    @pytest.mark.parametrize(
        "cfg",
        _low_rank_cfgs(attention_block_tkg_model_configs.get(ModelTestType.TIER0, [])),
        ids=[cfg.test_id() for cfg in _low_rank_cfgs(attention_block_tkg_model_configs.get(ModelTestType.TIER0, []))],
    )
    def test_tier0(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        cfg: AttnBlkTestConfig,
    ):
        """TIER0: Critical model configs - highest priority for model validation."""
        self._run_model_test(test_manager, collector, platform_target, cfg)

    @pytest.mark.optimal
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN3, Platforms.TRN3_A0])
    @pytest.mark.parametrize(
        "cfg",
        _low_rank_cfgs(attention_block_tkg_model_configs.get(ModelTestType.OPTIMAL, [])),
        ids=[cfg.test_id() for cfg in _low_rank_cfgs(attention_block_tkg_model_configs.get(ModelTestType.OPTIMAL, []))],
    )
    def test_optimal(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        cfg: AttnBlkTestConfig,
    ):
        """OPTIMAL: Performance-optimized model configs."""
        self._run_model_test(test_manager, collector, platform_target, cfg)

    @pytest.mark.generality
    @pytest.mark.parametrize(
        "cfg",
        _low_rank_cfgs(attention_block_tkg_model_configs.get(ModelTestType.GENERALITY, [])),
        ids=[
            cfg.test_id() for cfg in _low_rank_cfgs(attention_block_tkg_model_configs.get(ModelTestType.GENERALITY, []))
        ],
    )
    def test_generality(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        cfg: AttnBlkTestConfig,
    ):
        """GENERALITY: Broad coverage model configs."""
        self._run_model_test(test_manager, collector, platform_target, cfg)
