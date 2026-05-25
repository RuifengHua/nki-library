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
Test utilities for MoE BWMM MXFP4/MXFP8 CTE kernel tests.

Provides input builders and golden functions for testing the blockwise
matrix multiplication kernel with MXFP4/MXFP8 quantization.
"""

import hashlib
import math
import os
import pickle
from typing import Optional

import nki.language as nl
import numpy as np

from nkilib_src.nkilib.core.moe.moe_cte.moe_cte_utils import SkipMode
from nkilib_src.nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
)
from nkilib_src.nkilib.core.utils.kernel_assert import kernel_assert
from test.integration.nkilib.core.moe.moe_cte.test_moe_cte_common import (
    generate_token_position_to_id_and_experts,
    get_n_blocks,
    map_skip_mode,
)
from test.integration.nkilib.utils.tensor_generators import generate_stabilized_mx_data

# MXFP4 quantization block dimensions
_q_width = 4  # quantization width
_q_height = 8  # quantization height
_pmax = 128  # sbuf max partition dim (128)

# Explicit parameter ordering per kernel variant, matching kernel function signatures.
# From bwmm_shard_on_block_mx.py::bwmm_shard_on_block_mx
_SHARD_ON_BLOCK_MX_ORDER = [
    'hidden_states',
    'expert_affinities_masked',
    'gate_up_proj_weight',
    'down_proj_weight',
    'token_position_to_id',
    'block_to_expert',
    'conditions',
    'gate_and_up_proj_bias',
    'down_proj_bias',
    'gate_up_proj_scale',
    'down_proj_scale',
    'block_size',
    'n_static_blocks',
    'n_dynamic_blocks',
    'gate_up_activations_T',
    'down_activations',
    'activation_function',
    'skip_dma',
    'compute_dtype',
    'weight_dtype',
    'is_tensor_update_accumulating',
    'expert_affinities_scaling_mode',
    'gate_clamp_upper_limit',
    'gate_clamp_lower_limit',
    'up_clamp_lower_limit',
    'up_clamp_upper_limit',
]

# From bwmm_shard_on_I_mx.py::blockwise_mm_shard_intermediate_mx
_SHARD_ON_I_MX_ORDER = [
    'hidden_states',
    'expert_affinities_masked',
    'gate_up_proj_weight',
    'down_proj_weight',
    'token_position_to_id',
    'block_to_expert',
    'gate_and_up_proj_bias',
    'down_proj_bias',
    'gate_up_proj_scale',
    'down_proj_scale',
    'block_size',
    'activation_function',
    'skip_dma',
    'compute_dtype',
    'weight_dtype',
    'is_tensor_update_accumulating',
    'expert_affinities_scaling_mode',
    'gate_clamp_upper_limit',
    'gate_clamp_lower_limit',
    'up_clamp_lower_limit',
    'up_clamp_upper_limit',
]

# From bwmm_shard_on_I_mx.py::blockwise_mm_shard_intermediate_mx_hybrid
_SHARD_ON_I_MX_HYBRID_ORDER = [
    'conditions',
    'hidden_states',
    'expert_affinities_masked',
    'gate_up_proj_weight',
    'down_proj_weight',
    'token_position_to_id',
    'block_to_expert',
    'gate_and_up_proj_bias',
    'down_proj_bias',
    'gate_up_proj_scale',
    'down_proj_scale',
    'block_size',
    'num_static_block',
    'activation_function',
    'skip_dma',
    'compute_dtype',
    'weight_dtype',
    'is_tensor_update_accumulating',
    'expert_affinities_scaling_mode',
    'gate_clamp_upper_limit',
    'gate_clamp_lower_limit',
    'up_clamp_lower_limit',
    'up_clamp_upper_limit',
]

_KERNEL_INPUT_ORDER = {
    'shard_on_block_mx': _SHARD_ON_BLOCK_MX_ORDER,
    'shard_on_I_mx': _SHARD_ON_I_MX_ORDER,
    'shard_on_I_mx_hybrid': _SHARD_ON_I_MX_HYBRID_ORDER,
}


def order_kernel_input(kernel_input, variant):
    """Reorder kernel_input dict to match kernel function signature ordering.

    Args:
        kernel_input: Dict from build_moe_bwmm_mx_cte.
        variant: One of 'shard_on_block_mx', 'shard_on_I_mx', 'shard_on_I_mx_hybrid'.

    Returns:
        New dict with keys ordered to match the kernel's parameter list.
        Internal keys (prefixed with '_') are excluded.
    """
    key_order = _KERNEL_INPUT_ORDER[variant]
    ordered = {}
    for key in key_order:
        if key in kernel_input:
            ordered[key] = kernel_input[key]
    for key in kernel_input:
        if key not in ordered and not key.startswith('_'):
            ordered[key] = kernel_input[key]
    return ordered


# Golden cache directory (same pattern as test_nki_moe.py)
_GOLDEN_CACHE_DIR = os.path.expanduser('~/unit_test_input_golden_cache/moe_bwmm_mxfp4_cte')


def _compute_golden_cache_key(
    H: int,
    T: int,
    E: int,
    B: int,
    TOPK: int,
    I_TP: int,
    dtype,
    weight_dtype,
    skip_mode: int,
    bias: bool,
    activation_function,
    expert_affinities_scaling_mode,
    is_dynamic: bool,
    vnc_degree: int,
    gate_clamp_upper_limit,
    gate_clamp_lower_limit,
    up_clamp_upper_limit,
    up_clamp_lower_limit,
    alpha,
) -> str:
    """Compute a hash key from test parameters for caching."""
    key_data = (
        H,
        T,
        E,
        B,
        TOPK,
        I_TP,
        str(dtype),
        str(weight_dtype),
        skip_mode,
        bias,
        activation_function.value if hasattr(activation_function, 'value') else activation_function,
        expert_affinities_scaling_mode.value
        if hasattr(expert_affinities_scaling_mode, 'value')
        else expert_affinities_scaling_mode,
        is_dynamic,
        vnc_degree,
        gate_clamp_upper_limit,
        gate_clamp_lower_limit,
        up_clamp_upper_limit,
        up_clamp_lower_limit,
        alpha,
    )
    key_str = str(key_data)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _generate_token_experts_by_count(
    T: int,
    E: int,
    num_non_zero: int,
    alpha: np.float32 = None,
) -> np.ndarray:
    """Generate a [T, E] binary matrix with exactly num_non_zero ones.

    Args:
        T: Number of tokens.
        E: Number of experts.
        num_non_zero: Total number of nonzero (token, expert) entries to place.
        alpha: Skew parameter for expert selection. Larger alpha = more skewed. None = uniform.

    Returns:
        token_experts: [T, E] binary ndarray.
    """
    assert num_non_zero <= T * E, f"num_non_zero ({num_non_zero}) cannot exceed T*E ({T * E})"

    np.random.seed(0)
    token_experts = np.zeros((T, E))

    if alpha is not None and alpha > 0:
        expert_probs = np.random.dirichlet(np.ones(E) * (1.0 / alpha))
    else:
        expert_probs = np.ones(E) / E

    placed = 0
    while placed < num_non_zero:
        t = np.random.randint(0, T)
        e = np.random.choice(E, p=expert_probs)
        if token_experts[t, e] == 0:
            token_experts[t, e] = 1
            placed += 1

    return token_experts


# Input Builder
def build_moe_bwmm_mx_cte_from_model_test_config(
    H: int,
    T: int,
    E: int,
    B: int,
    I_TP: int,
    skewness_pct: float,
    global_top_k: int,
    ep_degree: int,
    dtype=nl.bfloat16,
    weight_dtype=nl.float4_e2m1fn_x4,
    skip_mode: int = 0,
    bias: bool = False,
    activation_function: ActFnType = ActFnType.SiLU,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    is_dynamic: bool = True,
    vnc_degree: int = 2,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    alpha: Optional[float] = None,
    is_shard_on_I: bool = False,
) -> dict:
    """Build input tensors for MoE BWMM MX CTE model test configs using skewness-based routing.

    Unlike build_moe_bwmm_mx_cte which uses TOPK-based routing, this function uses
    skewness_pct, global_top_k, and ep_degree to control expert affinity distribution,
    simulating realistic model routing patterns.

    Args:
        H: Hidden dimension size
        T: Total number of tokens
        E: Number of local experts (after EP sharding)
        B: Block size (tokens per block)
        I_TP: Intermediate size per TP degree
        skewness_pct: Float in [0.0, 1.0] that interpolates between the best-case and
            worst-case number of nonzero expert affinities:

                num_non_zero = best + skewness_pct * (worst - best)

            where:
                best  = T * global_top_k / ep_degree   (perfectly balanced across EP shards)
                worst = T * min(E, global_top_k)        (all tokens routed to every local expert)

            Examples (T=4096, global_top_k=8, total_experts=128):
                EP64 (E=2):
                    best=512, worst=8192
                    skew 0.0 → 512,  skew 0.5 → 4352,  skew 1.0 → 8192
                EP8 (E=16):
                    best=4096, worst=32768
                    skew 0.0 → 4096, skew 1.0 → 32768
        global_top_k: Global top-K experts per token before EP sharding
        ep_degree: Expert parallelism degree
        dtype: Data type for activations
        weight_dtype: Data type for weights
        skip_mode: DMA skip mode (0-3)
        bias: Whether to include bias tensors
        activation_function: Activation function type
        expert_affinities_scaling_mode: Expert affinity scaling mode
        is_dynamic: Whether to use dynamic loop
        vnc_degree: LNC sharding degree
        gate_clamp_upper_limit: Upper clamp limit for gate projection
        gate_clamp_lower_limit: Lower clamp limit for gate projection
        up_clamp_upper_limit: Upper clamp limit for up projection
        up_clamp_lower_limit: Lower clamp limit for up projection
        alpha: Expert distribution skew parameter for _generate_token_experts_by_count
        is_shard_on_I: Whether to use shard-on-I variant

    Returns:
        Dictionary with all kernel input tensors and parameters
    """
    np.random.seed(0)

    dma_skip = map_skip_mode(skip_mode)
    is_block_parallel = not is_shard_on_I

    # Compute N (total blocks) for skewness-based routing
    n_block_per_iter_eff = vnc_degree if is_block_parallel else 1
    N = math.ceil((T * min(E, global_top_k) - (E - 1)) / B) + E - 1
    N = n_block_per_iter_eff * math.ceil(N / n_block_per_iter_eff)

    # Compute num_non_zero expert affinities based on skewness
    best = T * global_top_k // ep_degree
    worst = T * min(E, global_top_k)
    num_non_zero = int(best + skewness_pct * (worst - best))

    # Generate token-expert assignments using count-based method
    token_experts = _generate_token_experts_by_count(T, E, num_non_zero, alpha)

    blocks_per_expert = np.ceil(token_experts.sum(0) / B).astype(np.int32)
    n_padding_block = N - np.sum(blocks_per_expert)
    blocks_per_expert[E - 1] += n_padding_block

    cumulative_blocks_per_expert = np.cumsum(blocks_per_expert)
    block_to_expert = np.arange(E).repeat(blocks_per_expert).astype(np.int32)

    token_position_by_id_and_expert = np.cumsum(token_experts, axis=0)
    expert_block_offsets = cumulative_blocks_per_expert * B
    token_position_by_id_and_expert[:, 1:] += expert_block_offsets[:-1]
    token_position_by_id_and_expert = np.where(token_experts, token_position_by_id_and_expert, 0).astype(np.int32)

    if dma_skip.skip_token:
        token_position_to_id = np.full((int(N * B + 1),), -1)
    else:
        token_position_to_id = np.full((int(N * B + 1),), T)

    tokens_ids = np.arange(T)
    token_position_to_id[token_position_by_id_and_expert] = np.expand_dims(tokens_ids, 1)
    token_position_to_id = token_position_to_id[1:]
    token_position_to_id = token_position_to_id.astype(np.int32)

    # Generate conditions
    if not is_block_parallel:
        conditions = np.ones((N + 1,), dtype=np.int32)
        conditions[-(n_padding_block + 1) :] = 0
    else:
        conditions = np.ones((N + 2,), dtype=np.int32)
        conditions[-(n_padding_block + 2) :] = 0

    num_static_block = math.ceil(math.ceil(T * global_top_k / ep_degree) / B)

    return _build_kernel_input_from_routing(
        H=H,
        T=T,
        E=E,
        B=B,
        I_TP=I_TP,
        expert_masks=token_experts,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        conditions=conditions,
        N=N,
        dma_skip=dma_skip,
        dtype=dtype,
        weight_dtype=weight_dtype,
        bias=bias,
        activation_function=activation_function,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        is_tensor_update_accumulating=min(E, global_top_k) > 1,
        is_dynamic=is_dynamic,
        is_shard_on_I=is_shard_on_I,
        n_static_blocks=num_static_block,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
    )


def build_moe_bwmm_mx_cte(
    H: int,
    T: int,
    E: int,
    B: int,
    TOPK: int,
    I_TP: int,
    dtype=nl.bfloat16,
    weight_dtype=nl.float4_e2m1fn_x4,
    skip_mode: int = 0,
    bias: bool = False,
    activation_function: ActFnType = ActFnType.SiLU,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    is_dynamic: bool = False,
    vnc_degree: int = 2,
    n_dynamic_blocks: int = 55,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    alpha: Optional[float] = None,
    use_cache: bool = False,
    is_shard_on_I: bool = False,
    n_static_blocks: Optional[int] = None,
) -> dict:
    """
    Build input tensors for MoE BWMM MXFP4/MXFP8 CTE kernel testing.

    Args:
        H: Hidden dimension size
        T: Total number of tokens
        E: Number of experts
        B: Block size (tokens per block)
        TOPK: Top-K experts per token
        I_TP: Intermediate size per TP degree
        dtype: Data type for activations
        weight_dtype: Data type for weights (e.g., nl.float4_e2m1fn_x4 for MXFP4,
                     nl.float8_e4m3fn_x4 or nl.float8_e5m2_x4 for MXFP8)
        skip_mode: DMA skip mode (0-3)
        bias: Whether to include bias tensors
        activation_function: Activation function type
        expert_affinities_scaling_mode: Expert affinity scaling mode
        is_dynamic: Whether to use dynamic loop
        vnc_degree: LNC sharding degree
        n_dynamic_blocks: Number of blocks to process with dynamic loop (default: 55)
        gate_clamp_upper_limit: Upper clamp limit for gate projection
        gate_clamp_lower_limit: Lower clamp limit for gate projection
        up_clamp_upper_limit: Upper clamp limit for up projection
        up_clamp_lower_limit: Lower clamp limit for up projection
        alpha: Expert distribution sparsity parameter (None for uniform distribution)
        use_cache: Whether to use cached inputs if available (default: False)

    Returns:
        Dictionary with all kernel input tensors and parameters
    """
    # Check for cached inputs
    cache_key = _compute_golden_cache_key(
        H,
        T,
        E,
        B,
        TOPK,
        I_TP,
        dtype,
        weight_dtype,
        skip_mode,
        bias,
        activation_function,
        expert_affinities_scaling_mode,
        is_dynamic,
        vnc_degree,
        gate_clamp_upper_limit,
        gate_clamp_lower_limit,
        up_clamp_upper_limit,
        up_clamp_lower_limit,
        alpha,
    )
    cache_file = os.path.join(_GOLDEN_CACHE_DIR, f"input_{cache_key}.pkl")

    if use_cache and os.path.exists(cache_file):
        print(f"Found cached inputs in {cache_file}, reusing...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    np.random.seed(0)

    dma_skip = map_skip_mode(skip_mode)
    if is_shard_on_I:
        N = get_n_blocks(T, TOPK, E, B, n_block_per_iter=1)
    else:
        N = get_n_blocks(T, TOPK, E, B, n_block_per_iter=vnc_degree)

    # Generate token assignments
    expert_masks, token_position_to_id, block_to_expert, conditions = generate_token_position_to_id_and_experts(
        T,
        TOPK,
        E,
        B,
        dma_skip,
        N,
        vnc_degree=vnc_degree,
        alpha=alpha,
        is_block_parallel=False if is_shard_on_I else True,
        quantize=weight_dtype,
    )

    kernel_input = _build_kernel_input_from_routing(
        H=H,
        T=T,
        E=E,
        B=B,
        I_TP=I_TP,
        expert_masks=expert_masks,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        conditions=conditions,
        N=N,
        dma_skip=dma_skip,
        dtype=dtype,
        weight_dtype=weight_dtype,
        bias=bias,
        activation_function=activation_function,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        is_tensor_update_accumulating=TOPK != 1,
        is_dynamic=is_dynamic,
        is_shard_on_I=is_shard_on_I,
        n_dynamic_blocks=n_dynamic_blocks,
        n_static_blocks=n_static_blocks,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
    )

    # Cache the generated inputs for future reuse
    if use_cache:
        try:
            os.makedirs(_GOLDEN_CACHE_DIR, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(kernel_input, f)
            print(f"Cached inputs saved to {cache_file}")
        except Exception as e:
            print(f"Warning: Failed to cache inputs to {cache_file}: {e}")

    return kernel_input


def _build_kernel_input_from_routing(
    *,
    H: int,
    T: int,
    E: int,
    B: int,
    I_TP: int,
    expert_masks,
    token_position_to_id,
    block_to_expert,
    conditions,
    N: int,
    dma_skip: SkipMode,
    dtype,
    weight_dtype,
    bias: bool,
    activation_function: ActFnType,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode,
    is_tensor_update_accumulating: bool,
    is_dynamic: bool,
    is_shard_on_I: bool,
    n_dynamic_blocks: int = 55,
    n_static_blocks: Optional[int] = None,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
) -> dict:
    """Build kernel input tensors and dict from pre-computed routing assignments.

    This is the shared implementation used by both build_moe_bwmm_mx_cte (TOPK routing)
    and build_moe_bwmm_mx_cte_from_model_test_config (skewness routing).
    """
    # Calculate MXFP4 tensor dimensions
    kernel_assert(H % (_pmax * _q_width) == 0, f"H must be divisible by {_pmax * _q_width}, got {H}")
    n_H512_tile = H // (_pmax * _q_width)

    kernel_assert(
        I_TP % (_pmax * _q_width) == 0 or (I_TP < (_pmax * _q_width) and I_TP % (_q_height * _q_width) == 0),
        f"I_TP must be divisible by {_pmax * _q_width} or (I_TP < {_pmax * _q_width} and I_TP divisible by {_q_height * _q_width}), got {I_TP}",
    )
    n_I512_tile, r_I512_tile = divmod(I_TP, _pmax * _q_width)
    I_TP_par_dim = _pmax
    if r_I512_tile > 0:
        kernel_assert(n_I512_tile == 0, f"Expected n_I512_tile == 0 when remainder exists, got {n_I512_tile}")
        n_I512_tile = 1
        I_TP_par_dim = r_I512_tile // _q_width

    # Generate hidden states with MXFP4-compatible layout
    # When skip_token is True, we use T tokens; otherwise T+1 (with padding token)
    if dma_skip.skip_token:
        hidden_T = T
    else:
        hidden_T = T + 1  # Include padding token

    hidden_states_fp32, _, _ = generate_stabilized_mx_data(
        mx_dtype=nl.float8_e4m3fn_x4,
        shape=(hidden_T * n_H512_tile * _pmax, _q_width),
        val_range=5,
    )
    hidden_states = (
        hidden_states_fp32.reshape(hidden_T, n_H512_tile, _pmax, _q_width)
        .transpose(0, 3, 1, 2)
        .reshape(hidden_T, H)
        .astype(dtype)
    )

    # Zero out padding token (only when not skipping tokens)
    if not dma_skip.skip_token:
        hidden_states[T, :] = 0

    # Generate expert affinities
    if dma_skip.skip_token:
        expert_affinities_masked = np.random.random_sample([T, E]).astype(dtype)
        expert_affinities_masked = (expert_affinities_masked * expert_masks).astype(dtype)
    else:
        expert_affinities_masked = np.random.random_sample([T + 1, E]).astype(dtype)
        expert_affinities_masked[:T] = (expert_affinities_masked[:T] * expert_masks).astype(dtype)
        expert_affinities_masked[T] = 0  # Zero padding token affinities

    # Generate MXFP4 gate/up projection weights
    gate_up_proj_weights_fp32, gate_up_proj_weights, gate_up_proj_scale = generate_stabilized_mx_data(
        mx_dtype=weight_dtype,
        shape=(E * _pmax, 2 * n_H512_tile * I_TP * _q_width),
    )
    gate_up_proj_weights = gate_up_proj_weights.reshape(E, _pmax, 2, n_H512_tile, I_TP)
    gate_up_proj_scale = gate_up_proj_scale.reshape(E, _pmax // _q_height, 2, n_H512_tile, I_TP)

    # Generate MXFP4 down projection weights
    down_proj_weights_fp32, down_proj_weights, down_proj_scale = generate_stabilized_mx_data(
        mx_dtype=weight_dtype,
        shape=(E * I_TP_par_dim, n_I512_tile * H * _q_width),
    )
    down_proj_weights = down_proj_weights.reshape(E, I_TP_par_dim, n_I512_tile, H)
    down_proj_scale = down_proj_scale.reshape(E, I_TP_par_dim // _q_height, n_I512_tile, H)

    # Build kernel input dictionary in exact KLIR test order
    # Order must match build_blockwise_mm input_list:
    # [hidden_states, expert_affinities, gate_and_up_proj_weights, down_proj_weights,
    #  token_position_to_id, block_to_expert]
    # then: conditions (if dynamic), bias tensors (if bias), scale tensors (if quantize)

    kernel_input = {
        'hidden_states': hidden_states,
        'expert_affinities_masked': expert_affinities_masked.reshape(-1, 1),
        'gate_up_proj_weight': gate_up_proj_weights,
        'down_proj_weight': down_proj_weights,
        'block_size': B,
        'token_position_to_id': token_position_to_id,
        'block_to_expert': block_to_expert,
        'skip_dma': dma_skip,
        'compute_dtype': dtype,
        'is_tensor_update_accumulating': is_tensor_update_accumulating,
        'expert_affinities_scaling_mode': expert_affinities_scaling_mode,
    }

    if is_dynamic and not is_shard_on_I:
        kernel_input['n_dynamic_blocks'] = n_dynamic_blocks
    if n_static_blocks is not None:
        if is_shard_on_I:
            kernel_input['num_static_block'] = n_static_blocks
        else:
            kernel_input['n_static_blocks'] = n_static_blocks

    # Add clamp limits only if they have non-None values
    if gate_clamp_upper_limit is not None:
        kernel_input['gate_clamp_upper_limit'] = gate_clamp_upper_limit
    if gate_clamp_lower_limit is not None:
        kernel_input['gate_clamp_lower_limit'] = gate_clamp_lower_limit
    if up_clamp_lower_limit is not None:
        kernel_input['up_clamp_lower_limit'] = up_clamp_lower_limit
    if up_clamp_upper_limit is not None:
        kernel_input['up_clamp_upper_limit'] = up_clamp_upper_limit

    # Add activation function after clamp limits
    kernel_input['activation_function'] = activation_function

    # Add weight_dtype to specify target MXFP format
    kernel_input['weight_dtype'] = weight_dtype

    # Add dynamic conditions BEFORE bias (matches build_blockwise_mm order)
    if is_dynamic:
        kernel_input['conditions'] = conditions

    # Add bias tensors (matches build_blockwise_mm order: gate_and_up_proj_bias, down_proj_bias)
    if bias:
        gate_and_up_proj_bias = np.random.uniform(
            -2.0625, 0.52, size=(E, I_TP_par_dim, 2, n_I512_tile, _q_width)
        ).astype(dtype)
        down_proj_bias = np.random.uniform(-1.632, 1.4375, size=[E, H]).astype(dtype)
        kernel_input['gate_and_up_proj_bias'] = gate_and_up_proj_bias
        kernel_input['down_proj_bias'] = down_proj_bias
    else:
        kernel_input['gate_and_up_proj_bias'] = None
        kernel_input['down_proj_bias'] = None

    # Add scale tensors AFTER bias (matches build_blockwise_mm order)
    kernel_input['gate_up_proj_scale'] = gate_up_proj_scale
    kernel_input['down_proj_scale'] = down_proj_scale

    # Store additional data needed for golden computation
    kernel_input['_internal'] = {
        'gate_up_proj_weights_fp32': gate_up_proj_weights_fp32,
        'down_proj_weights_fp32': down_proj_weights_fp32,
        'expert_masks': expert_masks,
        'N': N,
        'n_H512_tile': n_H512_tile,
        'n_I512_tile': n_I512_tile,
        'I_TP_par_dim': I_TP_par_dim,
    }

    return kernel_input
