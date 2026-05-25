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

"""Shared torch reference logic for MoE BWMM MX kernels (shard-on-block and shard-on-I).

Implements the per-block MoE loop using MX projection torch refs from mlp_proj_mx_torch.py.
"""

from typing import Any, Optional

import nki.language as nl
import numpy as np
import torch

from ...mlp.mlp_tkg.mlp_proj_mx_torch import (
    down_proj_mx_torch_ref,
    gate_up_proj_mx_torch_ref,
)
from ...utils.common_types import ActFnType, ExpertAffinityScaleMode
from ...utils.mx_torch_common import (
    quantize_to_mx,
    unpack_float4_x4,
    unpack_float8_e4m3fn_x4,
    unpack_float8_e5m2_x4,
)
from .moe_cte_torch_utils import torch_act_fn
from .moe_cte_utils import SkipMode

_pmax = 128
_q_width = 4
_q_height = 8


def _get_unpack_fn(weight_dtype):
    """Return the appropriate unpack function for the given MX weight dtype."""
    dtype_str = str(weight_dtype)
    if 'float4_e2m1fn' in dtype_str:
        return unpack_float4_x4
    elif 'float8_e4m3fn' in dtype_str:
        return unpack_float8_e4m3fn_x4
    elif 'float8_e5m2' in dtype_str:
        return unpack_float8_e5m2_x4
    raise ValueError(f"Unsupported weight dtype: {weight_dtype}")


def _quantize_hidden_to_mx(hidden_np, H, BxS):
    """Quantize raw hidden states to MX float8_e4m3fn format.

    Args:
        hidden_np: numpy float32 [BxS, H]
        H, BxS: dimensions

    Returns:
        (hidden_mx numpy x4 [128, H//512, BxS], hidden_scale numpy uint8 [16, H//512, BxS])
    """
    # Reshape to MX layout: [BxS, 4, H/512, 128] -> transpose -> [128, H/512*BxS*4]
    reshaped = hidden_np.reshape(BxS, _q_width, H // _pmax // _q_width, _pmax)
    transposed = reshaped.transpose(3, 2, 0, 1).reshape(_pmax, -1)
    mx_data, mx_scale = quantize_to_mx(transposed, nl.float8_e4m3fn_x4)
    hidden_mx = mx_data.reshape(_pmax, H // _pmax // _q_width, BxS)
    hidden_scale = mx_scale.reshape(_pmax // _q_height, H // _pmax // _q_width, BxS)
    return hidden_mx, hidden_scale


def _to_numpy(x):
    """Convert torch tensor or numpy array to numpy float32."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.float().numpy()
    return np.asarray(x, dtype=np.float32) if x.dtype != np.float32 else x


def bwmm_mx_blockwise_loop(
    hidden_states: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    gate_up_proj_scale: Optional[torch.Tensor],
    down_proj_scale: Optional[torch.Tensor],
    block_size: int,
    activation_function: ActFnType,
    skip_dma: SkipMode,
    weight_dtype: Any,
    is_tensor_update_accumulating: bool,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode,
    gate_and_up_proj_bias: Optional[torch.Tensor] = None,
    down_proj_bias: Optional[torch.Tensor] = None,
    conditions: Optional[torch.Tensor] = None,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    separate_outputs: bool = False,
) -> dict:
    """Core per-block MoE loop for MX kernels, shared by shard-on-block and shard-on-I.

    Returns:
        dict with 'output' torch.Tensor
    """
    # Convert all inputs to numpy for MX operations
    hidden_states_np = _to_numpy(hidden_states)
    affinities_np = _to_numpy(expert_affinities_masked)

    if isinstance(token_position_to_id, torch.Tensor):
        tok_ids = token_position_to_id.numpy().astype(np.int32)
    else:
        tok_ids = token_position_to_id.astype(np.int32)

    if isinstance(block_to_expert, torch.Tensor):
        b2e = block_to_expert.numpy().astype(np.int32).flatten()
    else:
        b2e = block_to_expert.astype(np.int32).flatten()

    # Ensure weight/scale are numpy (x4 packed types can't be torch tensors)
    # Handle uint16/uint32 (torch tensors from torch_ref_wrapper) or numpy x4 arrays
    def _ensure_numpy_weights(w):
        if isinstance(w, torch.Tensor):
            return w.numpy()
        return w

    gup_w = _ensure_numpy_weights(gate_up_proj_weight)
    down_w = _ensure_numpy_weights(down_proj_weight)
    gup_s = gate_up_proj_scale if isinstance(gate_up_proj_scale, np.ndarray) else gate_up_proj_scale.numpy()
    down_s = down_proj_scale if isinstance(down_proj_scale, np.ndarray) else down_proj_scale.numpy()

    # Handle uint16/uint32 weights (NxD simulation): view back to original x4 dtype
    _UINT_TO_X4 = {np.dtype('uint16'): nl.float4_e2m1fn_x4, np.dtype('uint32'): nl.float8_e4m3fn_x4}
    if gup_w.dtype in _UINT_TO_X4:
        x4_dtype = _UINT_TO_X4.get(gup_w.dtype)
        if weight_dtype is not None:
            x4_np = np.dtype(weight_dtype)
        else:
            x4_np = np.dtype(x4_dtype)
        gup_w = gup_w.view(x4_np)
        down_w = down_w.view(x4_np)

    # Bias: convert to numpy if torch
    gup_bias_np = _to_numpy(gate_and_up_proj_bias)
    down_bias_np = _to_numpy(down_proj_bias)

    E = gup_w.shape[0]
    H = hidden_states_np.shape[-1]
    I_TP = gup_w.shape[-1]
    B = block_size
    T_dim = hidden_states_np.shape[0]
    T = T_dim if skip_dma.skip_token else T_dim - 1

    N = len(tok_ids) // B
    tok_ids_2d = tok_ids.reshape(N, B)

    affinities_2d = affinities_np.reshape(-1, E)

    unpack_fn = _get_unpack_fn(weight_dtype)

    # Output - use bfloat16 to match numpy golden accumulation behavior
    # Use T+1 rows (extra padding row) so -1 index doesn't alias last real token
    lnc_degree = 2
    out_T = T + 1  # always allocate T+1 for accumulation
    if separate_outputs:
        output = np.zeros((lnc_degree, out_T, H), dtype='bfloat16')
    else:
        output = np.zeros((out_T, H), dtype='bfloat16')

    # Working copies for skip_token mode
    hidden_work = hidden_states_np.copy()
    aff_work = affinities_2d.copy()

    for b_idx in range(N):
        if conditions is not None:
            cond = conditions[b_idx] if isinstance(conditions, np.ndarray) else conditions[b_idx].item()
            if cond == 0:
                break

        local_ids = tok_ids_2d[b_idx]
        expert_idx = int(b2e[b_idx])
        # For weight skipping, expert_idx may be >= E (sentinel) but we still
        # need the real expert's weights for computation
        real_expert_idx = expert_idx

        if skip_dma.skip_weight:
            is_same = b_idx > 0 and b2e[b_idx] == b2e[b_idx - 1]
            if is_same:
                real_expert_idx = expert_idx
                expert_idx = E  # sentinel to trigger skip logic below

        if skip_dma.skip_token:
            hidden_work = np.concatenate([hidden_work, np.zeros((1, H), dtype=np.float32)], axis=0)
            aff_work = np.concatenate([aff_work, np.zeros((1, E), dtype=np.float32)], axis=0)

        local_hidden = hidden_work[local_ids].astype(np.float32)  # [B, H]
        local_aff = aff_work[local_ids, real_expert_idx].reshape(-1, 1).astype(np.float32)

        if expert_affinities_scaling_mode in (
            ExpertAffinityScaleMode.PRE_SCALE,
            ExpertAffinityScaleMode.PRE_SCALE_DELAYED,
        ):
            local_hidden = local_aff * local_hidden

        if real_expert_idx >= E:
            continue

        # Gate/up projection via MX torch ref
        hidden_mx, hidden_scale = _quantize_hidden_to_mx(local_hidden, H, B)

        gate_result = gate_up_proj_mx_torch_ref(
            hidden_qtz=hidden_mx,
            hidden_scale=torch.from_numpy(hidden_scale),
            weight_qtz=gup_w[real_expert_idx, :, 0, :, :],  # [128, n_H512, I_TP]
            weight_scale=torch.from_numpy(gup_s[real_expert_idx, :, 0, :, :]),
            bias=torch.from_numpy(gup_bias_np[real_expert_idx, :, 0, :, :]) if gup_bias_np is not None else None,
            H=H,
            I=I_TP,
            BxS=B,
            hidden_unpack_fn=unpack_float8_e4m3fn_x4,
            weight_unpack_fn=unpack_fn,
        )
        up_result = gate_up_proj_mx_torch_ref(
            hidden_qtz=hidden_mx,
            hidden_scale=torch.from_numpy(hidden_scale),
            weight_qtz=gup_w[real_expert_idx, :, 1, :, :],
            weight_scale=torch.from_numpy(gup_s[real_expert_idx, :, 1, :, :]),
            bias=torch.from_numpy(gup_bias_np[real_expert_idx, :, 1, :, :]) if gup_bias_np is not None else None,
            H=H,
            I=I_TP,
            BxS=B,
            hidden_unpack_fn=unpack_float8_e4m3fn_x4,
            weight_unpack_fn=unpack_fn,
        )

        gate_act = gate_result["out"]  # [128, n_I512, B, 4]
        up_act = up_result["out"]

        # Clamp
        if gate_clamp_lower_limit is not None or gate_clamp_upper_limit is not None:
            gate_act = torch.clamp(gate_act, min=gate_clamp_lower_limit, max=gate_clamp_upper_limit)
        if up_clamp_lower_limit is not None or up_clamp_upper_limit is not None:
            up_act = torch.clamp(up_act, min=up_clamp_lower_limit, max=up_clamp_upper_limit)

        # Activation + element-wise multiply
        intermediate = torch_act_fn(gate_act, activation_function) * up_act

        # Down projection via MX torch ref
        down_result = down_proj_mx_torch_ref(
            inter=intermediate,
            weight_qtz=down_w[real_expert_idx],
            weight_scale=torch.from_numpy(down_s[real_expert_idx]),
            bias=torch.from_numpy(down_bias_np[real_expert_idx].reshape(1, H)) if down_bias_np is not None else None,
            H=H,
            I=I_TP,
            BxS=B,
            weight_unpack_fn=unpack_fn,
        )

        down_activation = down_result["out"].numpy()  # [B, H]

        # Scale by expert affinities
        if expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
            scaled = down_activation * local_aff
        else:
            scaled = down_activation

        # Accumulate in bfloat16 to match numpy golden
        scaled_cast = scaled.astype(output.dtype)
        if separate_outputs:
            output[0, local_ids, :] += scaled_cast
        else:
            output[local_ids, :] += scaled_cast

    # Slice output to match expected shape
    if skip_dma.skip_token:
        if separate_outputs:
            output = output[:, :T, :]
        else:
            output = output[:T, :]

    return {"output": torch.from_numpy(output.astype(np.float32))}
