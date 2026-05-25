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
"""PyTorch reference implementation for ring attention backward pass."""

import torch

from ...core.attention.attention_bwd_torch import attention_bwd_torch_ref, compute_o_lse


def _full_attention_bwd(q, k, v, dy, scale, causal=False):
    """
    Run attention_bwd_torch_ref on full (concatenated) tensors.

    Converts from (bs, seqlen, d) layout to kernel layout (bs, 1, d, seqlen),
    calls the existing torch ref, and converts back.

    Args:
        q (torch.Tensor): (bs, seqlen_q, d)
        k (torch.Tensor): (bs, seqlen_k, d)
        v (torch.Tensor): (bs, seqlen_k, d)
        dy (torch.Tensor): (bs, seqlen_q, d)
        scale (float): Softmax scale factor.
        causal (bool): Whether to apply causal masking.

    Returns:
        tuple: (dq, dk, dv) each (bs, seqlen, d)
    """
    bs, seqlen_q, d = q.shape
    seqlen_k = k.shape[1]

    # Convert to kernel layout: (bs, 1, d, seqlen)
    q_k = q.permute(0, 2, 1).unsqueeze(1)
    k_k = k.permute(0, 2, 1).unsqueeze(1)
    v_k = v.permute(0, 2, 1).unsqueeze(1)
    dy_k = dy.permute(0, 2, 1).unsqueeze(1)

    # Compute O and LSE via existing torch ref
    o_k, lse_k, _ = compute_o_lse(q_k, k_k, v_k, False, True, softmax_scale=scale)

    result = attention_bwd_torch_ref(
        q_ref=q_k,
        k_ref=k_k,
        v_ref=v_k,
        o_ref=o_k,
        dy_ref=dy_k,
        lse_ref=lse_k,
        use_causal_mask=causal,
        mixed_precision=True,
        softmax_scale=scale,
    )

    # Convert back to (bs, seqlen, d)
    dq = result["out_dq_ref"].squeeze(1).permute(0, 2, 1)
    dk = result["out_dk_ref"].squeeze(1).permute(0, 2, 1)
    dv = result["out_dv_ref"].squeeze(1).permute(0, 2, 1)
    return dq, dk, dv


def ring_attention_spmd_bwd_torch_ref(
    q_shards,
    k_shards,
    v_shards,
    dy_shards,
    scale,
    tp_degree,
    causal=False,
    striped=False,
):
    """
    PyTorch reference for ring attention backward.

    Computes golden dQ, dK, dV by concatenating all shards and running
    full attention backward using the existing attention_bwd_torch_ref.

    Args:
        q_shards (list[torch.Tensor]): Per-rank Q, each (bs, seqlen_per_rank, d).
        k_shards (list[torch.Tensor]): Per-rank K.
        v_shards (list[torch.Tensor]): Per-rank V.
        dy_shards (list[torch.Tensor]): Per-rank dY.
        scale (float): Softmax scale factor.
        tp_degree (int): Number of ranks.
        causal (bool): Whether to apply causal masking.
        striped (bool): Whether striped attention layout is used.

    Returns:
        tuple: (dq_shards, dk_shards, dv_shards) lists of torch.Tensor per rank.
    """
    seqlen_per_rank = q_shards[0].shape[1]

    if striped:
        full_seqlen = seqlen_per_rank * tp_degree
        bs, _, d = q_shards[0].shape

        q_full = torch.zeros(bs, full_seqlen, d)
        k_full = torch.zeros_like(q_full)
        v_full = torch.zeros_like(q_full)
        dy_full = torch.zeros_like(q_full)
        for rank in range(tp_degree):
            q_full[:, rank::tp_degree, :] = q_shards[rank]
            k_full[:, rank::tp_degree, :] = k_shards[rank]
            v_full[:, rank::tp_degree, :] = v_shards[rank]
            dy_full[:, rank::tp_degree, :] = dy_shards[rank]

        dq_full, dk_full, dv_full = _full_attention_bwd(q_full, k_full, v_full, dy_full, scale, causal=True)

        dq_shards = [dq_full[:, rank::tp_degree, :] for rank in range(tp_degree)]
        dk_shards = [dk_full[:, rank::tp_degree, :] for rank in range(tp_degree)]
        dv_shards = [dv_full[:, rank::tp_degree, :] for rank in range(tp_degree)]
    else:
        q_full = torch.cat(q_shards, dim=1)
        k_full = torch.cat(k_shards, dim=1)
        v_full = torch.cat(v_shards, dim=1)
        dy_full = torch.cat(dy_shards, dim=1)

        dq_full, dk_full, dv_full = _full_attention_bwd(q_full, k_full, v_full, dy_full, scale, causal)

        dq_shards = [dq_full[:, rank * seqlen_per_rank : (rank + 1) * seqlen_per_rank, :] for rank in range(tp_degree)]
        dk_shards = [dk_full[:, rank * seqlen_per_rank : (rank + 1) * seqlen_per_rank, :] for rank in range(tp_degree)]
        dv_shards = [dv_full[:, rank * seqlen_per_rank : (rank + 1) * seqlen_per_rank, :] for rank in range(tp_degree)]

    return dq_shards, dk_shards, dv_shards


def compute_per_rank_o_lse(q_shards, k_shards, v_shards, scale, tp_degree, causal=False, striped=False):
    """
    Compute per-rank O and LSE using full K/V (needed as kernel inputs).

    Args:
        q_shards (list[torch.Tensor]): Per-rank Q, each (bs_flat, seqlen_per_rank, d).
        k_shards (list[torch.Tensor]): Per-rank K.
        v_shards (list[torch.Tensor]): Per-rank V.
        scale (float): Softmax scale factor.
        tp_degree (int): Number of ranks.
        causal (bool): Whether to apply causal masking.
        striped (bool): Whether striped attention layout is used.

    Returns:
        tuple: (o_per_rank, lse_per_rank) lists of numpy arrays in kernel layout.
    """
    seqlen_per_rank = q_shards[0].shape[1]
    full_seqlen = seqlen_per_rank * tp_degree

    if striped:
        k_full = torch.zeros(k_shards[0].shape[0], full_seqlen, k_shards[0].shape[2])
        v_full = torch.zeros_like(k_full)
        for rank in range(tp_degree):
            k_full[:, rank::tp_degree, :] = k_shards[rank]
            v_full[:, rank::tp_degree, :] = v_shards[rank]
    else:
        k_full = torch.cat(k_shards, dim=1)
        v_full = torch.cat(v_shards, dim=1)

    o_per_rank = []
    lse_per_rank = []
    for rank in range(tp_degree):
        q_t = q_shards[rank].permute(0, 2, 1).unsqueeze(1).float()
        k_t = k_full.permute(0, 2, 1).unsqueeze(1).float()
        v_t = v_full.permute(0, 2, 1).unsqueeze(1).float()

        if causal:
            if striped:
                q_pos = torch.arange(seqlen_per_rank).unsqueeze(1) * tp_degree + rank
            else:
                q_pos = torch.arange(rank * seqlen_per_rank, (rank + 1) * seqlen_per_rank).unsqueeze(1)
            k_pos = torch.arange(full_seqlen).unsqueeze(0)
            causal_bias = torch.where(q_pos >= k_pos, 0.0, float("-inf")).unsqueeze(0).unsqueeze(0)
            o_proj, lse, _ = compute_o_lse(q_t, k_t, v_t, False, True, softmax_scale=scale, logit_bias=causal_bias)
        else:
            o_proj, lse, _ = compute_o_lse(q_t, k_t, v_t, False, True, softmax_scale=scale)

        o_per_rank.append(o_proj.numpy())
        lse_per_rank.append(lse.numpy())

    return o_per_rank, lse_per_rank
