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
"""Tests for ring attention forward kernel."""

import math

import numpy as np
import pytest

from nkilib_src.nkilib.experimental.attention.ring_attention_fwd import ring_attention_spmd_fwd
from test.utils.common_dataclasses import (
    CompilerArgs,
    Platforms,
)
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import CollectiveUnitTestFramework

# ============================================================
# NumPy reference: attention forward
# ============================================================


def ref_attention_fwd(q, k, v, scale, causal=False):
    """
    Reference attention forward pass.

    Args:
        q: (bs, seqlen_q, d)
        k: (bs, seqlen_k, d)
        v: (bs, seqlen_k, d)
        scale: float, softmax scaling factor

    Returns:
        o: (bs, seqlen_q, d) — attention output
        lse: (bs, seqlen_q) — log-sum-exp per row
    """
    scores = scale * (q @ k.transpose(0, 2, 1))  # (bs, seqlen_q, seqlen_k)
    if causal:
        sq, sk = scores.shape[1], scores.shape[2]
        mask = np.triu(np.ones((sq, sk), dtype=bool), k=1)
        scores[:, mask] = -float("inf")

    row_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - row_max)
    row_sum = exp_scores.sum(axis=-1, keepdims=True)
    softmax_weights = exp_scores / row_sum
    o = softmax_weights @ v
    lse = (row_max + np.log(row_sum)).squeeze(-1)  # (bs, seqlen_q)
    return o, lse


def ref_ring_attention_fwd(q_per_rank, k_per_rank, v_per_rank, scale, causal=False):
    """
    Reference ring attention forward: each rank's Q attends to ALL K/V.

    Args:
        q_per_rank: list of (bs, seqlen_per_rank, d) per rank
        k_per_rank: list of (bs, seqlen_per_rank, d) per rank
        v_per_rank: list of (bs, seqlen_per_rank, d) per rank
        scale: float
        causal: bool

    Returns:
        o_per_rank: list of (bs, seqlen_per_rank, d) per rank
        lse_per_rank: list of (bs, seqlen_per_rank) per rank
    """
    k_full = np.concatenate(k_per_rank, axis=1)
    v_full = np.concatenate(v_per_rank, axis=1)
    q_full = np.concatenate(q_per_rank, axis=1)

    o_full, lse_full = ref_attention_fwd(q_full, k_full, v_full, scale, causal)

    num_ranks = len(q_per_rank)
    seqlen_per_rank = q_per_rank[0].shape[1]
    o_per_rank = [o_full[:, r * seqlen_per_rank : (r + 1) * seqlen_per_rank, :] for r in range(num_ranks)]
    lse_per_rank = [lse_full[:, r * seqlen_per_rank : (r + 1) * seqlen_per_rank] for r in range(num_ranks)]

    return o_per_rank, lse_per_rank


def _make_ring_attention_torch_ref(per_rank_input_generator, collective_ranks):
    """Build a torch_ref + input override pair for ring attention.

    Ring attention golden requires all ranks' K/V, but the framework calls
    torch_ref per-rank with only that rank's inputs. This factory returns a
    closure that internally gathers all ranks' data from per_rank_input_generator.

    The rank_id is communicated via a side-channel dict set by the
    per_rank_torch_ref_input_override callback before each torch_ref call.

    Returns:
        (torch_ref, per_rank_torch_ref_input_override) tuple.
    """
    _state = {}  # side-channel: set by override, read by torch_ref

    def _override(rank_id, raw_input):
        _state["rank_id"] = rank_id
        return raw_input

    def _torch_ref(
        q,
        k,
        v,
        replica_groups=None,
        num_workers=1,
        softmax_scale=None,
        use_causal_mask=False,
        striped_input=False,
        training=False,
        lse_dtype=None,
        tp_q=False,
        tp_k=False,
    ):
        rank_id = _state["rank_id"]

        # Find this rank's replica group and position within it
        my_group = next(g for g in replica_groups if rank_id in g)
        my_worker_idx = list(my_group).index(rank_id)

        def _to_ref_3d(arr, transposed):
            """Kernel 4-D layout -> ref (bs*h, spr, d)."""
            if transposed:
                # (bs, h, d, spr) -> (bs, h, spr, d)
                arr = arr.transpose(0, 1, 3, 2)
            bs, h, spr, d = arr.shape
            return arr.reshape(bs * h, spr, d).astype(np.float32)

        # Gather all workers' Q/K/V in this replica group in ref layout
        q_workers, k_workers, v_workers = [], [], []
        for worker_rank in my_group:
            inp = per_rank_input_generator(worker_rank)
            q_workers.append(_to_ref_3d(inp["q"], transposed=not inp.get("tp_q", False)))
            k_workers.append(_to_ref_3d(inp["k"], transposed=not inp.get("tp_k", False)))
            v_workers.append(_to_ref_3d(inp["v"], transposed=False))

        # Undo pre-scaling on Q when causal (test pre-scales Q and passes scale=1.0)
        actual_scale = softmax_scale
        if use_causal_mask and softmax_scale == 1.0:
            d = q_workers[0].shape[-1]
            actual_scale = 1.0 / math.sqrt(d)
            q_workers = [qw / actual_scale for qw in q_workers]

        # Compute reference attention
        if striped_input:
            nw = len(my_group)
            bs_h, spr, d = q_workers[0].shape
            seqlen = spr * nw
            q_global = np.empty((bs_h, seqlen, d), dtype=np.float32)
            k_global = np.empty_like(q_global)
            v_global = np.empty_like(q_global)
            for w in range(nw):
                q_global[:, w::nw, :] = q_workers[w]
                k_global[:, w::nw, :] = k_workers[w]
                v_global[:, w::nw, :] = v_workers[w]
            o_global, lse_global = ref_attention_fwd(q_global, k_global, v_global, actual_scale, causal=True)
            o_ref = o_global[:, my_worker_idx::nw, :]
            lse_ref = lse_global[:, my_worker_idx::nw]
        else:
            o_all, lse_all = ref_ring_attention_fwd(
                q_workers, k_workers, v_workers, actual_scale, causal=use_causal_mask
            )
            o_ref = o_all[my_worker_idx]
            lse_ref = lse_all[my_worker_idx]

        # Convert to kernel output layout
        bs, h = q.shape[0], q.shape[1]
        spr = o_ref.shape[1]
        d = o_ref.shape[2]
        out_o = o_ref.reshape(bs, h, spr, d).astype(np.float16)
        lse_2d = lse_ref.reshape(bs, h, spr)
        out_lse = lse_2d.reshape(bs, h, spr // 128, 128).transpose(0, 1, 3, 2).astype(np.float32)

        return {"out_o": out_o, "out_lse": out_lse}

    return _torch_ref, _override


# ============================================================
# Integration test: ring attention forward on hardware
# ============================================================


@pytest_test_metadata(
    name="RingAttentionFwd",
    pytest_marks=["collectives", "ring_attention"],
)
class TestRingAttentionFwd:
    """Integration tests for ring attention forward kernel."""

    @pytest.mark.parametrize(
        "bs, nheads, nkv_heads, seqlen_per_rank, d, tp_degree, lnc, causal, striped",
        [
            # ──── Non-causal, MHA ────
            pytest.param(2, 2, 2, 4096, 128, 2, 1, False, False, id="nocausal_mha_seqsmall_tp2_lnc1"),
            pytest.param(2, 2, 2, 4096, 128, 2, 2, False, False, id="nocausal_mha_seqsmall_tp2_lnc2_even"),
            # ──── Causal contiguous, MHA ────
            pytest.param(2, 2, 2, 4096, 128, 2, 1, True, False, id="causal_contig_mha_seqsmall_tp2_lnc1"),
            pytest.param(2, 2, 2, 4096, 128, 2, 2, True, False, id="causal_contig_mha_seqsmall_tp2_lnc2_even"),
            # ──── Causal striped, MHA ────
            pytest.param(2, 2, 2, 4096, 128, 2, 1, True, True, id="causal_striped_mha_seqsmall_tp2_lnc1"),
            pytest.param(2, 2, 2, 4096, 128, 2, 2, True, True, id="causal_striped_mha_seqsmall_tp2_lnc2_even"),
            # ──── LNC=2 odd cases (bs * nheads is odd) ────
            pytest.param(3, 1, 1, 4096, 128, 2, 2, False, False, id="nocausal_mha_seqsmall_tp2_lnc2_odd_bs3"),
            pytest.param(1, 3, 3, 4096, 128, 2, 2, True, False, id="causal_contig_mha_seqlarge_tp2_lnc2_odd"),
            # ──── Non-causal, MHA (large seqlen_per_rank, above 10k FA threshold) ────
            pytest.param(1, 1, 1, 1024 * 10, 128, 4, 1, False, False, id="nocausal_mha_seqlarge_tp4_lnc1"),
            pytest.param(1, 2, 2, 1024 * 10, 128, 4, 2, False, False, id="nocausal_mha_seqlarge_tp4_lnc2_even"),
            # ──── Causal contiguous, MHA (large seqlen_per_rank) ────
            pytest.param(1, 1, 1, 1024 * 10, 128, 4, 1, True, False, id="causal_contig_mha_seqlarge_tp4_lnc1"),
            pytest.param(1, 2, 2, 1024 * 10, 128, 4, 2, True, False, id="causal_contig_mha_seqlarge_tp4_lnc2_even"),
            # ──── Causal striped, MHA (large seqlen_per_rank) ────
            pytest.param(1, 1, 1, 1024 * 10, 128, 4, 1, True, True, id="causal_striped_mha_seqlarge_tp4_lnc1"),
            pytest.param(1, 2, 2, 1024 * 10, 128, 4, 2, True, True, id="causal_striped_mha_seqlarge_tp4_lnc2_even"),
        ],
    )
    def test_ring_attention_spmd_fwd(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        bs: int,
        nheads: int,
        nkv_heads: int,
        seqlen_per_rank: int,
        d: int,
        tp_degree: int,
        lnc: int,
        causal: bool,
        striped: bool,
    ):
        """Test ring attention forward pass against reference."""
        np.random.seed(42)
        scale = 1.0 / math.sqrt(d)

        # The kernel expects q_h == k_h (pre-broadcasted for GQA).
        # We generate data at the KV-head granularity, then broadcast Q heads.
        q_h_per_kv_h = nheads // nkv_heads
        # bs_flat folds KV heads into batch: each (batch, kv_head) pair is independent
        bs_flat = bs * nkv_heads

        if striped:
            # Generate global data in natural position order
            seqlen = seqlen_per_rank * tp_degree
            q_global = np.random.randn(bs_flat, seqlen, d).astype(np.float32)
            k_global = np.random.randn(bs_flat, seqlen, d).astype(np.float32)
            v_global = np.random.randn(bs_flat, seqlen, d).astype(np.float32)

            # Stripe-slice: rank r gets positions [r, r+tp, r+2*tp, ...]
            q_per_rank = [q_global[:, r::tp_degree, :] for r in range(tp_degree)]
            k_per_rank = [k_global[:, r::tp_degree, :] for r in range(tp_degree)]
            v_per_rank = [v_global[:, r::tp_degree, :] for r in range(tp_degree)]
        else:
            # Contiguous: generate per-rank data independently
            q_per_rank = [np.random.randn(bs_flat, seqlen_per_rank, d).astype(np.float32) for _ in range(tp_degree)]
            k_per_rank = [np.random.randn(bs_flat, seqlen_per_rank, d).astype(np.float32) for _ in range(tp_degree)]
            v_per_rank = [np.random.randn(bs_flat, seqlen_per_rank, d).astype(np.float32) for _ in range(tp_degree)]

        replica_groups = (tuple(range(tp_degree)),)

        def _to_kernel_layout_q(rank_id):
            """(bs_flat, seqlen_per_rank, d) -> (bs, nheads, d, seqlen_per_rank).

            Broadcast each KV head q_h_per_kv_h times to fill all Q heads.
            The kernel requires q_h == k_h, so after broadcast nheads == nheads.
            """
            arr = q_per_rank[rank_id]  # (bs_flat, spr, d)
            arr_4d = arr.reshape(bs, nkv_heads, seqlen_per_rank, d)  # (bs, nkv_heads, spr, d)
            arr_broadcast = np.repeat(arr_4d, q_h_per_kv_h, axis=1)  # (bs, nheads, spr, d)
            return arr_broadcast.transpose(0, 1, 3, 2)  # (bs, nheads, d, spr)

        def _to_kernel_layout_k(rank_id, data_per_rank):
            """(bs_flat, seqlen_per_rank, d) -> (bs, nheads, d, seqlen_per_rank).

            Broadcast KV heads to match Q heads (kernel requires q_h == k_h).
            K uses transposed layout (d, seqlen).
            """
            arr = data_per_rank[rank_id]  # (bs_flat, spr, d)
            arr_4d = arr.reshape(bs, nkv_heads, seqlen_per_rank, d)
            arr_broadcast = np.repeat(arr_4d, q_h_per_kv_h, axis=1)  # (bs, nheads, spr, d)
            return arr_broadcast.transpose(0, 1, 3, 2)  # (bs, nheads, d, spr)

        def _to_kernel_layout_v(rank_id, data_per_rank):
            """(bs_flat, seqlen_per_rank, d) -> (bs, nheads, seqlen_per_rank, d).

            Broadcast KV heads to match Q heads (kernel requires q_h == k_h).
            V uses non-transposed layout (seqlen, d).
            """
            arr = data_per_rank[rank_id]  # (bs_flat, spr, d)
            arr_4d = arr.reshape(bs, nkv_heads, seqlen_per_rank, d)
            arr_broadcast = np.repeat(arr_4d, q_h_per_kv_h, axis=1)  # (bs, nheads, spr, d)
            return arr_broadcast  # (bs, nheads, spr, d) — no transpose

        def create_inputs(rank_id: int):
            q_input = _to_kernel_layout_q(rank_id).astype(np.float16)
            kernel_scale = scale

            # When testing pre-scaled Q path: multiply Q by scale on the host
            # and pass softmax_scale=1.0 so the kernel skips its own pre-scaling.
            if causal:
                q_input = (q_input.astype(np.float32) * scale).astype(np.float16)
                kernel_scale = 1.0

            inputs = {
                "q": q_input,
                "k": _to_kernel_layout_k(rank_id, k_per_rank).astype(np.float16),
                "v": _to_kernel_layout_v(rank_id, v_per_rank).astype(np.float16),
                "replica_groups": replica_groups,
                "num_workers": tp_degree,
                "softmax_scale": kernel_scale,
                "use_causal_mask": causal,
                "striped_input": striped,
                "training": True,
            }
            return inputs

        torch_ref, ref_input_override = _make_ring_attention_torch_ref(create_inputs, tp_degree)

        framework = CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=ring_attention_spmd_fwd,
            torch_ref=torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=tp_degree,
            per_rank_torch_ref_input_override=ref_input_override,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
        )

    @pytest.mark.parametrize(
        "bs, nheads, seqlen_per_rank, d, num_workers, total_ranks, lnc, additional_cmd_args",
        [
            # Flux: TP=4, CP=2, 8 total ranks, 4 replica groups of 2
            pytest.param(1, 6, 2304, 128, 2, 8, 1, [], id="flux_tp4_cp2_lnc1"),
            # Pipeline host is trn2.3xl, which has 8 cores with LNC1, but 4 cores if you're using LNC2.
            # TP4/CP2 doesn't work on pipeline with LNC2. Manual test only.
            # pytest.param(1, 6, 2304, 128, 2, 8, 2, [
            #     # "--internal-backend-options=--print-format=condensed",
            #     # "--internal-compiler-debug-mode=all",
            # ], id="flux_tp4_cp2_lnc2"),
        ],
    )
    def test_ring_attention_spmd_fwd_multi_group(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        bs: int,
        nheads: int,
        seqlen_per_rank: int,
        d: int,
        num_workers: int,
        total_ranks: int,
        lnc: int,
        additional_cmd_args: list,
    ):
        """Test ring attention with multiple replica groups (Flux CP layout).

        Flux uses TP=4, DP=2 with context parallelism over the DP dimension.
        This gives 8 total ranks with 4 independent replica groups:
        ((0,4), (1,5), (2,6), (3,7)).
        Each group runs ring attention with num_workers=2 independently.
        """
        np.random.seed(42)
        scale = 1.0 / math.sqrt(d)
        num_groups = total_ranks // num_workers

        # Build replica groups: Flux DP groups pattern
        # ranks [0..tp-1] are TP group 0, [tp..2*tp-1] are TP group 1
        # CP pairs: (i, i + num_groups) for i in range(num_groups)
        replica_groups = tuple(tuple(i + g * num_groups for g in range(num_workers)) for i in range(num_groups))
        # e.g. ((0,4), (1,5), (2,6), (3,7)) for num_workers=2, total_ranks=8

        # Generate independent data for each replica group
        # group_data[group_idx] = (q_per_worker, k_per_worker, v_per_worker)
        group_data = []
        for _ in range(num_groups):
            q_per_worker = [
                np.random.randn(bs * nheads, seqlen_per_rank, d).astype(np.float32) for _ in range(num_workers)
            ]
            k_per_worker = [
                np.random.randn(bs * nheads, seqlen_per_rank, d).astype(np.float32) for _ in range(num_workers)
            ]
            v_per_worker = [
                np.random.randn(bs * nheads, seqlen_per_rank, d).astype(np.float32) for _ in range(num_workers)
            ]
            group_data.append((q_per_worker, k_per_worker, v_per_worker))

        # Map global rank -> (group_idx, worker_idx_within_group)
        rank_to_group = {}
        for group_idx, group in enumerate(replica_groups):
            for worker_idx, rank in enumerate(group):
                rank_to_group[rank] = (group_idx, worker_idx)

        def create_inputs(rank_id: int):
            group_idx, worker_idx = rank_to_group[rank_id]
            q_w, k_w, v_w = group_data[group_idx]

            # (bs*nheads, spr, d) -> (bs, nheads, spr, d) for Q/K/V (non-transposed layout)
            # tp_q=True and tp_k=True let attention_cte handle transpose via dma_transpose
            q_arr = q_w[worker_idx].reshape(bs, nheads, seqlen_per_rank, d)
            k_arr = k_w[worker_idx].reshape(bs, nheads, seqlen_per_rank, d)
            v_arr = v_w[worker_idx].reshape(bs, nheads, seqlen_per_rank, d)

            return {
                "q": q_arr.astype(np.float16),
                "k": k_arr.astype(np.float16),
                "v": v_arr.astype(np.float16),
                "replica_groups": replica_groups,
                "num_workers": num_workers,
                "softmax_scale": scale,
                "use_causal_mask": False,
                "striped_input": False,
                "training": True,
                "tp_q": True,
                "tp_k": True,
            }

        torch_ref, ref_input_override = _make_ring_attention_torch_ref(create_inputs, total_ranks)

        framework = CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=ring_attention_spmd_fwd,
            torch_ref=torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=total_ranks,
            per_rank_torch_ref_input_override=ref_input_override,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=lnc,
                platform_target=platform_target,
                additional_cmd_args=additional_cmd_args,
            ),
        )
