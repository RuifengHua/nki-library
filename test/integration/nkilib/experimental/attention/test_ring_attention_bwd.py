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
"""Tests for ring attention backward kernel."""

import math

import numpy as np
import pytest
import torch

from nkilib_src.nkilib.experimental.attention.ring_attention_bwd import ring_attention_spmd_bwd
from nkilib_src.nkilib.experimental.attention.ring_attention_bwd_torch import (
    compute_per_rank_o_lse,
    ring_attention_spmd_bwd_torch_ref,
)
from test.utils.common_dataclasses import (
    CompilerArgs,
    InferenceArgs,
    Platforms,
)
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import CollectiveUnitTestFramework


@pytest_test_metadata(
    name="RingAttentionBwd",
    pytest_marks=["collectives", "ring_attention"],
)
class TestRingAttentionBwd:
    """Integration tests for ring attention backward kernel."""

    @pytest.mark.parametrize(
        "batch, nheads, seqlen, d, tp_degree, lnc, causal, striped",
        [
            # Non-causal configs
            pytest.param(1, 2, 8192, 128, 4, 2, False, False, id="bs1_nh2_s8192_d128_tp4_lnc2_nocausal"),
            pytest.param(2, 2, 8192, 128, 4, 2, False, False, id="bs2_nh2_s8192_d128_tp4_lnc2_nocausal"),
            pytest.param(3, 3, 4096, 128, 4, 2, False, False, id="bs3_nh3_s4096_d128_tp4_lnc2_nocausal"),
            # Causal configs
            pytest.param(1, 2, 8192, 128, 4, 2, True, False, id="bs1_nh2_s8192_d128_tp4_lnc2_causal"),
            pytest.param(2, 2, 8192, 128, 4, 2, True, False, id="bs2_nh2_s8192_d128_tp4_lnc2_causal"),
            pytest.param(3, 1, 8192, 128, 4, 2, True, False, id="bs3_nh1_s8192_d128_tp4_lnc2_causal"),
            pytest.param(3, 3, 4096, 128, 4, 2, True, False, id="bs3_nh3_s4096_d128_tp4_lnc2_causal"),
            # Striped causal configs
            pytest.param(1, 2, 8192, 128, 4, 2, True, True, id="bs1_nh2_s8192_d128_tp4_lnc2_striped"),
            # 32K seqlen configs (per-shard seqlen = 8192, equivalent to flash attn baseline)
            pytest.param(1, 2, 32768, 128, 4, 2, True, True, id="bs1_nh2_s32768_d128_tp4_lnc2_causal"),
            pytest.param(1, 2, 32768, 128, 4, 2, False, False, id="bs1_nh2_s32768_d128_tp4_lnc2_nocausal"),
        ],
    )
    def test_ring_attention_spmd_bwd(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch: int,
        nheads: int,
        seqlen: int,
        d: int,
        tp_degree: int,
        lnc: int,
        causal: bool,
        striped: bool,
    ):
        """Test ring attention backward pass against reference."""
        np.random.seed(42)
        scale = 1.0 / math.sqrt(d)
        seqlen_per_rank = seqlen // tp_degree
        bs_flat = batch * nheads

        # Generate per-rank input data
        if striped:
            q_full = np.random.randn(bs_flat, seqlen, d).astype(np.float32)
            k_full = np.random.randn(bs_flat, seqlen, d).astype(np.float32)
            v_full = np.random.randn(bs_flat, seqlen, d).astype(np.float32)
            dy_full = np.random.randn(bs_flat, seqlen, d).astype(np.float32)
            q_all = [q_full[:, rank::tp_degree, :] for rank in range(tp_degree)]
            k_all = [k_full[:, rank::tp_degree, :] for rank in range(tp_degree)]
            v_all = [v_full[:, rank::tp_degree, :] for rank in range(tp_degree)]
            dy_all = [dy_full[:, rank::tp_degree, :] for rank in range(tp_degree)]
        else:
            q_all = [np.random.randn(bs_flat, seqlen_per_rank, d).astype(np.float32) for _ in range(tp_degree)]
            k_all = [np.random.randn(bs_flat, seqlen_per_rank, d).astype(np.float32) for _ in range(tp_degree)]
            v_all = [np.random.randn(bs_flat, seqlen_per_rank, d).astype(np.float32) for _ in range(tp_degree)]
            dy_all = [np.random.randn(bs_flat, seqlen_per_rank, d).astype(np.float32) for _ in range(tp_degree)]

        # Compute golden gradients using torch reference
        q_torch = [torch.from_numpy(q) for q in q_all]
        k_torch = [torch.from_numpy(k) for k in k_all]
        v_torch = [torch.from_numpy(v) for v in v_all]
        dy_torch = [torch.from_numpy(dy) for dy in dy_all]
        dq_golden_t, dk_golden_t, dv_golden_t = ring_attention_spmd_bwd_torch_ref(
            q_torch,
            k_torch,
            v_torch,
            dy_torch,
            scale,
            tp_degree,
            causal=causal,
            striped=striped,
        )
        dq_golden = [dq.numpy() for dq in dq_golden_t]
        dk_golden = [dk.numpy() for dk in dk_golden_t]
        dv_golden = [dv.numpy() for dv in dv_golden_t]

        # Compute per-rank O and LSE (needed as kernel inputs)
        o_per_rank, lse_per_rank = compute_per_rank_o_lse(
            q_torch,
            k_torch,
            v_torch,
            scale,
            tp_degree,
            causal=causal,
            striped=striped,
        )

        replica_groups = (tuple(range(tp_degree)),)

        def _to_kernel_layout(arr, rank_id):
            """(bs_flat, seqlen_per_rank, d) -> (bs, nheads, d, seqlen_per_rank)."""
            return arr[rank_id].transpose(0, 2, 1).reshape(batch, nheads, d, seqlen_per_rank)

        def _o_lse_to_kernel_layout(o_arr, lse_arr, rank_id):
            """Reshape compute_o_lse outputs from (bs_flat,1,...) to (bs,nheads,...)."""
            o_out = o_arr[rank_id].reshape(batch, nheads, d, seqlen_per_rank)
            lse_out = lse_arr[rank_id].reshape(batch, nheads, 128, seqlen_per_rank // 128)
            return o_out, lse_out

        def create_inputs(rank_id: int):
            o_rank, lse_rank = _o_lse_to_kernel_layout(o_per_rank, lse_per_rank, rank_id)
            inputs = {
                "q_ref": _to_kernel_layout(q_all, rank_id).astype(np.float16),
                "k_ref": _to_kernel_layout(k_all, rank_id).astype(np.float16),
                "v_ref": _to_kernel_layout(v_all, rank_id).astype(np.float16),
                "o_ref": o_rank.astype(np.float16),
                "dy_ref": _to_kernel_layout(dy_all, rank_id).astype(np.float16),
                "lse_ref": lse_rank.astype(np.float32),
                "use_causal_mask": causal,
                "mixed_precision": True,
                "softmax_scale": scale,
                "num_workers": tp_degree,
                "lnc_size": lnc,
                "replica_groups": replica_groups,
            }
            if striped:
                inputs["striped_attention"] = True
            return inputs

        def create_golden(rank_id: int):
            return {
                "out_dq_ref": _to_kernel_layout(dq_golden, rank_id).astype(np.float32),
                "out_dk_ref": _to_kernel_layout(dk_golden, rank_id).astype(np.float32),
                "out_dv_ref": _to_kernel_layout(dv_golden, rank_id).astype(np.float32),
            }

        _state = {}

        def _override(rank_id, raw_input):
            _state["rank_id"] = rank_id
            return raw_input

        def _torch_ref(
            q_ref,
            k_ref,
            v_ref,
            o_ref,
            dy_ref,
            lse_ref,
            use_causal_mask=False,
            mixed_precision=True,
            softmax_scale=None,
            num_workers=1,
            lnc_size=1,
            replica_groups=None,
            striped_attention=False,
        ):
            rank_id = _state["rank_id"]
            return create_golden(rank_id)

        env_vars = {"NEURON_RT_ULTRASERVER_MODE": "4"} if platform_target.is_trn3() else None
        framework = CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=ring_attention_spmd_bwd,
            torch_ref=_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=tp_degree,
            per_rank_torch_ref_input_override=_override,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            rtol=1e-2,
            atol=1e-2,
            inference_args=InferenceArgs(collective_ranks=tp_degree, env_vars=env_vars),
        )
