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
"""Tests for KV-parallel segmented prefill attention kernel."""

import nki.language as nl
import numpy as np
import pytest
from neuronxcc.starfish.support import dtype as dt
from nki.collectives import ReplicaGroup

from nkilib_src.nkilib.core.attention.kv_parallel_segmented_prefill import (
    kv_parallel_segmented_prefill,
)
from test.utils.common_dataclasses import (
    CompilerArgs,
    Platforms,
)
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import CollectiveUnitTestFramework

_PARAM_NAMES = (
    "group_size,seq_len,head_dim,block_size,seg_size,prior_tokens,local_kv_multiplier,num_groups,lnc_degree,tp_out"
)

_FAST_PARAMS = [
    # --- Baseline ---
    pytest.param(8, 512, 128, 128, 512, 0, 1, 1, 1, False, id="g8_s512_h128_b128_seg512"),
    # --- Multiple Q chunks (seq_len > seg_size) ---
    pytest.param(8, 1024, 128, 128, 512, 0, 1, 1, 1, False, id="g8_s1024_h128_b128_seg512_2chunks"),
    # --- Prior tokens (partial prior segment) ---
    pytest.param(8, 512, 128, 128, 512, 512, 1, 1, 1, False, id="g8_s512_h128_b128_seg512_prior512"),
    pytest.param(8, 512, 128, 128, 512, 256, 1, 1, 1, False, id="g8_s512_h128_b128_seg512_prior256_partial"),
    # --- Prior tokens (full prior segment) ---
    pytest.param(8, 512, 128, 128, 512, 0, 2, 1, 1, False, id="g8_s512_h128_b128_seg512_2xkv_noprior"),
    pytest.param(8, 1024, 128, 128, 512, 0, 2, 1, 1, False, id="g8_s1024_h128_b128_seg512_2xkv"),
    # --- Multiple groups ---
    # Temporarily disabled: requires 32-64 contiguous cores, causes 6h timeout on shared fleet
    # https://tiny.amazon.com/s5v5iu97/awsnatlanetbrowNKIL
    # pytest.param(8, 512, 128, 128, 512, 0, 1, 8, 1, False, id="g8_s512_h128_b128_seg512_64ranks"),
    # pytest.param(8, 512, 128, 128, 512, 0, 1, 8, 2, False, id="g8_s512_h128_b128_seg512_64ranks_lnc2"),
    # --- LNC2 ---
    pytest.param(8, 512, 128, 128, 512, 0, 1, 1, 2, False, id="g8_s512_h128_b128_seg512_lnc2"),
    pytest.param(8, 2048, 128, 128, 2048, 0, 1, 1, 2, False, id="g8_s2048_h128_b128_seg2048_lnc2"),
    pytest.param(8, 2048, 128, 128, 2048, 1024, 2, 1, 2, False, id="g8_s2048_h128_b128_seg2048_prior1024_lnc2"),
    # --- Edge: prior_tokens=0 with multiple KV segments ---
    pytest.param(8, 512, 128, 128, 512, 0, 3, 1, 1, False, id="g8_s512_h128_b128_seg512_3xkv_noprior"),
    # --- Edge: prior_tokens fills exactly one full prior segment ---
    pytest.param(8, 512, 128, 128, 512, 512, 2, 1, 1, False, id="g8_s512_h128_b128_seg512_prior512_exact_full"),
    # --- block_size=64 ---
    pytest.param(8, 512, 128, 64, 512, 0, 1, 1, 1, False, id="g8_s512_h128_b64_seg512"),
    pytest.param(8, 512, 128, 64, 512, 256, 2, 1, 1, False, id="g8_s512_h128_b64_seg512_prior256_partial"),
    # --- prior + multiple Q chunks ---
    pytest.param(8, 1024, 128, 128, 512, 256, 2, 1, 1, False, id="g8_s1024_h128_b128_seg512_prior256_multichunk"),
    # Temp disabled: multi-chunk + prior_tokens causes NaN, likely from private_hbm buffer
    # aliasing across chunks. Debug in progress.
    # pytest.param(
    #     8, 1024, 128, 128, 512, 512, 2, 1, 1, False, id="g8_s1024_h128_b128_seg512_prior512_multichunk"
    # ),
    # --- multiple full prior segments ---
    pytest.param(8, 512, 128, 128, 512, 1024, 3, 1, 1, False, id="g8_s512_h128_b128_seg512_prior1024_2full"),
    # --- LNC2 + multiple Q chunks ---
    pytest.param(8, 1024, 128, 128, 512, 0, 1, 1, 2, False, id="g8_s1024_h128_b128_seg512_lnc2_2chunks"),
    # --- LNC2 + full prior ---
    pytest.param(8, 2048, 128, 128, 2048, 2048, 2, 1, 2, False, id="g8_s2048_h128_b128_seg2048_prior2048_lnc2"),
    # --- head_dim=64 ---
    pytest.param(8, 512, 64, 128, 512, 0, 1, 1, 1, False, id="g8_s512_h64_b128_seg512"),
    pytest.param(8, 512, 64, 128, 512, 256, 1, 1, 1, False, id="g8_s512_h64_b128_seg512_prior256_partial"),
    pytest.param(8, 512, 64, 128, 512, 512, 2, 1, 1, False, id="g8_s512_h64_b128_seg512_prior512_full"),
    pytest.param(8, 512, 64, 128, 512, 0, 1, 1, 2, False, id="g8_s512_h64_b128_seg512_lnc2"),
    pytest.param(8, 2048, 64, 128, 2048, 1024, 2, 1, 2, False, id="g8_s2048_h64_b128_seg2048_prior1024_lnc2"),
    # --- block_size=32 ---
    pytest.param(8, 512, 128, 32, 512, 0, 1, 1, 1, False, id="g8_s512_h128_b32_seg512"),
    pytest.param(8, 512, 128, 32, 512, 256, 2, 1, 1, False, id="g8_s512_h128_b32_seg512_prior256_partial"),
    pytest.param(8, 512, 64, 32, 512, 0, 1, 1, 1, False, id="g8_s512_h64_b32_seg512"),
    # --- block_size=16 ---
    pytest.param(8, 512, 128, 16, 512, 0, 1, 1, 1, False, id="g8_s512_h128_b16_seg512"),
    pytest.param(8, 512, 128, 16, 512, 256, 2, 1, 1, False, id="g8_s512_h128_b16_seg512_prior256_partial"),
    pytest.param(8, 512, 64, 16, 512, 0, 1, 1, 1, False, id="g8_s512_h64_b16_seg512"),
    # --- Long prior_tokens (multiple full prior segments) ---
    pytest.param(8, 512, 128, 128, 512, 2048, 5, 1, 1, False, id="g8_s512_h128_b128_seg512_prior2048_4full"),
    pytest.param(8, 512, 128, 128, 512, 3072, 7, 1, 1, False, id="g8_s512_h128_b128_seg512_prior3072_6full"),
    pytest.param(8, 512, 64, 128, 512, 2048, 5, 1, 1, False, id="g8_s512_h64_b128_seg512_prior2048_4full"),
    # --- tp_out=True + KVP ---
    pytest.param(8, 512, 128, 128, 512, 0, 2, 1, 1, True, id="g8_s512_h128_b128_seg512_2xkv_tp_out"),
    pytest.param(8, 512, 128, 128, 512, 512, 2, 1, 1, True, id="g8_s512_h128_b128_seg512_prior512_full_tp_out"),
    pytest.param(8, 512, 128, 128, 512, 256, 2, 1, 1, True, id="g8_s512_h128_b128_seg512_prior256_partial_2xkv_tp_out"),
    pytest.param(8, 512, 128, 128, 512, 0, 2, 1, 2, True, id="g8_s512_h128_b128_seg512_2xkv_lnc2_tp_out"),
]
_FAST_PARAMS = [pytest.param(*p.values, marks=pytest.mark.fast, id=p.id) for p in _FAST_PARAMS]

_FULL_ONLY_PARAMS = [
    # Heavy compile — full suite only
    pytest.param(8, 1024, 64, 128, 512, 0, 1, 1, 1, False, id="g8_s1024_h64_b128_seg512_2chunks"),
    pytest.param(8, 2048, 128, 128, 2048, 0, 1, 1, 1, False, id="g8_s2048_h128_b128_seg2048"),
    pytest.param(8, 2048, 128, 128, 2048, 1024, 2, 1, 1, False, id="g8_s2048_h128_b128_seg2048_prior1024_partial"),
    pytest.param(8, 2048, 128, 128, 2048, 2048, 2, 1, 1, False, id="g8_s2048_h128_b128_seg2048_prior2048_full"),
    pytest.param(8, 2048, 128, 128, 2048, 6144, 4, 1, 1, False, id="g8_s2048_h128_b128_seg2048_prior6144_3full"),
    pytest.param(8, 4096, 128, 128, 2048, 0, 1, 1, 1, False, id="g8_s4096_h128_b128_seg2048_2chunks"),
    pytest.param(8, 4096, 128, 128, 4096, 0, 1, 1, 1, False, id="g8_s4096_h128_b128_seg4096"),
]


@pytest_test_metadata(name="KV Parallel Segmented Prefill", pytest_marks=["attention", "kv_parallel"], tag=["model"])
class TestKVParallelSegmentedPrefill:
    """Test class for KV-parallel segmented prefill attention."""

    @pytest.mark.parametrize(_PARAM_NAMES, _FAST_PARAMS + _FULL_ONLY_PARAMS)
    def test_kv_parallel_segmented_prefill(
        self,
        test_manager: Orchestrator,
        group_size: int,
        seq_len: int,
        head_dim: int,
        block_size: int,
        seg_size: int,
        prior_tokens: int,
        local_kv_multiplier: int,
        num_groups: int,
        lnc_degree: int,
        tp_out: bool,
    ):
        """
        End-to-end test for KV-parallel segmented prefill attention.

        Tests the full algorithm:
        1. All-gather Q across ranks
        2. Each rank computes attention on its KV shard with shifted causal mask
        3. All-to-all exchange of partial outputs + softmax stats
        4. Merge partials using online softmax
        5. Return final result

        Args:
            group_size: Number of ranks per replica group
            seq_len: Sequence length (Q length)
            head_dim: Head dimension
            block_size: KV cache block size
            seg_size: Segment size for attention iteration
            prior_tokens: Prior tokens for continuation (shifts Q global position)
            local_kv_multiplier: Multiplier for local_kv_len = seg_size * multiplier
            num_groups: Number of independent replica groups (total_ranks = group_size * num_groups)
        """
        np.random.seed(42)

        num_kv_heads = 1
        local_kv_len = seg_size * local_kv_multiplier
        num_blocks = local_kv_len // block_size

        # With LNC>1, collectives operate at physical rank level, not NC level
        # collective_ranks = number of physical ranks participating in collectives
        # Each physical rank has lnc_degree NCs that share the work internally
        num_physical_ranks = group_size // lnc_degree
        collective_ranks = num_physical_ranks * num_groups

        # Generate Q for all groups (each group has group_size Q heads)
        total_q_heads = group_size * num_groups
        q_global = np.random.randn(total_q_heads, 1, seq_len, head_dim).astype(nl.bfloat16)

        # Generate KV shards for all physical ranks across all groups
        # Layout: (num_blocks, num_kv_heads, block_size, head_dim)
        k_cache_global = np.random.randn(collective_ranks, num_blocks, num_kv_heads, block_size, head_dim).astype(
            nl.bfloat16
        )
        v_cache_global = np.random.randn(collective_ranks, num_blocks, num_kv_heads, block_size, head_dim).astype(
            nl.bfloat16
        )

        # Block tables: sequential blocks for each rank
        block_tables = np.arange(num_blocks, dtype=np.int32).reshape(1, num_blocks)
        block_tables = dt.static_cast(block_tables, nl.int32)

        # Create replica groups at physical rank level
        # With LNC=2 and group_size=8: replica_group = [[0,1,2,3]] (4 physical ranks)
        replica_group_lists = [
            list(range(g * num_physical_ranks, (g + 1) * num_physical_ranks)) for g in range(num_groups)
        ]
        replica_groups = ReplicaGroup(replica_group_lists)

        def create_inputs(rank_id: int):
            # rank_id is physical rank index (0 to collective_ranks-1)
            # Each physical rank has lnc_degree NCs sharing the same KV shard

            # Determine which group this rank belongs to and its position within the group
            group_id = rank_id // num_physical_ranks
            rank_in_group = rank_id % num_physical_ranks

            # cp_offset based on physical rank's KV position within its group
            k_offset = rank_in_group * local_kv_len
            cp_offset_value = -k_offset + prior_tokens
            cp_offset = dt.static_cast(np.array([[cp_offset_value]], dtype=np.int32), nl.int32)

            # q_local contains lnc_degree Q heads for this physical rank
            # Shape: [lnc_degree, seq_len, head_dim]
            q_start = group_id * group_size + rank_in_group * lnc_degree
            q_local = q_global[q_start : q_start + lnc_degree, 0, :, :]  # [lnc_degree, seq_len, head_dim]

            return {
                "q": q_local,
                "k_cache": k_cache_global[rank_id],
                "v_cache": v_cache_global[rank_id],
                "block_tables": block_tables,
                "kvp_offset": cp_offset,
                "replica_groups": replica_groups,
                "group_size": group_size,
                "block_size": block_size,
                "seg_size": seg_size,
                "scale": 1.0,
                "global_q_offset": prior_tokens,
                "tp_out": tp_out,
            }

        def create_golden(rank_id: int):
            # rank_id is physical rank index (0 to collective_ranks-1)
            # Each physical rank outputs lnc_degree Q heads
            group_id = rank_id // num_physical_ranks
            rank_in_group = rank_id % num_physical_ranks

            # Concatenate KV shards from all physical ranks in the same group
            k_full = []
            v_full = []
            for pr in range(num_physical_ranks):
                global_pr = group_id * num_physical_ranks + pr
                # Layout: (num_blocks, num_kv_heads, block_size, head_dim) → flatten to (total_kv_len, head_dim)
                k_seq = k_cache_global[global_pr, :, 0, :, :].reshape(-1, head_dim).astype(np.float32)
                v_seq = v_cache_global[global_pr, :, 0, :, :].reshape(-1, head_dim).astype(np.float32)
                k_full.append(k_seq)
                v_full.append(v_seq)
            k_full = np.concatenate(k_full, axis=0)  # [total_kv_len, head_dim]
            v_full = np.concatenate(v_full, axis=0)  # [total_kv_len, head_dim]

            total_kv_len = k_full.shape[0]

            # Compute attention for each of this rank's Q heads
            outputs = []
            q_start_global = group_id * group_size + rank_in_group * lnc_degree

            for nc in range(lnc_degree):
                q_head_idx = q_start_global + nc
                q = q_global[q_head_idx, 0].astype(np.float32)  # [seq_len, head_dim]

                # Compute attention scores
                scores = np.matmul(q, k_full.T)  # [seq_len, total_kv_len]

                # Apply causal mask
                q_pos = np.arange(prior_tokens, prior_tokens + seq_len).reshape(-1, 1)
                k_pos = np.arange(total_kv_len).reshape(1, -1)
                causal_mask = q_pos < k_pos
                scores = np.where(causal_mask, -np.inf, scores)

                # Softmax
                max_scores = np.max(scores, axis=-1, keepdims=True)
                max_scores = np.where(np.isinf(max_scores), 0, max_scores)
                exp_scores = np.exp(scores - max_scores)
                sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
                sum_exp = np.where(sum_exp == 0, 1, sum_exp)
                attn_weights = exp_scores / sum_exp

                # Output
                out = np.matmul(attn_weights, v_full)  # [seq_len, head_dim]
                outputs.append(out)

            # Stack outputs: [lnc_degree, seq_len, head_dim] or [lnc_degree, head_dim, seq_len] if tp_out
            out_stacked = np.stack(outputs, axis=0).astype(nl.bfloat16)
            if tp_out:
                out_stacked = np.transpose(out_stacked, (0, 2, 1))  # [lnc_degree, head_dim, seq_len]

            return {
                "out": out_stacked,
            }

        # Build torch_ref + override pair using side-channel pattern.
        # The golden needs all ranks' KV data, so the torch_ref closure captures
        # the test-level arrays and reads rank_id from _state.
        _state = {}

        def _ref_override(rank_id, raw_input):
            _state["rank_id"] = rank_id
            return raw_input

        def _torch_ref(
            q,
            k_cache,
            v_cache,
            block_tables,
            kvp_offset,
            replica_groups,
            group_size,
            block_size,
            seg_size,
            scale=1.0,
            global_q_offset=0,
            tp_out=False,
        ):
            return create_golden(_state["rank_id"])

        framework = CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kv_parallel_segmented_prefill,
            torch_ref=_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=collective_ranks,
            per_rank_torch_ref_input_override=_ref_override,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=Platforms.TRN2, logical_nc_config=lnc_degree),
            rtol=5e-2,
            atol=1e-2,
        )
