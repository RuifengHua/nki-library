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

from nkilib_src.nkilib.core.attention.attention_kv_parallel_segmented_cte import (
    attention_kv_parallel_segmented_cte,
)
from nkilib_src.nkilib.experimental.collectives.distributed_adapter import get_rank
from test.utils.common_dataclasses import (
    CompilerArgs,
    Platforms,
)
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_collective_framework import CollectiveUnitTestFramework

_CONTIGUOUS_PARAM_NAMES = (
    "group_size,seqlen,head_dim,block_size,seg_size,prior_tokens,local_kv_multiplier,num_groups,lnc_degree,tp_out"
)

_CONTIGUOUS_FAST_PARAMS = [
    # Contiguous KV distribution (backward compatibility, not the production path)
    # LNC2 (default)
    pytest.param(8, 512, 128, 128, 512, 0, 1, 1, 2, False, id="g8_s512_h128_b128_seg512_lnc2"),
    pytest.param(8, 512, 128, 128, 512, 512, 2, 1, 2, False, id="g8_s512_h128_b128_seg512_prior512_full_lnc2"),
    pytest.param(8, 512, 128, 128, 512, 256, 1, 1, 2, False, id="g8_s512_h128_b128_seg512_prior256_partial_lnc2"),
    pytest.param(8, 1024, 128, 128, 512, 0, 1, 1, 2, False, id="g8_s1024_h128_b128_seg512_2chunks_lnc2"),
    pytest.param(8, 512, 128, 64, 512, 0, 1, 1, 2, False, id="g8_s512_h128_b64_seg512_lnc2"),
    pytest.param(8, 512, 128, 32, 512, 0, 1, 1, 2, False, id="g8_s512_h128_b32_seg512_lnc2"),
    pytest.param(8, 512, 64, 128, 512, 0, 1, 1, 2, False, id="g8_s512_h64_b128_seg512_lnc2"),
    pytest.param(8, 512, 128, 128, 512, 0, 2, 1, 2, True, id="g8_s512_h128_b128_seg512_2xkv_tp_out_lnc2"),
    # LNC1 (baseline only)
    pytest.param(8, 512, 128, 128, 512, 0, 1, 1, 1, False, id="g8_s512_h128_b128_seg512"),
]
_CONTIGUOUS_FAST_PARAMS = [pytest.param(*p.values, marks=pytest.mark.fast, id=p.id) for p in _CONTIGUOUS_FAST_PARAMS]

_CONTIGUOUS_FULL_ONLY_PARAMS = [
    # Heavy compile — full suite only
    pytest.param(8, 2048, 128, 128, 2048, 2048, 2, 1, 2, False, id="g8_s2048_h128_b128_seg2048_prior2048_lnc2"),
]

_INTERLEAVED_PARAM_NAMES = "group_size,seqlen,head_dim,block_size,seg_size,prior_tokens,num_global_blocks,num_groups,lnc_degree,tp_out,sliding_window"

_INTERLEAVED_FAST_PARAMS = [
    # === LNC2 (production target) ===
    # --- block_size=128 ---
    pytest.param(8, 512, 128, 128, 512, 0, 32, 1, 2, False, 0, id="ilv_lnc2_s512_h128_b128_seg512"),
    pytest.param(8, 512, 128, 128, 512, 512, 32, 1, 2, False, 0, id="ilv_lnc2_s512_h128_b128_seg512_prior512"),
    pytest.param(8, 512, 128, 128, 512, 1024, 48, 1, 2, False, 0, id="ilv_lnc2_s512_h128_b128_seg512_prior1024"),
    pytest.param(8, 1024, 128, 128, 512, 0, 32, 1, 2, False, 0, id="ilv_lnc2_s1024_h128_b128_seg512_2chunks"),
    # --- block_size=64 ---
    pytest.param(8, 512, 128, 64, 512, 0, 64, 1, 2, False, 0, id="ilv_lnc2_s512_h128_b64_seg512"),
    pytest.param(8, 512, 128, 64, 512, 512, 64, 1, 2, False, 0, id="ilv_lnc2_s512_h128_b64_seg512_prior512"),
    # --- block_size=32 ---
    pytest.param(8, 512, 128, 32, 512, 0, 128, 1, 2, False, 0, id="ilv_lnc2_s512_h128_b32_seg512"),
    pytest.param(8, 512, 128, 32, 512, 512, 128, 1, 2, False, 0, id="ilv_lnc2_s512_h128_b32_seg512_prior512"),
    # --- head_dim=64 ---
    pytest.param(8, 512, 64, 128, 512, 0, 32, 1, 2, False, 0, id="ilv_lnc2_s512_h64_b128_seg512"),
    pytest.param(8, 512, 64, 128, 512, 512, 32, 1, 2, False, 0, id="ilv_lnc2_s512_h64_b128_seg512_prior512"),
    pytest.param(8, 512, 64, 64, 512, 0, 64, 1, 2, False, 0, id="ilv_lnc2_s512_h64_b64_seg512"),
    pytest.param(8, 512, 64, 64, 512, 512, 64, 1, 2, False, 0, id="ilv_lnc2_s512_h64_b64_seg512_prior512"),
    # --- long prior (multiple full prior segments) ---
    pytest.param(8, 512, 128, 128, 512, 2048, 80, 1, 2, False, 0, id="ilv_lnc2_s512_h128_b128_seg512_prior2048"),
    pytest.param(8, 512, 128, 64, 512, 2048, 160, 1, 2, False, 0, id="ilv_lnc2_s512_h128_b64_seg512_prior2048"),
    # === LNC1 (basic coverage) ===
    pytest.param(8, 512, 128, 128, 512, 0, 32, 1, 1, False, 0, id="ilv_lnc1_s512_h128_b128_seg512"),
    pytest.param(8, 512, 128, 128, 512, 512, 32, 1, 1, False, 0, id="ilv_lnc1_s512_h128_b128_seg512_prior512"),
    pytest.param(8, 512, 128, 64, 512, 0, 64, 1, 1, False, 0, id="ilv_lnc1_s512_h128_b64_seg512"),
    # --- partial prior ---
    pytest.param(8, 512, 128, 64, 512, 256, 64, 1, 2, False, 0, id="ilv_lnc2_s512_h128_b64_seg512_prior256_partial"),
    pytest.param(8, 512, 128, 128, 512, 256, 32, 1, 2, False, 0, id="ilv_lnc2_s512_h128_b128_seg512_prior256_partial"),
    # --- multi-chunk + prior ---
    pytest.param(8, 1024, 128, 128, 512, 512, 48, 1, 2, False, 0, id="ilv_lnc2_s1024_h128_b128_seg512_2chunks_prior"),
    # --- block_size=16 ---
    pytest.param(8, 512, 128, 16, 512, 0, 256, 1, 2, False, 0, id="ilv_lnc2_s512_h128_b16_seg512"),
    pytest.param(8, 512, 128, 16, 512, 512, 256, 1, 2, False, 0, id="ilv_lnc2_s512_h128_b16_seg512_prior512"),
    # --- tp_out=True ---
    pytest.param(8, 512, 128, 128, 512, 0, 32, 1, 2, True, 0, id="ilv_lnc2_s512_h128_b128_seg512_tp_out"),
    pytest.param(8, 512, 128, 128, 512, 512, 32, 1, 2, True, 0, id="ilv_lnc2_s512_h128_b128_seg512_prior512_tp_out"),
    # --- multi-chunk + b64 ---
    pytest.param(8, 1024, 128, 64, 512, 0, 64, 1, 2, False, 0, id="ilv_lnc2_s1024_h128_b64_seg512_2chunks"),
    # --- h64 + b32 ---
    pytest.param(8, 512, 64, 32, 512, 0, 128, 1, 2, False, 0, id="ilv_lnc2_s512_h64_b32_seg512"),
    pytest.param(8, 512, 64, 32, 512, 512, 128, 1, 2, False, 0, id="ilv_lnc2_s512_h64_b32_seg512_prior512"),
    # --- h64 + multi-chunk ---
    pytest.param(8, 1024, 64, 128, 512, 0, 32, 1, 2, False, 0, id="ilv_lnc2_s1024_h64_b128_seg512_2chunks"),
    # === SWA (sliding window attention) + interleaved ===
    pytest.param(8, 512, 128, 128, 512, 512, 32, 1, 2, False, 256, id="ilv_lnc2_s512_h128_b128_seg512_prior512_sw256"),
    pytest.param(
        8, 512, 128, 128, 512, 1024, 48, 1, 2, False, 512, id="ilv_lnc2_s512_h128_b128_seg512_prior1024_sw512"
    ),
    pytest.param(8, 512, 128, 64, 512, 512, 64, 1, 2, False, 256, id="ilv_lnc2_s512_h128_b64_seg512_prior512_sw256"),
    pytest.param(8, 512, 64, 128, 512, 512, 32, 1, 2, False, 256, id="ilv_lnc2_s512_h64_b128_seg512_prior512_sw256"),
]
_INTERLEAVED_FAST_PARAMS = [pytest.param(*p.values, marks=pytest.mark.fast, id=p.id) for p in _INTERLEAVED_FAST_PARAMS]

_INTERLEAVED_FULL_ONLY_PARAMS = [
    # Heavy compile — full suite only
    pytest.param(8, 2048, 128, 128, 2048, 0, 64, 1, 2, False, 0, id="ilv_lnc2_s2048_h128_b128_seg2048"),
    pytest.param(8, 2048, 128, 128, 2048, 2048, 128, 1, 2, False, 0, id="ilv_lnc2_s2048_h128_b128_seg2048_prior2048"),
    pytest.param(8, 2048, 128, 64, 2048, 0, 128, 1, 2, False, 0, id="ilv_lnc2_s2048_h128_b64_seg2048"),
    pytest.param(8, 2048, 128, 64, 2048, 2048, 192, 1, 2, False, 0, id="ilv_lnc2_s2048_h128_b64_seg2048_prior2048"),
    pytest.param(8, 2048, 128, 32, 2048, 0, 256, 1, 2, False, 0, id="ilv_lnc2_s2048_h128_b32_seg2048"),
    pytest.param(8, 2048, 64, 128, 2048, 0, 64, 1, 2, False, 0, id="ilv_lnc2_s2048_h64_b128_seg2048"),
    pytest.param(8, 2048, 64, 128, 2048, 2048, 128, 1, 2, False, 0, id="ilv_lnc2_s2048_h64_b128_seg2048_prior2048"),
    pytest.param(8, 2048, 128, 128, 2048, 0, 128, 1, 1, False, 0, id="ilv_lnc1_s2048_h128_b128_seg2048"),
    pytest.param(8, 2048, 128, 32, 2048, 2048, 512, 1, 2, False, 0, id="ilv_lnc2_s2048_h128_b32_seg2048_prior2048"),
    pytest.param(
        8, 2048, 128, 128, 2048, 2048, 128, 1, 2, False, 512, id="ilv_lnc2_s2048_h128_b128_seg2048_prior2048_sw512"
    ),
    pytest.param(8, 4096, 128, 128, 4096, 0, 128, 1, 2, False, 0, id="ilv_lnc2_s4096_h128_b128_seg4096"),
    pytest.param(8, 4096, 128, 128, 4096, 4096, 256, 1, 2, False, 0, id="ilv_lnc2_s4096_h128_b128_seg4096_prior4096"),
]


@pytest_test_metadata(name="KV Parallel Segmented Prefill", pytest_marks=["attention", "kv_parallel"], tag=["model"])
@pytest.mark.skip_simulation
class TestKVParallelSegmentedPrefill:
    """Test class for KV-parallel segmented prefill attention."""

    @pytest.mark.parametrize(_CONTIGUOUS_PARAM_NAMES, _CONTIGUOUS_FAST_PARAMS + _CONTIGUOUS_FULL_ONLY_PARAMS)
    def test_kv_parallel_segmented_prefill(
        self,
        test_manager: Orchestrator,
        group_size: int,
        seqlen: int,
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
            seqlen: Sequence length (Q length)
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
        q_global = np.random.randn(total_q_heads, 1, seqlen, head_dim).astype(nl.bfloat16)

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
            list(range(group_idx * num_physical_ranks, (group_idx + 1) * num_physical_ranks))
            for group_idx in range(num_groups)
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
            # Shape: [lnc_degree, seqlen, head_dim]
            q_start = group_id * group_size + rank_in_group * lnc_degree
            q_local = q_global[q_start : q_start + lnc_degree, 0, :, :]  # [lnc_degree, seqlen, head_dim]

            return {
                "q": q_local,
                "k_cache": k_cache_global[rank_id],
                "v_cache": v_cache_global[rank_id],
                "block_tables": block_tables,
                "kvp_q_offset": cp_offset,
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
                q = q_global[q_head_idx, 0].astype(np.float32)  # [seqlen, head_dim]

                # Compute attention scores
                scores = np.matmul(q, k_full.T)  # [seqlen, total_kv_len]

                # Apply causal mask
                q_pos = np.arange(prior_tokens, prior_tokens + seqlen).reshape(-1, 1)
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
                out = np.matmul(attn_weights, v_full)  # [seqlen, head_dim]
                outputs.append(out)

            # Stack outputs: [lnc_degree, seqlen, head_dim] or [lnc_degree, head_dim, seqlen] if tp_out
            out_stacked = np.stack(outputs, axis=0).astype(nl.bfloat16)
            if tp_out:
                out_stacked = np.transpose(out_stacked, (0, 2, 1))  # [lnc_degree, head_dim, seqlen]

            return {
                "out": out_stacked,
            }

        def _torch_ref(
            q,
            k_cache,
            v_cache,
            block_tables,
            kvp_q_offset,
            replica_groups,
            group_size,
            block_size,
            seg_size,
            scale=1.0,
            global_q_offset=0,
            tp_out=False,
            sliding_window=0,
            kvp_rank_id=None,
            kvp_group_size=0,
        ):
            return create_golden(get_rank())

        framework = CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=attention_kv_parallel_segmented_cte,
            torch_ref=_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=collective_ranks,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=Platforms.TRN2, logical_nc_config=lnc_degree),
            output_keys=["out"],
            rtol=5e-2,
            atol=1e-2,
        )

    @pytest.mark.parametrize(_INTERLEAVED_PARAM_NAMES, _INTERLEAVED_FAST_PARAMS + _INTERLEAVED_FULL_ONLY_PARAMS)
    def test_kv_parallel_segmented_prefill_interleaved(
        self,
        test_manager: Orchestrator,
        group_size: int,
        seqlen: int,
        head_dim: int,
        block_size: int,
        seg_size: int,
        prior_tokens: int,
        num_global_blocks: int,
        num_groups: int,
        lnc_degree: int,
        tp_out: bool,
        sliding_window: int,
    ):
        """
        Test KV-parallel segmented prefill with interleaved (round-robin) block distribution.

        Each rank holds non-contiguous blocks distributed round-robin across the global KV timeline.
        The per-block masking in _attention_cte handles the non-contiguous causal mask.
        """
        np.random.seed(42)

        num_kv_heads = 1
        num_physical_ranks = group_size // lnc_degree
        collective_ranks = num_physical_ranks * num_groups

        # Global KV: num_global_blocks blocks, each block_size tokens
        global_kv_len = num_global_blocks * block_size
        k_global_flat = np.random.randn(num_global_blocks, num_kv_heads, block_size, head_dim).astype(nl.bfloat16)
        v_global_flat = np.random.randn(num_global_blocks, num_kv_heads, block_size, head_dim).astype(nl.bfloat16)

        # Round-robin assignment: global block b goes to rank (b % num_physical_ranks)
        # Each rank's local blocks, sorted by global position
        blocks_per_rank = num_global_blocks // num_physical_ranks

        # Generate Q for all groups
        total_q_heads = group_size * num_groups
        q_global = np.random.randn(total_q_heads, 1, seqlen, head_dim).astype(nl.bfloat16)

        # Replica groups
        replica_group_lists = [
            list(range(group_idx * num_physical_ranks, (group_idx + 1) * num_physical_ranks))
            for group_idx in range(num_groups)
        ]
        replica_groups = ReplicaGroup(replica_group_lists)

        def create_inputs(rank_id: int):
            group_id = rank_id // num_physical_ranks
            rank_in_group = rank_id % num_physical_ranks

            # Collect this rank's blocks (round-robin)
            local_global_block_ids = list(range(rank_in_group, num_global_blocks, num_physical_ranks))
            num_local_blocks = len(local_global_block_ids)

            # Build local KV cache from global blocks
            k_local = np.zeros((num_local_blocks, num_kv_heads, block_size, head_dim), dtype=nl.bfloat16)
            v_local = np.zeros((num_local_blocks, num_kv_heads, block_size, head_dim), dtype=nl.bfloat16)

            for local_idx, global_blk_id in enumerate(local_global_block_ids):
                k_local[local_idx] = k_global_flat[global_blk_id]
                v_local[local_idx] = v_global_flat[global_blk_id]

            block_tables = np.arange(num_local_blocks, dtype=np.int32).reshape(1, num_local_blocks)
            block_tables = dt.static_cast(block_tables, nl.int32)

            # kvp_offset = global_q_offset for interleaved KV
            cp_offset_value = prior_tokens
            cp_offset = dt.static_cast(np.array([[cp_offset_value]], dtype=np.int32), nl.int32)

            q_start = group_id * group_size + rank_in_group * lnc_degree
            q_local = q_global[q_start : q_start + lnc_degree, 0, :, :]

            return {
                "q": q_local,
                "k_cache": k_local,
                "v_cache": v_local,
                "block_tables": block_tables,
                "kvp_q_offset": cp_offset,
                "replica_groups": replica_groups,
                "group_size": group_size,
                "block_size": block_size,
                "seg_size": seg_size,
                "scale": 1.0,
                "global_q_offset": prior_tokens,
                "tp_out": tp_out,
                "sliding_window": sliding_window,
                "kvp_rank_id": dt.static_cast(np.array([[rank_in_group]], dtype=np.int32), nl.int32),
                "kvp_group_size": num_physical_ranks,
            }

        def create_golden(rank_id: int):
            group_id = rank_id // num_physical_ranks
            rank_in_group = rank_id % num_physical_ranks

            # Concatenate ALL ranks' KV in global order for the golden reference
            k_full = k_global_flat[:, 0, :, :].reshape(-1, head_dim).astype(np.float32)
            v_full = v_global_flat[:, 0, :, :].reshape(-1, head_dim).astype(np.float32)

            outputs = []
            q_start_global = group_id * group_size + rank_in_group * lnc_degree

            for nc in range(lnc_degree):
                q_head_idx = q_start_global + nc
                q = q_global[q_head_idx, 0].astype(np.float32)

                scores = np.matmul(q, k_full.T)

                # Causal mask: q_pos >= k_pos means visible
                q_pos = np.arange(prior_tokens, prior_tokens + seqlen).reshape(-1, 1)
                k_pos = np.arange(global_kv_len).reshape(1, -1)
                causal_mask = q_pos < k_pos
                scores = np.where(causal_mask, -np.inf, scores)

                # SWA mask: mask if k_pos < q_pos - (sliding_window - 1)
                if sliding_window > 0:
                    swa_mask = k_pos < q_pos - (sliding_window - 1)
                    scores = np.where(swa_mask, -np.inf, scores)

                max_scores = np.max(scores, axis=-1, keepdims=True)
                max_scores = np.where(np.isinf(max_scores), 0, max_scores)
                exp_scores = np.exp(scores - max_scores)
                sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
                sum_exp = np.where(sum_exp == 0, 1, sum_exp)
                attn_weights = exp_scores / sum_exp

                out = np.matmul(attn_weights, v_full)
                outputs.append(out)

            out_stacked = np.stack(outputs, axis=0).astype(nl.bfloat16)
            if tp_out:
                out_stacked = np.transpose(out_stacked, (0, 2, 1))
            return {"out": out_stacked}

        def _torch_ref(
            q,
            k_cache,
            v_cache,
            block_tables,
            kvp_q_offset,
            replica_groups,
            group_size,
            block_size,
            seg_size,
            scale=1.0,
            global_q_offset=0,
            tp_out=False,
            sliding_window=0,
            kvp_rank_id=None,
            kvp_group_size=0,
        ):
            return create_golden(get_rank())

        framework = CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=attention_kv_parallel_segmented_cte,
            torch_ref=_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=collective_ranks,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=Platforms.TRN2, logical_nc_config=lnc_degree),
            output_keys=["out"],
            rtol=5e-2,
            atol=1e-2,
        )
