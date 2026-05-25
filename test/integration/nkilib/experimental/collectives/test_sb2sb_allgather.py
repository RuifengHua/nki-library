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
"""Tests for SBUF-to-SBUF All-Gather kernels."""

import nki.language as nl
import numpy as np
import pytest
from nki.collectives import ReplicaGroup

from nkilib_src.nkilib.experimental.collectives.sb2sb_allgather import (
    allgather_sb2sb,
    allgather_sb2sb_tiled,
)
from test.integration.nkilib.experimental.collectives.test_collectives import make_golden_torch_ref
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import CollectiveUnitTestFramework

SB2SB_PARAM_NAMES = "m, k, dtype, tp_degree"
SB2SB_TEST_PARAMS = [
    # Basic tests
    (128, 512, nl.bfloat16, 8),
    (64, 1024, nl.bfloat16, 8),
    (128, 2048, nl.bfloat16, 8),
    (96, 512, nl.bfloat16, 16),
    # dtype variations
    (128, 512, np.float32, 8),
    (64, 1024, np.float16, 8),
    # Different TP degrees
    (128, 256, nl.bfloat16, 64),
    (128, 256, nl.bfloat16, 32),
    # Non-power-of-2 k
    (128, 384, nl.bfloat16, 16),
]

TILED_PARAM_NAMES = "m, k, dtype, tp_degree, lnc"
TILED_TEST_PARAMS = [
    # Single tile cases (m <= 128)
    (128, 512, nl.bfloat16, 4, 2),
    (64, 1024, nl.bfloat16, 4, 2),
    # Multi-tile cases (m > 128, m % 128 == 0)
    (256, 512, nl.bfloat16, 8, 1),
    (256, 512, nl.bfloat16, 4, 2),
    (512, 1024, nl.bfloat16, 8, 1),
    (512, 1024, nl.bfloat16, 8, 2),
    # dtype variations
    (256, 512, np.float32, 8, 1),
    (512, 1024, np.float16, 4, 2),
    (256, 512, nl.bfloat16, 8, 2),
]
_ABBREVS = {"tp_degree": "tp"}


def _run_sb2sb_allgather_test(test_manager, platform_target, kernel_entry, m, k, dtype, tp_degree, lnc, output_key):
    """Shared test logic for SBUF-to-SBUF all-gather kernels."""
    np.random.seed(42)
    # Each rank has different input data
    x_global = np.random.randn(tp_degree, m, k).astype(dtype)
    replica_groups = ReplicaGroup([list(range(tp_degree))])

    # Golden: concatenate all ranks along k dimension
    gathered = np.concatenate([x_global[r] for r in range(tp_degree)], axis=1)

    def create_inputs(rank_id: int):
        return {
            "inp": x_global[rank_id],
            "replica_groups": replica_groups,
            "tp_degree": tp_degree,
        }

    def create_golden(rank_id: int):
        return {output_key: gathered}

    torch_ref, ref_override = make_golden_torch_ref(kernel_entry, create_golden)
    CollectiveUnitTestFramework(
        test_manager=test_manager,
        kernel_entry=kernel_entry,
        torch_ref=torch_ref,
        per_rank_input_generator=create_inputs,
        collective_ranks=tp_degree,
        per_rank_torch_ref_input_override=ref_override,
    ).run_test(
        test_config=None,
        compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
        rtol=1e-3,
        atol=1e-3,
    )


class TestSb2sbAllgather:
    """Test class for SBUF-to-SBUF all-gather kernels."""

    @pytest.mark.fast
    @pytest_parametrize(SB2SB_PARAM_NAMES, SB2SB_TEST_PARAMS, abbrevs=_ABBREVS)
    def test_allgather_sb2sb(
        self, test_manager: Orchestrator, platform_target: Platforms, m: int, k: int, dtype: np.dtype, tp_degree: int
    ):
        """Test basic SBUF-to-SBUF all-gather kernel."""
        _run_sb2sb_allgather_test(
            test_manager, platform_target, allgather_sb2sb, m, k, dtype, tp_degree, lnc=1, output_key="out"
        )

    @pytest.mark.fast
    @pytest_parametrize(TILED_PARAM_NAMES, TILED_TEST_PARAMS, abbrevs=_ABBREVS)
    def test_allgather_sb2sb_tiled(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        m: int,
        k: int,
        dtype: np.dtype,
        tp_degree: int,
        lnc: int,
    ):
        """Test tiled SBUF-to-SBUF all-gather kernel with LNC support."""
        _run_sb2sb_allgather_test(
            test_manager, platform_target, allgather_sb2sb_tiled, m, k, dtype, tp_degree, lnc, output_key="result"
        )
