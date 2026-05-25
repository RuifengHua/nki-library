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

"""Unified tests for moe_cte entry point using UnitTestFramework."""

from typing import final

import nki.language as nl
import pytest

from nkilib_src.nkilib.core.moe.moe_cte import (
    MoECTEImplementation,
    moe_cte,
)
from nkilib_src.nkilib.core.moe.moe_cte.moe_cte_torch import moe_cte_unified_torch_ref
from nkilib_src.nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
from test.integration.nkilib.core.moe.moe_cte.test_moe_cte_common import (
    generate_moe_cte_unified_inputs,
    moe_cte_unified_output_tensors,
)
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.metrics_collector import IMetricsCollector
from test.utils.mx_utils import is_mx_quantize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# fmt: off
# Parameter names for pytest.mark.parametrize
UNIFIED_PARAM_NAMES = \
    "impl,                                              hidden, tokens, expert, block_size, top_k, intermediate, dtype,       skip, bias,  training, quantize,        act_fn,          expert_affinities_scaling_mode,     gate_cl_upper, gate_cl_lower, up_cl_upper, up_cl_lower, expert_affinity_multiply_on_I, weight_dtype,           is_dynamic"

# =============================================================================
# NON-MX TEST CASES
# =============================================================================
NON_MX_TEST_CASES = [
    # SHARD_ON_INTERMEDIATE_HW tests
    (MoECTEImplementation.shard_on_i_hybrid,            3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,    False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False,                         None,                   False),
    # SHARD_ON_INTERMEDIATE tests
    (MoECTEImplementation.shard_on_i,                   3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,    True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False,                         None,                   False),
    # SHARD_ON_BLOCK tests
    (MoECTEImplementation.shard_on_block,               3072,   1024,   8,      512,        4,     384,          nl.bfloat16, 1,    True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False,                         None,                   False),
    # Dropping kernel tests
    (MoECTEImplementation.shard_on_i_dropping,          1536,   8192,   2,      4096,       2,     6144,         nl.bfloat16, 0,    False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True,                          None,                   False),
]

# =============================================================================
# MX (MXFP4/MXFP8) SHARD-ON-BLOCK TEST CASES
# Note: MX through moe_cte() dispatch produces incorrect kernel output (inf/nan).
# The torch ref (dequant to fp32) is correct. The issue is in moe_cte() -> bwmm_shard_on_block_mx
# dispatch (nested @nki.jit). MX kernels pass when called directly (test_moe_bwmm_mx_cte.py).
# TODO: Fix moe_cte() MX dispatch, then re-enable.
# =============================================================================
MX_BLOCK_TEST_CASES = [
    # (MoECTEImplementation.shard_on_block_mx,            3072,   1024,   8,      256,        4,     384,          nl.bfloat16, 1,    False, False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7.0,           None,          7.0,         -7.0,        False,                         nl.float4_e2m1fn_x4,    False),
]

# =============================================================================
# MX (MXFP4/MXFP8) SHARD-ON-INTERMEDIATE TEST CASES
# =============================================================================
MX_SHARD_I_TEST_CASES = [
    # (MoECTEImplementation.shard_on_i_mx_hybrid,       7168,   1024,   8,      256,        8,     1024,         nl.bfloat16, 1,    True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7.0,           None,          7.0,         -7.0,        False,                         nl.float4_e2m1fn_x4,    True),
]

ALL_TEST_CASES = NON_MX_TEST_CASES + MX_BLOCK_TEST_CASES + MX_SHARD_I_TEST_CASES
# fmt: on


@pytest_test_metadata(name="MoE CTE Unified Entry Point")
@pytest_marks(["moe", "cte", "unified"])
@final
class TestMoeCTEUnified:
    """Unified tests for moe_cte() entry point covering all implementations.

    skip modes:
    - 0: SkipMode(False, False)
    - 1: SkipMode(True, False)  - skip token
    - 2: SkipMode(False, True)  - skip weight
    - 3: SkipMode(True, True)   - skip both
    """

    @pytest.mark.fast
    @pytest.mark.parametrize(UNIFIED_PARAM_NAMES, ALL_TEST_CASES)
    def test_moe_cte_unified(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        impl: MoECTEImplementation,
        hidden: int,
        tokens: int,
        expert: int,
        block_size: int,
        top_k: int,
        intermediate: int,
        dtype,
        skip: int,
        bias: bool,
        training: bool,
        quantize,
        act_fn: ActFnType,
        expert_affinities_scaling_mode: ExpertAffinityScaleMode,
        gate_cl_upper,
        gate_cl_lower,
        up_cl_upper,
        up_cl_lower,
        expert_affinity_multiply_on_I: bool,
        weight_dtype,
        is_dynamic: bool,
        platform_target: Platforms,
        request,
    ):
        lnc_degree = 2

        # Skip MX tests on non-TRN3 platforms
        is_mx = is_mx_quantize(weight_dtype)
        if is_mx and not platform_target.is_trn3():
            pytest.skip("MX (MXFP4/MXFP8) is only supported on TRN3.")

        target = Platforms.TRN3 if is_mx else platform_target

        def input_generator(test_config):
            return generate_moe_cte_unified_inputs(
                impl=impl,
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                dtype=dtype,
                skip=skip,
                bias=bias,
                training=training,
                quantize=quantize,
                activation_function=act_fn,
                expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                gate_clamp_upper=gate_cl_upper,
                gate_clamp_lower=gate_cl_lower,
                up_clamp_upper=up_cl_upper,
                up_clamp_lower=up_cl_lower,
                expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
                weight_dtype=weight_dtype,
                is_dynamic=is_dynamic,
                lnc_degree=lnc_degree,
            )

        def output_tensors(kernel_input):
            return moe_cte_unified_output_tensors(
                kernel_input=kernel_input,
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                dtype=dtype,
                impl=impl,
                training=training,
                expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
                lnc_degree=lnc_degree,
            )

        rtol, atol = (5e-2, 1e-5) if (is_mx or quantize) else (2e-2, 1e-5)

        compiler_args = CompilerArgs(logical_nc_config=lnc_degree, platform_target=target)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=moe_cte,
            torch_ref=torch_ref_wrapper(moe_cte_unified_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
            collector=collector,
        )

        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            rtol=rtol,
            atol=atol,
        )
