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

"""Integration tests for the blockwise MM backward kernel."""

from typing import final

import nki.language as nl
import numpy as np
import pytest

from nkilib_src.nkilib.experimental.moe.bwd.blockwise_mm_backward import blockwise_mm_bwd
from nkilib_src.nkilib.experimental.moe.bwd.moe_bwd_parameters import (
    ActFnType,
    AffinityOption,
    ClampLimits,
    DownWeightGradBlocking,
    GateUpOutputGradBlocking,
    GateUpWeightGradBlocking,
    HiddenGradBlocking,
    MOEBwdDroplessBlockingParams,
    ShardOption,
)
from test.integration.nkilib.experimental.moe.test_bwmm_bwd_common import (
    blockwise_mm_bwd_torch_ref,
    build_bwmm_bwd_inputs,
    map_skip_mode,
)
from test.utils.common_dataclasses import CompilerArgs, InferenceArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

bfloat16 = nl.bfloat16

# fmt: off
AFFINITY_H = AffinityOption.AFFINITY_ON_H
AFFINITY_I = AffinityOption.AFFINITY_ON_I
SHARD_FREE = ShardOption.SHARD_ON_FREE
SHARD_H = ShardOption.SHARD_ON_HIDDEN
DEFAULT_BP = MOEBwdDroplessBlockingParams(gate_up_output_grad=GateUpOutputGradBlocking(),
                                          down_weight_grad=DownWeightGradBlocking(),
                                          hidden_grad=HiddenGradBlocking(),
                                          gate_up_weight_grad=GateUpWeightGradBlocking())

PARAM_NAMES = \
    "hidden, tokens, expert, block_size, top_k, intermediate, dtype, skip, clamp_limits, bias_flag, activation_type, affinity_option, blocking_params, shard_option"
TEST_PARAMS = [
# H,    T,    E,   B,   TOPK, I_TP, dtype,    skip, clamp_limits,                        bias,  activation_type,  affinity, blocking_params,                                                                                                                                                                                          shard_option
[5120,  8192, 16,  512, 1,    256,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE],
[5120,  8192, 16,  256, 4,    1024, bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE],
[5120,  8192, 128, 256, 1,    128,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE],

[6144,  4096, 16,  512, 4,    1024, bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE],
[6144,  4096, 16,  512, 4,    128,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE],
[6144,  4096, 1,   512, 1,    128,  bfloat16, 0,    ClampLimits(7, -7, 7, -7),            False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE],

[2880,  4096, 2,   512, 2,    2880, bfloat16, 0,    ClampLimits(7, -7, 7, -7),            True,  ActFnType.Swish,  AFFINITY_H, DEFAULT_BP, SHARD_FREE],
[2880,  4096, 2,   512, 2,    2880, bfloat16, 1,    ClampLimits(7, -7, 7, -7),            True,  ActFnType.Swish,  AFFINITY_H, DEFAULT_BP, SHARD_FREE],
[2880,  4096, 2,   256, 2,    2880, bfloat16, 1,    ClampLimits(7, -7, 7, -7),            True,  ActFnType.Swish,  AFFINITY_H, DEFAULT_BP, SHARD_FREE],
[4096,  4096, 2,   512, 2,    384,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE],
[4096,  4096, 4,   512, 2,    384,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE],
[4096,  4096, 4,   128, 2,    384,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE],

[4096,  4096, 4,   128, 2,    384,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE],
[4096,  4096, 4,   256, 2,    384,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE],

[4096,  4096, 4,   128, 2,    1536,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE],
[4096,  4096, 4,   256, 2,    1536,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE],

[2880,  4096, 2,   128, 2,    720, bfloat16, 0,    ClampLimits(7, -7, 7, -7),            True,  ActFnType.Swish,  AFFINITY_H, DEFAULT_BP, SHARD_FREE],

[5120,  4096, 4,   128, 1,    2048, bfloat16, 0,    ClampLimits(None, None, None, None),   False,  ActFnType.SiLU,  AFFINITY_H, DEFAULT_BP, SHARD_FREE],
[5120,  4096, 4,   256, 1,    2048, bfloat16, 0,    ClampLimits(None, None, None, None),   False,  ActFnType.SiLU,  AFFINITY_H, DEFAULT_BP, SHARD_FREE],

# Affinity I test cases
[4096,  4096, 4,   128, 2,    384,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, DEFAULT_BP, SHARD_FREE],
[4096,  4096, 4,   256, 2,    384,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, DEFAULT_BP, SHARD_FREE],

[4096,  4096, 4,   128, 2,    1536,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, DEFAULT_BP, SHARD_FREE],
[4096,  4096, 4,   256, 2,    1536,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, DEFAULT_BP, SHARD_FREE],

[5120,  4096, 4,   128, 1,    2048, bfloat16, 0,    ClampLimits(None, None, None, None),   False,  ActFnType.SiLU,  AFFINITY_I, DEFAULT_BP, SHARD_FREE],
[5120,  4096, 4,   256, 1,    2048, bfloat16, 0,    ClampLimits(None, None, None, None),   False,  ActFnType.SiLU,  AFFINITY_I, DEFAULT_BP, SHARD_FREE],

[2880,  4096, 2,  128, 2,    2880, bfloat16, 0,    ClampLimits(7, -7, 7, -7),  True, ActFnType.Swish,   AFFINITY_I, DEFAULT_BP, SHARD_FREE],
[2048,  4096, 2, 128, 2,    768,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, DEFAULT_BP, SHARD_FREE],
[2048,  4096, 2, 128, 2,    192,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, DEFAULT_BP, SHARD_FREE],
[5120,  4096, 2,  128, 2,    8192, bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, DEFAULT_BP, SHARD_FREE],
[5120,  4096, 2,  128, 2,    2048, bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, DEFAULT_BP, SHARD_FREE],
[2048,  4096, 2,  128, 2,    1408, bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, DEFAULT_BP, SHARD_FREE],
[2048,  4096, 2,  128, 2,    352,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, DEFAULT_BP, SHARD_FREE],

[2048,  512, 2, 512, 2,    256,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, MOEBwdDroplessBlockingParams(gate_up_output_grad=GateUpOutputGradBlocking(block_h=16, block_b=4, block_i=2),
                                                                                                                                                           down_weight_grad=DownWeightGradBlocking(block_h=16, block_b=4, block_i=2),
                                                                                                                                                           hidden_grad=HiddenGradBlocking(block_h=16, block_b=4, block_i=2),
                                                                                                                                                   gate_up_weight_grad=GateUpWeightGradBlocking(block_h=16, block_b=4, block_i=2)), SHARD_FREE],

# Shard H Test
[2048,  512,  2,  512, 2,    352,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, MOEBwdDroplessBlockingParams(gate_up_output_grad=GateUpOutputGradBlocking(block_h=16, block_b=4, block_i=3),
                                                                                                                                                           down_weight_grad=DownWeightGradBlocking(block_h=16, block_b=4, block_i=3),
                                                                                                                                                           hidden_grad=HiddenGradBlocking(block_h=16, block_b=4, block_i=3),
                                                                                                                                                   gate_up_weight_grad=GateUpWeightGradBlocking(block_h=16, block_b=4, block_i=3)), SHARD_H],

[2048,  16384, 64, 512, 6,    352,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, MOEBwdDroplessBlockingParams(gate_up_output_grad=GateUpOutputGradBlocking(block_h=16, block_b=4, block_i=3),
                                                                                                                                                           down_weight_grad=DownWeightGradBlocking(block_h=16, block_b=4, block_i=3),
                                                                                                                                                           hidden_grad=HiddenGradBlocking(block_h=16, block_b=4, block_i=3),
                                                                                                                                                   gate_up_weight_grad=GateUpWeightGradBlocking(block_h=16, block_b=4, block_i=3)), SHARD_H],

[3072,  8192, 32,   1024, 6,    720, bfloat16, 0,    ClampLimits(7, -7, 7, -7),            True,  ActFnType.Swish,  AFFINITY_I, MOEBwdDroplessBlockingParams(gate_up_output_grad=GateUpOutputGradBlocking(block_h=8, block_b=4, block_i=2),
                                                                                                                                                           down_weight_grad=DownWeightGradBlocking(block_h=16, block_b=8, block_i=6),
                                                                                                                                                           hidden_grad=HiddenGradBlocking(block_h=2, block_b=8, block_i=6),
                                                                                                                                                   gate_up_weight_grad=GateUpWeightGradBlocking(block_h=16, block_b=8, block_i=6)), SHARD_H],

]
# fmt: on

# (hidden, tokens, expert, block_size, top_k, intermediate) keys for full-only tests (excluded from fast suite)
_FULL_ONLY_KEYS = {
    (5120, 8192, 16, 256, 4, 1024),
    (5120, 8192, 128, 256, 1, 128),
    (6144, 4096, 16, 512, 4, 1024),
    (2880, 4096, 2, 128, 2, 2880),
    (5120, 4096, 2, 128, 2, 8192),
    (4096, 4096, 4, 128, 2, 1536),
    (4096, 4096, 4, 256, 2, 1536),
    (4096, 4096, 4, 128, 2, 384),
    (2880, 4096, 2, 256, 2, 2880),
    (2880, 4096, 2, 512, 2, 2880),
    (2880, 4096, 2, 128, 2, 720),
    (6144, 4096, 16, 512, 4, 128),
    (5120, 4096, 4, 256, 1, 2048),
    (2048, 4096, 2, 128, 2, 1408),
    (5120, 4096, 2, 128, 2, 2048),
    (5120, 4096, 4, 128, 1, 2048),
    (5120, 8192, 16, 512, 1, 256),
    (4096, 4096, 4, 256, 2, 384),
    (2048, 4096, 2, 128, 2, 768),
    (2048, 16384, 64, 512, 6, 256),
    (6144, 4096, 1, 512, 1, 128),
    (4096, 4096, 2, 512, 2, 384),
    (4096, 4096, 4, 512, 2, 384),
    (2048, 4096, 2, 128, 2, 192),
    (2048, 4096, 2, 128, 2, 352),
    (2048, 16384, 64, 512, 6, 352),
    (2048, 512, 2, 512, 2, 352),
    (3072, 8192, 32, 1024, 6, 720),
}

ALL_PARAMS = [
    pytest.param(*c, marks=pytest.mark.fast) if tuple(c[:6]) not in _FULL_ONLY_KEYS else c for c in TEST_PARAMS
]

_ABBREVS = {
    "hidden": "hid",
    "tokens": "tok",
    "expert": "exp",
    "block_size": "bs",
    "top_k": "k",
    "intermediate": "int",
    "dtype": "dt",
    "skip": "sk",
    "clamp_limits": "cl",
    "bias_flag": "bi",
    "activation_type": "act",
    "affinity_option": "aff",
    "blocking_params": "bp",
    "shard_option": "sh",
}


@pytest_test_metadata(name="MoE Blockwise MatMul BWD Dropless LNC2")
@pytest_marks(["moe", "blockwise_mm_bwd", "lnc2"])
@final
class TestMoeBlockwiseMatMulBwdShardHDroplessLnc2:
    """Tests for LNC2 blockwise matmul backward pass."""

    @pytest_parametrize(PARAM_NAMES, ALL_PARAMS, abbrevs=_ABBREVS)
    def test_moe_blockwise_mm_bwd_dropless_lnc2(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        hidden: int,
        tokens: int,
        expert: int,
        block_size: int,
        top_k: int,
        intermediate: int,
        dtype,
        skip: int,
        clamp_limits: ClampLimits,
        bias_flag: bool,
        activation_type: ActFnType,
        affinity_option: AffinityOption,
        blocking_params,
        shard_option: ShardOption,
    ):
        dma_skip = map_skip_mode(skip)

        def input_generator(test_config):
            inputs, _, _ = build_bwmm_bwd_inputs(
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                dtype=dtype,
                dma_skip=dma_skip,
                bias_flag=bias_flag,
                clamp_limits=clamp_limits,
                activation_type=activation_type,
                affinity_option=affinity_option,
                blocking_params=blocking_params,
                shard_option=shard_option,
            )
            return inputs

        def output_tensors(kernel_input):
            T_out = tokens if dma_skip.skip_token else tokens + 1
            result = {
                "hidden_states_grad": np.zeros((T_out, hidden), dtype=dtype),
                "expert_affinities_masked_grad": np.zeros((T_out * expert, 1), dtype=dtype),
                "gate_up_proj_weight_grad": np.zeros((expert, hidden, 2, intermediate), dtype=dtype),
                "down_proj_weight_grad": np.zeros((expert, intermediate, hidden), dtype=dtype),
            }
            if bias_flag:
                result["gate_and_up_proj_bias_grad"] = np.zeros((expert, 2, intermediate), dtype=dtype)
                result["down_proj_bias_grad"] = np.zeros((expert, hidden), dtype=dtype)
            return result

        lnc_count = 2
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=blockwise_mm_bwd,
            torch_ref=torch_ref_wrapper(blockwise_mm_bwd_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=lnc_count,
                enable_birsim=False,
                platform_target=platform_target,
                dump_after_lowering=False,
            ),
            inference_args=InferenceArgs(),
            rtol=2e-2,
            atol=1e-5,
        )
