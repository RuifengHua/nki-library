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

import math

try:
    from test.integration.nkilib.core.mlp.test_mlp_cte_model_config import (
        mlp_cte_model_configs,
    )
except ImportError:
    mlp_cte_model_configs = []

from typing import Any, final

import nki.language as nl
import pytest

from nkilib_src.nkilib.core.mlp.mlp_parameters import TKG_BS_SEQLEN_THRESHOLD
from nkilib_src.nkilib.core.utils.common_types import ActFnType, NormType, QuantizationType
from test.integration.nkilib.core.mlp.test_mlp_common import (
    _run_mlp_test,
    build_fused_norm_mlp,
    dedup_test_vectors,
    gaussian_tensor_generator,
    mlp_output_tensor_descriptor,
    modify_for_row_quant,
    modify_fp8_static_scale,
)
from test.utils.common_dataclasses import (
    CompilerArgs,
    ModelTestType,
    Platforms,
    prepare_model_parametrize,
)
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator

# ----------------------------------------------------
# Configuration-based testing to avoid combinatorial explosion of parameters
# ----------------------------------------------------

BASIC_MLP_CONFIG = {
    "norm_type": NormType.NO_NORM,
    "fused_add": False,
    "store_add": False,
    "skip_gate": False,
    "act_fn_type": ActFnType.SiLU,
    "gate_bias": False,
    "up_bias": False,
    "down_bias": False,
    "norm_bias": False,
}

FULL_FEATURES_CONFIG = {
    "norm_type": NormType.RMS_NORM,
    "fused_add": True,
    "store_add": True,
    "skip_gate": False,
    "act_fn_type": ActFnType.SiLU,
    "gate_bias": True,
    "up_bias": True,
    "down_bias": True,
    "norm_bias": False,
}

LAYER_NORM_CONFIG = {
    "norm_type": NormType.LAYER_NORM,
    "fused_add": False,
    "store_add": False,
    "skip_gate": False,
    "act_fn_type": ActFnType.SiLU,
    "gate_bias": False,
    "up_bias": False,
    "down_bias": False,
    "norm_bias": True,
}

# fmt: off
# Parameters: vnc_degree, batch, seqlen, hidden, intermediate, tpbSgCyclesSum, rtol, norm_type, quantization_type,
#             fused_add, store_add, skip_gate, act_fn_type, gate_bias, up_bias, down_bias, norm_bias
MLP_CTE_UNIT_TEST_CASES_GATE_BIAS_FALSE = [
    [2, 1, 128, 8192, 896, 238002211, 2e-2, NormType.LAYER_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 512, 1024, 448, 80453957, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 512, 1024, 448, 82809370, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 1024, 1024, 448, 92239439, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 1024, 1024, 448, 88019279, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 1024, 1024, 448, 92000000, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 1024, 1024, 448, 90017692, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 768, 1024, 896, 140496530, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 768, 1024, 896, 107670831, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 8192, 8192, 448, 4296859369, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 8192, 8192, 448, 2520729395, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 1024, 16384, 832, 756458068, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 1024, 16384, 832, 895261601, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 1024, 16384, 416, 882105371, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 1024, 16384, 416, 1176268151, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 1024, 8192, 448, 278000000, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 8192, 8192, 448, 4780395634, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 8192, 8192, 448, 3784829419, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 1024, 16384, 416, 1000000000, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 1024, 16384, 416, 1246253552, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 8192, 8192, 448, 2737231223, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 8192, 8192, 448, 2127094342, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 1024, 16384, 416, 595047246, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 1024, 16384, 416, 755884001, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [2, 2, 8192, 8192, 448, 5298071805, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [2, 4, 8192, 8192, 448, 8376700327, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [2, 2, 1024, 16384, 416, 1111000000, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [2, 4, 1024, 16384, 416, 2557272337, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 128, 8192, 896, 210306921, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 128, 8192, 896, 258002211, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 128, 8192, 896, 210306921, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 128, 8192, 896, 258002211, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 512, 1024, 448, 80453957, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 512, 1024, 448, 82809370, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 128, 8192, 896, 210306921, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 128, 8192, 896, 210306921, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 2048, 8448, 1408, 1437844003, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 2048, 8448, 1408, 1603737495, 2e-2, NormType.LAYER_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 4096, 8192, 448, 941321030, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 4096, 8192, 448, 1176746495, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 4096, 8192, 448, 1297751306, 2e-2, NormType.LAYER_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 128, 7168, 364, 1.52e8, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 128, 7168, 364, 1.21e8, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 256, 7168, 364, 1.59e8, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 256, 7168, 364, 1.41e8, 2e-2, NormType.RMS_NORM_SKIP_GAMMA, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 256, 7168, 1536, 281286893, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 256, 7168, 1536, 258002211, 2e-2, NormType.RMS_NORM_SKIP_GAMMA, QuantizationType.NONE, True, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 256, 7168, 1536, 258002211, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [1, 1, 578, 1408, 352, 66365979, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, True, ActFnType.SiLU, False, False, False, False],
    [1, 1, 578, 1408, 352, 66340813, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU, False, False, False, False],
    [2, 1, 578, 1408, 352, 65561487, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, True, ActFnType.SiLU, False, False, False, False],
    [2, 1, 578, 1408, 352, 66839480, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU, False, False, False, False],
    [1, 1, 578, 1408, 352, 71063222, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, True, ActFnType.SiLU, False, True, True, False],
    [1, 1, 578, 1408, 352, 71242055, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU, False, True, True, False],
    [2, 1, 578, 1408, 352, 69290891, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, True, ActFnType.SiLU, False, True, True, False],
    [2, 1, 578, 1408, 352, 71074390, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU, False, True, True, False],
    [1, 1, 578, 1408, 352, 81643622, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, True, True, False],
    [2, 1, 578, 1408, 352, 81231123, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, True, True, False],
    [1, 1, 578, 1408, 352, 84836617, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.GELU, False, False, False, False],
    [2, 1, 578, 1408, 352, 81332790, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.GELU, False, False, False, False],
    [1, 1, 578, 1408, 352, 93398179, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, True, ActFnType.SiLU, False, True, True, False],
    [2, 1, 578, 1408, 352, 83390954, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, True, ActFnType.SiLU, False, True, True, False],
    [1, 1, 578, 1408, 352, 97224098, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.GELU, False, True, True, False],
    [2, 1, 578, 1408, 352, 88505279, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.GELU, False, True, True, False],
    [1, 1, 578, 1408, 352, 90984235, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU_Tanh_Approx, False, True, True, False],
    [2, 1, 578, 1408, 352, 83078600, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU_Tanh_Approx, False, True, True, False],
    [1, 1, 578, 1408, 352, 91208523, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU, False, True, False, False],
    [2, 1, 578, 1408, 352, 75500049, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, True, ActFnType.GELU, False, True, False, False],
    [1, 1, 578, 1408, 352, 93929554, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, True, ActFnType.GELU, False, False, True, False],
    [2, 1, 578, 1408, 352, 79235293, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU, False, False, True, False],
    [1, 14, 578, 1408, 352, 564325284, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU, False, True, False, False],
    [2, 14, 578, 1408, 352, 351045201, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU, False, True, False, False],
    [1, 1, 578, 1408, 352, 105467001, 2e-2, NormType.LAYER_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU, False, True, True, True],
    [1, 1, 578, 1408, 352, 109600578, 2e-2, NormType.LAYER_NORM, QuantizationType.NONE, True, True, True, ActFnType.GELU, False, True, True, True],
    [2, 1, 578, 1408, 352, 98436096, 2e-2, NormType.LAYER_NORM, QuantizationType.NONE, False, False, True, ActFnType.GELU, False, True, True, True],
    [2, 1, 578, 1408, 352, 105223169, 2e-2, NormType.LAYER_NORM, QuantizationType.NONE, True, True, True, ActFnType.GELU, False, True, True, True],
    [2, 1, 36864, 8192, 512, 8367215426, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.GELU, False, False, False, False],
    [2, 1, 512, 8192, 3584, None, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
]

MLP_CTE_UNIT_TEST_CASES_GATE_BIAS_TRUE = [
    [2, 1, 10240, 3072, 112, 638_002_211, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 10240, 3072, 2160, 6_638_002_211, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 128, 8192, 896, 238002211, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 512, 1024, 448, 80453957, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 512, 1024, 448, 82809370, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 1024, 1024, 448, 92239439, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 1024, 1024, 448, 88019279, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 1024, 1024, 448, 92000000, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 1024, 1024, 448, 90017692, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 768, 1024, 896, 140496530, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 768, 1024, 896, 107670831, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 8192, 8192, 448, 4296859369, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 8192, 8192, 448, 2520729395, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 1024, 16384, 832, 756458068, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 1024, 16384, 832, 895261601, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 1024, 16384, 416, 882105371, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 1024, 16384, 416, 1176268151, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 1024, 8192, 448, 278000000, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 8192, 8192, 448, 4780395634, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 8192, 8192, 448, 3784829419, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 1024, 16384, 416, 1140095218, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 1024, 16384, 416, 1246253552, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 8192, 8192, 448, 2737231223, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 8192, 8192, 448, 2127094342, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 1024, 16384, 416, 595047246, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 1024, 16384, 416, 755884001, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [2, 2, 8192, 8192, 448, 5298071805, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [2, 4, 8192, 8192, 448, 8376700327, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [2, 2, 1024, 16384, 416, 1118514002, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [2, 4, 1024, 16384, 416, 2557272337, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 128, 8192, 896, 210306921, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 128, 8192, 896, 258002211, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 128, 8192, 896, 210306921, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 128, 8192, 896, 258002211, 2e-2, NormType.NO_NORM, QuantizationType.NONE, True, True, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 512, 1024, 448, 80453957, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 512, 1024, 448, 82809370, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 128, 8192, 896, 210306921, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 128, 8192, 896, 210306921, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 2048, 8448, 1408, 1437844003, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 2048, 8448, 1408, 1603737495, 2e-2, NormType.LAYER_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 4096, 8192, 448, 941321030, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 4096, 8192, 448, 1176746495, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 4096, 8192, 448, 1297751306, 2e-2, NormType.LAYER_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 128, 7168, 364, 1.52e8, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 128, 7168, 364, 1.21e8, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 256, 7168, 364, 1.59e8, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [2, 1, 256, 7168, 364, 1.41e8, 2e-2, NormType.RMS_NORM_SKIP_GAMMA, QuantizationType.NONE, True, False, False, ActFnType.SiLU, True, False, False, False],
    [1, 1, 578, 1408, 352, 81643622, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, True, True, False],
    [2, 1, 578, 1408, 352, 81231123, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.SiLU, True, True, True, False],
    [1, 1, 578, 1408, 352, 84836617, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.GELU, True, False, False, False],
    [2, 1, 578, 1408, 352, 81332790, 2e-2, NormType.NO_NORM, QuantizationType.NONE, False, False, False, ActFnType.GELU, True, False, False, False],
    [1, 1, 578, 1408, 352, 99168579, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.GELU, True, True, True, False],
    [2, 1, 578, 1408, 352, 88505279, 2e-2, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, ActFnType.GELU, True, True, True, False],
]

ceilalign = lambda n, a: math.ceil(n / a) * a

MLP_CTE_UNIT_TEST_CASES_ROW_QUANT = [
    [2, 1, 1024, 16384, ceilalign(896, 128), 5.42e8, 4e-2, NormType.NO_NORM, QuantizationType.ROW, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 512, 16384, ceilalign(896, 128), 5.44e8, 4e-2, NormType.NO_NORM, QuantizationType.ROW, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 1, 256, 16384, ceilalign(896, 128), 8.05e8, 4e-2, NormType.NO_NORM, QuantizationType.ROW, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 4, 1024, 16384, ceilalign(896, 128), 1.96e9, 4e-2, NormType.NO_NORM, QuantizationType.ROW, False, False, False, ActFnType.SiLU, False, False, False, False],
    [2, 2, 1024, 16384, ceilalign(896, 128), 1.01e9, 4e-2, NormType.NO_NORM, QuantizationType.ROW, False, False, False, ActFnType.SiLU, False, False, False, False],
]

MLP_CTE_UNIT_TEST_CASES_STATIC_QUANT = [
    # Llama 3.3 70B
    [2, 1, 10240, 8192, ceilalign(1792, 128), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP16
    # Qwen 3 32B
    [2, 1, 10240, 5120, ceilalign(3200, 128), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP8
    [2, 1, 10240, 5120, ceilalign(1600, 128), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP16
]

# Slow-compile
MLP_CTE_UNIT_TEST_CASES_STATIC_QUANT_SLOW = [
    # Llama 3.3 70B
    [2, 1, 10240, 8192, ceilalign(7168, 256), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP4
    [2, 1, 10240, 8192, ceilalign(3584, 128), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP8
    # Qwen 3 32B
    [2, 1, 10240, 5120, ceilalign(6400, 256), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP4
]

MLP_CTE_UNIT_TEST_CASES_STATIC_MX_QUANT = [
    # Llama 3.3 70B
    [2, 1, 10240, 8192, ceilalign(7168, 1024), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC_MX, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP4
    [2, 1, 10240, 8192, ceilalign(3584, 512), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC_MX, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP8
    [2, 1, 10240, 8192, ceilalign(1792, 512), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC_MX, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP16
    # Qwen 3 32B
    [2, 1, 10240, 5120, ceilalign(6400, 1024), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC_MX, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP4
    [2, 1, 10240, 5120, ceilalign(3200, 512), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC_MX, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP8
    [2, 1, 10240, 5120, ceilalign(1600, 512), None, 3e-2, NormType.NO_NORM, QuantizationType.STATIC_MX, False, False, False, ActFnType.SiLU, False, False, False, False],  # TP16
]
# fmt: on


# Parameter names for CTE unit test vectors (matches positional order in raw vector lists)
# fmt: off
CTE_UNIT_PARAM_NAMES = (
    "vnc_degree, batch, seqlen, hidden, intermediate, tpbSgCyclesSum, rtol, "
    "norm_type, quant_type, fused_add, store_add, skip_gate, act_fn_type, "
    "gate_bias, up_bias, down_bias, norm_bias"
)
# fmt: on

# Abbreviations for short test IDs
_CTE_ABBREVS = {
    "vnc_degree": "vnc",
    "batch": "b",
    "seqlen": "s",
    "hidden": "h",
    "intermediate": "i",
    "norm_type": "n_t",
    "quant_type": "q_t",
    "fused_add": "fa",
    "store_add": "sa",
    "skip_gate": "skip_gate",
    "act_fn_type": "act",
    "gate_bias": "gb",
    "up_bias": "ub",
    "down_bias": "db",
    "norm_bias": "nb",
}

# Combined raw vectors for all CTE unit tests
_ALL_CTE_UNIT_RAW_VECTORS = (
    MLP_CTE_UNIT_TEST_CASES_GATE_BIAS_FALSE
    + MLP_CTE_UNIT_TEST_CASES_GATE_BIAS_TRUE
    + MLP_CTE_UNIT_TEST_CASES_ROW_QUANT
    + MLP_CTE_UNIT_TEST_CASES_STATIC_QUANT
)


# Dedup fast vectors, then combine with full-only vectors (which don't get the fast mark)
_ALL_CTE_UNIT_RAW_VECTORS = dedup_test_vectors(_ALL_CTE_UNIT_RAW_VECTORS, ignore_indices={0, 5})

# (seqlen, hidden, intermediate) keys for full-only tests (excluded from fast suite)
_FULL_ONLY_KEYS = {
    (10240, 8192, 1792),
    (10240, 5120, 3200),
    (1024, 16384, 896),
    (10240, 5120, 1664),
    (8192, 8192, 448),
    (36864, 8192, 512),
}

_ALL_CTE_UNIT_PARAMS = [
    pytest.param(*c, marks=pytest.mark.fast) if tuple(c[2:5]) not in _FULL_ONLY_KEYS else c
    for c in _ALL_CTE_UNIT_RAW_VECTORS
] + MLP_CTE_UNIT_TEST_CASES_STATIC_QUANT_SLOW


# ----------------------------------------------------
# CTE sweep dimension values
# Trimmed from original ranges to keep sweep count under ~200 per test method.
# Retains boundary values and representative sizes.
# ----------------------------------------------------
_CTE_SWEEP_SEQLEN = [128, 256, 1024, 6272, 8192]
_CTE_SWEEP_HIDDEN = [128, 256, 512, 1024, 2048, 4096, 7168, 8192, 12288, 15360]
_CTE_SWEEP_INTERMEDIATE = [512, 896, 2048]

# CTE batch sweep dimension values
# Boundary values and ±1 perturbations on fixed dims
_CTE_BATCH_SWEEP_BATCH = [1, 2, 3, 4, 5]
_CTE_BATCH_SWEEP_SEQLEN = [783, 784, 785]
_CTE_BATCH_SWEEP_HIDDEN = [1279, 1280, 1281]
_CTE_BATCH_SWEEP_INTERMEDIATE = [511, 512, 513]

# Feature configs for sweep cross-product
_CTE_SWEEP_FEATURE_CONFIGS = [
    ("basic", BASIC_MLP_CONFIG),
    ("full_features", FULL_FEATURES_CONFIG),
    ("layer_norm", LAYER_NORM_CONFIG),
]

# Constants from kernel code
_SHORT_SEQLEN_THRESHOLD = 256


@pytest_test_metadata(name="MLP CTE", tags=["model"])
@pytest_marks(["mlp", "cte", "mx"])
@final
class TestMlpCteKernel:
    def _kernel_input_generator(self, vec_dict):
        """Generate kernel inputs from a parsed vector dict."""
        d = vec_dict
        lnc_degree = d["vnc_degree"]
        quant_type = d["quant_type"]

        if quant_type == QuantizationType.ROW:
            tensor_generator = gaussian_tensor_generator(
                mean=0.0,
                std=10.0,
                modifier_fn=modify_for_row_quant,
                lnc=lnc_degree,
            )
        elif quant_type in [QuantizationType.STATIC, QuantizationType.STATIC_MX]:
            tensor_generator = gaussian_tensor_generator(
                mean=0.0,
                std=5.0,
                modifier_fn=modify_fp8_static_scale,
                lnc=lnc_degree,
            )
        else:
            tensor_generator = gaussian_tensor_generator()

        kernel_input = build_fused_norm_mlp(
            batch=d["batch"],
            seqlen=d["seqlen"],
            hidden=d["hidden"],
            intermediate=d["intermediate"],
            dtype=nl.bfloat16,
            quantization_type=quant_type,
            quant_dtype=nl.float8_e4m3 if quant_type != QuantizationType.NONE else None,
            is_input_quantized=(quant_type != QuantizationType.NONE),
            fused_add=d["fused_add"],
            norm_type=d["norm_type"],
            store_add=d["store_add"],
            lnc_degree=lnc_degree if lnc_degree > 1 else None,
            skip_gate=d["skip_gate"],
            act_fn_type=d["act_fn_type"],
            gate_bias=d["gate_bias"],
            up_bias=d["up_bias"],
            down_bias=d["down_bias"],
            norm_bias=d["norm_bias"],
            tensor_generator=tensor_generator,
        )
        # Add missing params that mlp() kernel accepts but build_fused_norm_mlp doesn't produce
        kernel_input["quant_clipping_bound"] = 0.0
        kernel_input["force_cte_mode"] = False
        kernel_input["sbm"] = None
        return kernel_input

    @pytest_parametrize(CTE_UNIT_PARAM_NAMES, _ALL_CTE_UNIT_PARAMS, abbrevs=_CTE_ABBREVS)
    def test_mlp_cte_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden,
        intermediate,
        tpbSgCyclesSum,
        rtol,
        norm_type,
        quant_type,
        fused_add,
        store_add,
        skip_gate,
        act_fn_type,
        gate_bias,
        up_bias,
        down_bias,
        norm_bias,
    ):
        # MX/STATIC_MX quant is only supported on TRN3
        if quant_type == QuantizationType.STATIC_MX and not platform_target.is_trn3():
            pytest.skip("STATIC_MX quantization is only supported on TRN3")

        vec_dict = dict(
            vnc_degree=vnc_degree,
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            intermediate=intermediate,
            rtol=rtol,
            norm_type=norm_type,
            quant_type=quant_type,
            fused_add=fused_add,
            store_add=store_add,
            skip_gate=skip_gate,
            act_fn_type=act_fn_type,
            gate_bias=gate_bias,
            up_bias=up_bias,
            down_bias=down_bias,
            norm_bias=norm_bias,
        )
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        _run_mlp_test(
            test_manager=test_manager,
            kernel_input=self._kernel_input_generator(vec_dict),
            compiler_args=compiler_args,
            output_tensor_descriptor=mlp_output_tensor_descriptor,
            rtol=rtol,
        )

    # ============================================================================
    # CTE Sweep Tests
    # ============================================================================

    # @IGNORE_FAST
    @pytest.mark.coverage_parametrize(
        seqlen=_CTE_SWEEP_SEQLEN,
        hidden=_CTE_SWEEP_HIDDEN,
        intermediate=_CTE_SWEEP_INTERMEDIATE,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    @pytest.mark.parametrize(
        "config_name,config",
        _CTE_SWEEP_FEATURE_CONFIGS,
    )
    def test_mlp_cte_sweep(
        self,
        test_manager: Orchestrator,
        seqlen: int,
        hidden: int,
        intermediate: int,
        config_name: str,
        config: dict[str, Any],
        is_negative_test_case: bool,
        platform_target: Platforms,
    ):
        # Pre-existing compiler issues on new test vectors not present on mainline.
        # These fail during compilation (OOM, crashes, missing symbols), not validation.
        # NKILIB-847: https://aws-neuron.atlassian.net/browse/NKILIB-847
        _CTE_COMPILER_XFAILS = {
            (1024, 7168, 2048, "full_features"),  # Heap OOM
            (1024, 15360, 512, "full_features"),  # Heap OOM
            (1024, 12288, 896, "basic"),  # Compiler crash (exit code 70)
            (6272, 8192, 512, "basic"),  # Compiler crash (exit code 70)
            (8192, 12288, 512, "basic"),  # Entry function not found
            (6272, 256, 2048, "basic"),  # _reshape_io_tensor not found
            (1024, 4096, 512, "basic"),  # sigmoid not found
            (128, 512, 2048, "layer_norm"),  # Internal error
        }
        if (seqlen, hidden, intermediate, config_name) in _CTE_COMPILER_XFAILS:
            pytest.xfail(f"Pre-existing compiler issue at seqlen={seqlen},h={hidden},i={intermediate},{config_name}")

        batch = 1
        compiler_args = CompilerArgs(platform_target=platform_target)
        lnc_degree = compiler_args.logical_nc_config

        # --- Legacy negative test logic (from run_range_mlp_cte_test) ---
        bs = batch * seqlen
        gate_bias = config["gate_bias"]
        up_bias = config["up_bias"]
        norm_bias = config["norm_bias"]
        fused_add = config["fused_add"]

        # shard_on_i condition
        shard_on_i = (
            bs <= _SHORT_SEQLEN_THRESHOLD
            and intermediate % 256 == 0
            and not (gate_bias or up_bias or norm_bias)
            and lnc_degree > 1
        )

        # LNC2 sharding: if lnc > 1 and bs > TKG threshold and not shard_on_i,
        # then bs must be divisible by lnc_degree
        if lnc_degree > 1 and bs > TKG_BS_SEQLEN_THRESHOLD and not shard_on_i:
            if bs % lnc_degree != 0:
                is_negative_test_case = True

        # OOM skip: large seqlen with fused_add and large hidden or intermediate causes heap OOM
        if seqlen >= 6272 and fused_add and (intermediate == 2048 or hidden >= 12288):
            pytest.skip("OOM tests")

        # --- Build kernel inputs ---
        norm_type = config["norm_type"]
        store_add = config["store_add"]
        skip_gate = config["skip_gate"]
        act_fn_type = config["act_fn_type"]
        down_bias = config["down_bias"]

        tensor_generator = gaussian_tensor_generator()

        kernel_input = build_fused_norm_mlp(
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            intermediate=intermediate,
            dtype=nl.bfloat16,
            quantization_type=QuantizationType.NONE,
            quant_dtype=None,
            is_input_quantized=False,
            fused_add=fused_add,
            norm_type=norm_type,
            store_add=store_add,
            lnc_degree=lnc_degree if lnc_degree > 1 else None,
            skip_gate=skip_gate,
            act_fn_type=act_fn_type,
            gate_bias=gate_bias,
            up_bias=up_bias,
            down_bias=down_bias,
            norm_bias=norm_bias,
            tensor_generator=tensor_generator,
        )
        kernel_input["quant_clipping_bound"] = 0.0
        kernel_input["force_cte_mode"] = False
        kernel_input["sbm"] = None

        _run_mlp_test(
            test_manager,
            kernel_input,
            compiler_args,
            mlp_output_tensor_descriptor,
            is_negative_test=is_negative_test_case,
        )

    # ============================================================================
    # CTE Batch Sweep Tests
    # ============================================================================

    # @IGNORE_FAST
    @pytest.mark.coverage_parametrize(
        batch=_CTE_BATCH_SWEEP_BATCH,
        seqlen=_CTE_BATCH_SWEEP_SEQLEN,
        hidden=_CTE_BATCH_SWEEP_HIDDEN,
        intermediate=_CTE_BATCH_SWEEP_INTERMEDIATE,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    @pytest.mark.parametrize(
        "config_name,config",
        _CTE_SWEEP_FEATURE_CONFIGS,
    )
    def test_mlp_cte_batch_sweep(
        self,
        test_manager: Orchestrator,
        batch: int,
        seqlen: int,
        hidden: int,
        intermediate: int,
        config_name: str,
        config: dict[str, Any],
        is_negative_test_case: bool,
        platform_target: Platforms,
    ):
        compiler_args = CompilerArgs(platform_target=platform_target)
        lnc_degree = compiler_args.logical_nc_config

        # --- Kernel dimension constraints (boundary values may violate these) ---
        if hidden % 128 != 0 or batch * seqlen == 0:
            is_negative_test_case = True

        # --- Legacy negative test logic (from run_range_mlp_cte_test) ---
        bs = batch * seqlen
        gate_bias = config["gate_bias"]
        up_bias = config["up_bias"]
        norm_bias = config["norm_bias"]
        fused_add = config["fused_add"]

        # shard_on_i condition
        shard_on_i = (
            bs <= _SHORT_SEQLEN_THRESHOLD
            and intermediate % 256 == 0
            and not (gate_bias or up_bias or norm_bias)
            and lnc_degree > 1
        )

        # LNC2 sharding: H1 must be divisible by lnc_degree
        H1 = hidden // 128
        if lnc_degree > 1 and H1 % lnc_degree != 0:
            is_negative_test_case = True

        # LNC2 sharding: if lnc > 1 and bs > TKG threshold and not shard_on_i,
        # then bs must be divisible by lnc_degree
        if lnc_degree > 1 and bs > TKG_BS_SEQLEN_THRESHOLD and not shard_on_i:
            if bs % lnc_degree != 0:
                is_negative_test_case = True

        # OOM skip: large seqlen with fused_add and large hidden or intermediate causes heap OOM
        if seqlen >= 6272 and fused_add and (intermediate == 2048 or hidden >= 12288):
            pytest.skip("OOM tests")

        # --- Build kernel inputs ---
        norm_type = config["norm_type"]
        store_add = config["store_add"]
        skip_gate = config["skip_gate"]
        act_fn_type = config["act_fn_type"]
        down_bias = config["down_bias"]

        tensor_generator = gaussian_tensor_generator()

        kernel_input = build_fused_norm_mlp(
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            intermediate=intermediate,
            dtype=nl.bfloat16,
            quantization_type=QuantizationType.NONE,
            quant_dtype=None,
            is_input_quantized=False,
            fused_add=fused_add,
            norm_type=norm_type,
            store_add=store_add,
            lnc_degree=lnc_degree if lnc_degree > 1 else None,
            skip_gate=skip_gate,
            act_fn_type=act_fn_type,
            gate_bias=gate_bias,
            up_bias=up_bias,
            down_bias=down_bias,
            norm_bias=norm_bias,
            tensor_generator=tensor_generator,
        )
        kernel_input["quant_clipping_bound"] = 0.0
        kernel_input["force_cte_mode"] = False
        kernel_input["sbm"] = None

        _run_mlp_test(
            test_manager,
            kernel_input,
            compiler_args,
            mlp_output_tensor_descriptor,
            is_negative_test=is_negative_test_case,
        )


@pytest_marks(["mlp", "cte", "model", "mx"])
@final
class TestMlpCteModel:
    """Model-driven tests for MLP CTE kernel, organized by tier."""

    _OPTIMAL_PARAMS, _OPTIMAL_IDS = (
        prepare_model_parametrize({ModelTestType.OPTIMAL: mlp_cte_model_configs.get(ModelTestType.OPTIMAL, [])})
        if mlp_cte_model_configs
        else ([], [])
    )

    def _run_model_test(self, **kwargs):
        """Common test logic for model tiers."""
        seqlen = kwargs["seqlen"]
        quant_type = kwargs["quant_type"]
        platform_target = kwargs["platform_target"]
        test_manager = kwargs["test_manager"]
        vnc_degree = kwargs["vnc_degree"]
        rtol = kwargs["rtol"]

        if seqlen <= 64 and quant_type == QuantizationType.ROW:
            pytest.skip("ROW quant with very small seqlen is a known kernel limitation")
        if quant_type == QuantizationType.STATIC_MX and not platform_target.is_trn3():
            pytest.skip("STATIC_MX only supported on TRN3")
        if quant_type in (QuantizationType.STATIC, QuantizationType.ROW) and platform_target.is_trn3():
            pytest.skip("STATIC/ROW fp8 not supported on TRN3")

        vec_dict = {k: v for k, v in kwargs.items() if k not in ["test_manager", "platform_target", "tpbSgCyclesSum"]}
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        _run_mlp_test(
            test_manager=test_manager,
            kernel_input=TestMlpCteKernel()._kernel_input_generator(vec_dict),
            compiler_args=compiler_args,
            output_tensor_descriptor=mlp_output_tensor_descriptor,
            rtol=rtol,
        )

    @pytest.mark.optimal
    @pytest.mark.parametrize(CTE_UNIT_PARAM_NAMES, _OPTIMAL_PARAMS, ids=_OPTIMAL_IDS)
    def test_optimal(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden,
        intermediate,
        tpbSgCyclesSum,
        rtol,
        norm_type,
        quant_type,
        fused_add,
        store_add,
        skip_gate,
        act_fn_type,
        gate_bias,
        up_bias,
        down_bias,
        norm_bias,
    ):
        """OPTIMAL: Performance-optimized model configs."""
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        self._run_model_test(**kwargs)
