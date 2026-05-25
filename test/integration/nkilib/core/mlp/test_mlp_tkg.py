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
from typing import Any, final

try:
    from test.integration.nkilib.core.mlp.test_mlp_tkg_model_config import (
        mlp_tkg_model_configs,
    )
except ImportError:
    mlp_tkg_model_configs = []

import nki.language as nl
import pytest

from nkilib_src.nkilib.core.mlp.mlp import mlp as mlp_kernel
from nkilib_src.nkilib.core.utils.common_types import ActFnType, ComputationMode, NormType, QuantizationType
from test.integration.nkilib.core.mlp.test_mlp_common import (
    _run_mlp_test,
    build_fused_norm_mlp,
    copy_sbuf_output_to_hbm,
    dedup_test_vectors,
    gaussian_tensor_generator,
    mlp_output_tensor_descriptor,
    modify_down_proj_lhs_rhs_swap_unit_stride_layout,
    random_lhs_and_random_bound_weight_tensor_generator,
    setup_sbuf_input,
)
from test.utils.common_dataclasses import (
    TKG_INFERENCE_ARGS,
    CompilerArgs,
    ModelTestType,
    Platforms,
    SeparationPassMode,
    prepare_model_parametrize,
)
from test.utils.coverage_parametrized_tests import FilterResult
from test.utils.metadata_loader import load_model_configs
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator

# ----------------------------------------------------
# Configuration-based testing to avoid combinatorial explosion
# ----------------------------------------------------

COLUMN_TILING_BASIC_CONFIG = {
    "dtype": nl.bfloat16,
    "norm_type": NormType.NO_NORM,
    "fused_add": False,
    "store_add": False,
    "act_fn_type": ActFnType.SiLU,
    "gate_bias": False,
    "up_bias": False,
    "down_bias": False,
    "norm_bias": False,
    "quant_dtype": None,
    "use_tkg_gate_up_proj_column_tiling": True,
    "use_tkg_down_proj_column_tiling": True,
    "use_tkg_down_proj_optimized_layout": False,
}

COLUMN_TILING_FULL_FEATURE_RMSNORM_CONFIG = {
    "dtype": nl.bfloat16,
    "norm_type": NormType.RMS_NORM,
    "fused_add": True,
    "store_add": True,
    "act_fn_type": ActFnType.SiLU,
    "gate_bias": True,
    "up_bias": True,
    "down_bias": True,
    "norm_bias": False,
    "quant_dtype": None,
    "use_tkg_gate_up_proj_column_tiling": True,
    "use_tkg_down_proj_column_tiling": True,
    "use_tkg_down_proj_optimized_layout": False,
}

COLUMN_TILING_FULL_FEATURE_LAYERNORM_CONFIG = {
    "dtype": nl.bfloat16,
    "norm_type": NormType.LAYER_NORM,
    "fused_add": True,
    "store_add": True,
    "act_fn_type": ActFnType.SiLU,
    "gate_bias": True,
    "up_bias": True,
    "down_bias": True,
    "norm_bias": True,
    "quant_dtype": None,
    "use_tkg_gate_up_proj_column_tiling": True,
    "use_tkg_down_proj_column_tiling": True,
    "use_tkg_down_proj_optimized_layout": False,
}

NON_COLUMN_TILING_BASIC_CONFIG = {
    "dtype": nl.bfloat16,
    "norm_type": NormType.NO_NORM,
    "fused_add": False,
    "store_add": False,
    "act_fn_type": ActFnType.SiLU,
    "gate_bias": False,
    "up_bias": False,
    "down_bias": False,
    "norm_bias": False,
    "quant_dtype": None,
    "use_tkg_gate_up_proj_column_tiling": False,
    "use_tkg_down_proj_column_tiling": False,
    "use_tkg_down_proj_optimized_layout": False,
}

NON_COLUMN_TILING_FULL_FEATURE_CONFIG = {
    "dtype": nl.bfloat16,
    "norm_type": NormType.RMS_NORM,
    "fused_add": True,
    "store_add": True,
    "act_fn_type": ActFnType.SiLU,
    "gate_bias": True,
    "up_bias": True,
    "down_bias": True,
    "norm_bias": False,
    "quant_dtype": None,
    "use_tkg_gate_up_proj_column_tiling": False,
    "use_tkg_down_proj_column_tiling": False,
    "use_tkg_down_proj_optimized_layout": True,
}


# fmt: off
# Parameters: vnc_degree, batch, seqlen, hidden, intermediate, dtype, quant_dtype, quant_type,
#             tpbSgCyclesSum, norm_type, fused_add, store_add, act_fn_type, skip_gate_proj,
#             gate_bias, up_bias, down_bias, norm_bias, use_tkg_gate_up_proj_column_tiling,
#             use_tkg_down_proj_column_tiling, use_tkg_down_proj_optimized_layout
nki_tkg_fused_norm_mlp_kernel_spmd_vnc2_params = [
    [2, 2, 4, 8448, 1408, nl.bfloat16, None, QuantizationType.NONE, 146806437, NormType.LAYER_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 4, 2, 4096, 1024, nl.bfloat16, None, QuantizationType.NONE, 83028204, NormType.LAYER_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 1, 1, 8448, 1408, nl.bfloat16, None, QuantizationType.NONE, 135269789, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 1, 1, 8192, 448, nl.bfloat16, None, QuantizationType.NONE, 62157403, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 4, 5, 8192, 896, nl.bfloat16, None, QuantizationType.NONE, 122126476, NormType.RMS_NORM, True, True, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 1, 5, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 153948092, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 4, 7, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 157358921, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    # GPT-OSS draft
    [2, 64, 1, 3072, 135, nl.bfloat16, None, QuantizationType.NONE, 50378255, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 4, 1, 3072, 2160, nl.bfloat16, None, QuantizationType.NONE, 102256507, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    # Llama high batch with 2x array tiling
    [2, 8, 5, 8192, 896, nl.bfloat16, None, QuantizationType.NONE, 115811486, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 8, 5, 16384, 896, nl.bfloat16, None, QuantizationType.NONE, 172327230, NormType.RMS_NORM, True, True, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    # Llama3 1B fused speculation
    [2, 1, 1, 2048, 512, nl.bfloat16, None, QuantizationType.NONE, 42469100, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    # Llama3 8B fused speculation
    [2, 1, 1, 4096, 896, nl.bfloat16, None, QuantizationType.NONE, 66799895, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    # Text
    [2, 1, 1, 7168, 364, nl.bfloat16, None, QuantizationType.NONE, 50260755, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    #  Llama3 2T fused speculation
    [2, 1, 5, 32768, 896, nl.bfloat16, None, QuantizationType.NONE, 292073710, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 1, 5, 32768, 896, nl.bfloat16, None, QuantizationType.NONE, 300053697, NormType.RMS_NORM, True, True, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    #  Llama3 470B
    [2, 1, 5, 20480, 832, nl.bfloat16, None, QuantizationType.NONE, 179771386, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 2, 7, 20480, 832, nl.bfloat16, None, QuantizationType.NONE, 197313858, NormType.RMS_NORM, True, True, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    # Llama 4 sharedExpert
    [2, 4, 1, 5120, 128, nl.bfloat16, None, QuantizationType.NONE, 34008280, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 8, 7, 16384, 896, nl.bfloat16, None, QuantizationType.NONE, 190164703, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, True, False, False, False],
    [2, 8, 5, 8192, 512, nl.bfloat16, None, QuantizationType.NONE, 93846520, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, True, False, False, False],
    # Functional Test I > 4096
    [2, 4, 1, 8192, 5120, nl.bfloat16, None, QuantizationType.NONE, 435362653, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, True, False, False, False],
    # Store Add, sb_input feature in rmsnorm/layernorm
    [2, 4, 1, 8192, 512, nl.bfloat16, None, QuantizationType.NONE, 75694882, NormType.RMS_NORM, True, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 4, 8, 8192, 512, nl.bfloat16, None, QuantizationType.NONE, 109073163, NormType.RMS_NORM, True, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 4, 1, 8192, 512, nl.bfloat16, None, QuantizationType.NONE, 82688204, NormType.LAYER_NORM, True, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 4, 8, 8192, 512, nl.bfloat16, None, QuantizationType.NONE, 104209837, NormType.LAYER_NORM, True, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    # Gemma3
    [2, 1, 1, 5376, 336, nl.bfloat16, None, QuantizationType.NONE, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
]

nki_tkg_fused_norm_mlp_kernel_spmd_vnc1_params = [
    [1, 1, 1, 8192, 448, nl.bfloat16, None, QuantizationType.NONE, 76487380, NormType.RMS_NORM, True, True, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [1, 1, 1, 7168, 896, nl.bfloat16, None, QuantizationType.NONE, 112156491, NormType.RMS_NORM, True, True, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [1, 1, 1, 16384, 416, nl.bfloat16, None, QuantizationType.NONE, 118365648, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [1, 1, 1, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 224507149, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    # Speculative tests, seqlen > 1
    [1, 3, 8, 8192, 832, nl.bfloat16, None, QuantizationType.NONE, 124893971, NormType.LAYER_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [1, 4, 2, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 229137142, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    # Bias test.
    [1, 4, 2, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 242479621, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, True, False, False, False],
    # Bias test with larger BxS
  	[1, 32, 2, 8192, 512, nl.bfloat16, None, QuantizationType.NONE, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, True, False, False, False],
    # small H test
    [1, 1, 1, 256, 448, nl.bfloat16, None, QuantizationType.NONE, None, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
]

nki_tkg_fused_norm_mlp_kernel_spmd_vnc2_swap_perms = [
    # LLaMA4 sharedExpert
    [2, 1, 1, 4096, 128, nl.bfloat16, None, QuantizationType.NONE, 25566627, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, True, False, False, False],
    [2, 4, 1, 5120, 256, nl.bfloat16, None, QuantizationType.NONE, 45419096, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    [2, 8, 1, 5120, 128, nl.bfloat16, None, QuantizationType.NONE, 41267436, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    [2, 16, 1, 5120, 128, nl.bfloat16, None, QuantizationType.NONE, 45513262, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    [2, 32, 1, 5120, 128, nl.bfloat16, None, QuantizationType.NONE, 47358259, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # LLaMA3 TP64
    [2, 4, 1, 8192, 512, nl.bfloat16, None, QuantizationType.NONE, 82889870, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # LLaMA3 TP32
    [2, 4, 5, 8192, 1024, nl.bfloat16, None, QuantizationType.NONE, 131043962, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # Functional Test I > 1024
    [2, 4, 5, 8192, 1560, nl.bfloat16, None, QuantizationType.NONE, 182945547, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False, False, False],
    # Functional test
    [2, 4, 1, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 153945593, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 4, 1, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 186434709, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False, False, False],
    [2, 4, 1, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 150452265, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, True, False, False, False],
    [2, 4, 1, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 189018038, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # Llama 3 TP32 all projections swapped, down_w layout optimized for unit stride loading
    [1, 4, 1, 8192, 1024, nl.bfloat16, None, QuantizationType.NONE, 166719739, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, True, False, False],
    [2, 4, 1, 8192, 1024, nl.bfloat16, None, QuantizationType.NONE, 117065650, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, True, False, False],
]

nki_tkg_fused_norm_mlp_kernel_spmd_skip_gate = [
    # Functional test
    [2, 4, 1, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 109052330, NormType.NO_NORM, False, False, ActFnType.SiLU, True, False, False, False, False, True, True, False, False, False],
    [2, 4, 1, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 146529771, NormType.NO_NORM, False, False, ActFnType.SiLU, True, False, False, False, False, True, False, False, False, False],
    [2, 4, 1, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 116152318, NormType.NO_NORM, False, False, ActFnType.SiLU, True, False, False, False, False, False, True, False, False, False],
    [2, 4, 1, 16384, 832, nl.bfloat16, None, QuantizationType.NONE, 142690611, NormType.NO_NORM, False, False, ActFnType.SiLU, True, False, False, False, False, False, False, False, False, False],
]

nki_tkg_fused_norm_mlp_row_quant_kernel_params = [
    [2, 1, 1, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 106305668, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [1, 1, 1, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 140940613, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [1, 1, 5, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 145970605, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 1, 5, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 108908163, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 1, 7, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 105090669, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 2, 7, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 109548162, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [1, 4, 7, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 158341419, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 8, 7, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 152673928, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    # BxS > 64
    [2, 14, 5, 8192, 128, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 64228233, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 16, 5, 8192, 128, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 62225736, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    # LLaMA3 70B
    [2, 4, 5, 8192, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 77140713, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 8, 5, 8192, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 87974029, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    # llama3 470B
    [2, 4, 7, 20480, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 126873135, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 2, 7, 20480, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 134724789, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 4, 7, 20480, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 145678105, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    #llama3-2T
    [2, 4, 7, 32768, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, None, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, True, False, False, False],
    [2, 2, 7, 32768, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 188562206, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 4, 7, 32768, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 186326376, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    # Functional Test I > 4096
    [2, 1, 1, 16384, 4986, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, None, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, True, False, False, False],
]

nki_tkg_fused_norm_mlp_row_quant_kernel_layout_swap_perms = [
    # LLaMA3 70B TP64-BS8
    [2, 8, 5, 8192, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # LLaMA3 70B TP32-DP2-BS4 (effective BS8)
    [2, 4, 5, 8192, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 112488158, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # With gate/up column tiling
    # LLaMA3 70B TP64-BS8
    [2, 8, 5, 8192, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 83822369, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, True, False, False, False],
    # LLaMA3 70B TP32-DP2-BS4 (effective BS8)
    [2, 4, 5, 8192, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 93608187, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, True, False, False, False],
    # Column tiling / swap option minimal tests
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 51984086, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 51378253, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False, False, False],
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 43303265, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, True, False, False, False],
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 48179925, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # Llama 3 TP32 all projections swapped, down_w layout optimized for unit stride loading
    [1, 4, 1, 8192, 1024, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 127450635, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, True, False, False],
    [2, 4, 1, 8192, 1024, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 87849029, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, True, False, False],
    # Bias test
    [2, 4, 1, 8192, 893, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 89110694, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, True, False, False, False],
    [2, 4, 1, 8192, 893, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 95652350, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, False, False, False, False],
    [2, 4, 1, 8192, 893, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 71973221, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, True, False, False, False],
    [2, 4, 1, 8192, 893, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 88949028, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False, False, False],
    # Functional Test I > 1024
    [2, 4, 1, 8192, 1560, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 131467295, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False, False, False],
]

nki_tkg_fused_norm_mlp_static_quant_kernel_params = [
    [2, 1, 1, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 105202336, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [1, 1, 1, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 151766430, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [1, 1, 5, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 166568907, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 1, 5, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 110590661, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 1, 7, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 112503157, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 2, 7, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 119785646, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [1, 4, 7, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 177883055, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 8, 7, 16384, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 147633102, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    # BxS > 64
    [2, 14, 5, 8192, 128, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 57351578, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 16, 5, 8192, 128, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 56858245, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    # LLaMA3 70B
    [2, 4, 5, 8192, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 79875708, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 8, 5, 8192, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 89981526, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    # llama3 470B
    [2, 4, 7, 20480, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 141169779, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 2, 7, 20480, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 137068120, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 4, 7, 20480, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 139171450, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    #llama3-2T
    [2, 4, 7, 32768, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, None, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, False, False, True, True, False, False, False],
    [2, 2, 7, 32768, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 201567185, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 4, 7, 32768, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 202325517, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    # Functional Test I > 4096
    [2, 4, 7, 8192, 5120, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, True, False, False, False],
]

nki_tkg_fused_norm_mlp_static_quant_kernel_layout_swap_perms = [
    # LLaMA3 70B TP64-BS8
    [2, 8, 5, 8192, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, None, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # LLaMA3 70B TP32-DP2-BS4 (effective BS8)
    [2, 4, 5, 8192, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 95099018, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # With gate/up column tiling
    # LLaMA3 70B TP64-BS8
    [2, 8, 5, 8192, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 88882361, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, True, False, False, False],
    # LLaMA3 70B TP32-DP2-BS4 (effective BS8)
    [2, 4, 5, 8192, 896, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 87385697, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, True, False, False, False],
    # Column tiling / swap option minimal tests
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 46429095, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 48652424, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False, False, False],
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 42154934, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, True, False, False, False],
    [2, 4, 5, 1024, 512, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 43614932, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # Llama 3 TP32 all projections swapped, down_w layout optimized for unit stride loading
    [1, 4, 1, 8192, 1024, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 124933138, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, True, False, False],
    [2, 4, 1, 8192, 1024, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 86152365, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, True, False, False],
    # Bias test
    [2, 4, 1, 8192, 893, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 91222358, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, True, False, False, False],
    [2, 4, 1, 8192, 893, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 99567345, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, True, False, False, False, False],
    [2, 4, 1, 8192, 893, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 75601548, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, True, False, False, False],
    [2, 4, 1, 8192, 893, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 87649030, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False, False, False],
    # Functional Test I > 1024
    [2, 4, 1, 8192, 1560, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 132462293, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False, False, False],
]

nki_tkg_fused_norm_mlp_mx_quant_kernel_params = [
    # LNC1 functional test
    [1, 2, 1, 3072, 384, nl.bfloat16, nl.float4_e2m1fn_x4, QuantizationType.MX, None, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False, False, False],
    [1, 2, 1, 8192, 448, nl.bfloat16, nl.float8_e4m3fn_x4, QuantizationType.MX, None, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False, False, False],

    # mxfp4 test - using float4_e2m1fn_x4 dtype
    [2, 4, 1, 3072, 384, nl.bfloat16, nl.float4_e2m1fn_x4, QuantizationType.MX, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # llama 3.3 70B TP64
    [2, 64, 1, 8192, 448, nl.bfloat16, nl.float4_e2m1fn_x4, QuantizationType.MX, None, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False, False, False],
    [2, 64, 1, 8192, 448, nl.bfloat16, nl.float4_e2m1fn_x4, QuantizationType.MX, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False, False, False],
    # llama 3.3 70B TP8
    [2, 64, 1, 8192, 3584, nl.bfloat16, nl.float4_e2m1fn_x4, QuantizationType.MX, None, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False, False, False],
    [2, 64, 1, 8192, 3584, nl.bfloat16, nl.float4_e2m1fn_x4, QuantizationType.MX, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False, False, False],

    # mxfp8 test - using float8_e4m3fn_x4, float8_e5m2_x4 dtype
    [2, 4, 1, 3072, 384, nl.bfloat16, nl.float8_e4m3fn_x4, QuantizationType.MX, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # llama 3.3 70B TP64
    [2, 64, 1, 8192, 448, nl.bfloat16, nl.float8_e4m3fn_x4, QuantizationType.MX, None, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False, False, False],
    [2, 64, 1, 8192, 448, nl.bfloat16, nl.float8_e5m2_x4, QuantizationType.MX, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False, False, False],
    # llama 3.3 70B TP8
    [2, 64, 1, 8192, 3584, nl.bfloat16, nl.float8_e4m3fn_x4, QuantizationType.MX, None, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False, False, False],
    [2, 64, 1, 8192, 3584, nl.bfloat16, nl.float8_e5m2_x4, QuantizationType.MX, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False, False, False],
]
# fmt: on

# fmt: off
nki_tkg_fused_norm_mlp_static_mx_quant_kernel_params = [
    # ── Core paths: RMS_NORM (SBUF) vs NO_NORM (HBM), single vs multi-token ──
    [2, 1, 1, 8192, 448, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC_MX, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    [2, 4, 1, 8192, 448, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC_MX, None, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    [2, 1, 5, 8192, 448, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC_MX, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # ── I >= 512 (large-I path) ──
    [2, 64, 1, 8192, 3584, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC_MX, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False, False, False],
    # ── Very small I (I < 512, I=224) ──
    [2, 1, 1, 8192, 224, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC_MX, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # ── LNC1 ──
    [1, 2, 1, 8192, 448, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC_MX, None, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # ── Bias: RMS_NORM + bias ──
    [2, 4, 1, 8192, 448, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC_MX, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False, False, False],
    # ── Bias: NO_NORM + bias ──
    [2, 4, 1, 8192, 448, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC_MX, None, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False, False, False],
    # ── LNC1 + bias ──
    [1, 2, 1, 8192, 448, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC_MX, None, NormType.NO_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False, False, False],
    # ── Large batch (boundary near TKG threshold) ──
    [2, 64, 1, 8192, 448, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC_MX, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
]
# fmt: on

# fmt: off
nki_tkg_fused_norm_mlp_row_mx_quant_kernel_params = [
    # ── Core paths: RMS_NORM (SBUF) vs NO_NORM (HBM), single vs multi-token ──
    [2, 1, 1, 5120, 384, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW_MX, None, NormType.RMS_NORM, False, False, ActFnType.GELU_Tanh_Approx, False, False, False, False, False, False, False, False, False, False],
    [2, 4, 1, 5120, 384, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW_MX, None, NormType.NO_NORM, False, False, ActFnType.GELU_Tanh_Approx, False, False, False, False, False, False, False, False, False, False],
    [2, 1, 5, 5120, 384, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW_MX, None, NormType.RMS_NORM, False, False, ActFnType.GELU_Tanh_Approx, False, False, False, False, False, False, False, False, False, False],
    # ── I >= 512 (multi-tile I, non-512-aligned) ──
    [2, 4, 1, 5120, 800, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW_MX, None, NormType.RMS_NORM, False, False, ActFnType.GELU_Tanh_Approx, False, False, False, False, False, False, False, False, False, False],
    # ── Large I (real model config) ──
    [2, 1, 1, 5120, 3200, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW_MX, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # ── LNC1 ──
    [1, 2, 1, 5120, 384, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW_MX, None, NormType.NO_NORM, False, False, ActFnType.GELU_Tanh_Approx, False, False, False, False, False, False, False, False, False, False],
    # ── Bias: gate + up + down with RMS_NORM ──
    [2, 4, 1, 5120, 384, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW_MX, None, NormType.RMS_NORM, False, False, ActFnType.GELU_Tanh_Approx, False, True, True, True, False, False, False, False, False, False],
    # ── Bias: NO_NORM + non-512-aligned I ──
    [2, 4, 1, 5120, 800, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW_MX, None, NormType.NO_NORM, False, False, ActFnType.GELU_Tanh_Approx, False, True, True, True, False, False, False, False, False, False],
    # ── SiLU activation ──
    [2, 1, 1, 5120, 384, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW_MX, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # ── SiLU + bias combined ──
    [2, 4, 1, 5120, 384, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW_MX, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, True, True, True, False, False, False, False, False, False],
    # ── Large batch (boundary: b=64 near TKG threshold) ──
    [2, 64, 1, 5120, 384, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW_MX, None, NormType.RMS_NORM, False, False, ActFnType.GELU_Tanh_Approx, False, False, False, False, False, False, False, False, False, False],
    # ── Small H (higher relative FP8 quantization noise) ──
    [2, 1, 1, 1024, 384, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW_MX, None, NormType.RMS_NORM, False, False, ActFnType.GELU_Tanh_Approx, False, False, False, False, False, False, False, False, False, False],
]
# fmt: on


# fmt: off
nki_tkg_fused_norm_mlp_kernel_separation_pass_params = [
    # Llama3 1B - small
    [2, 1, 1, 2048, 512, nl.bfloat16, None, QuantizationType.NONE, 42469100, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    # Llama3 8B - medium
    [2, 1, 1, 4096, 896, nl.bfloat16, None, QuantizationType.NONE, 66799895, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    # Large hidden with fused_add + store_add
    [2, 4, 5, 8192, 896, nl.bfloat16, None, QuantizationType.NONE, 122126476, NormType.RMS_NORM, True, True, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
    # Llama3 470B - large
    [2, 1, 5, 20480, 832, nl.bfloat16, None, QuantizationType.NONE, 179771386, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, False, False],
]
# fmt: on


# ============================================================================
# UTF Migration: TestMlpTkgKernel (new UTF-based class)
# ============================================================================

# Parameter names for TKG unit test vectors (matches positional order in raw vector lists)
# fmt: off
TKG_UNIT_PARAM_NAMES = (
    "vnc_degree, batch, seqlen, hidden, intermediate, dtype, quant_dtype, quant_type, "
    "tpbSgCyclesSum, norm_type, fused_add, store_add, act_fn_type, skip_gate, "
    "gate_bias, up_bias, down_bias, norm_bias, "
    "use_tkg_gate_up_proj_column_tiling, use_tkg_down_proj_column_tiling, use_tkg_down_proj_optimized_layout, "
    "transposed_in, transposed_out"
)
# fmt: on

# Abbreviations for short test IDs (matching legacy DIM_NAME constants)
_TKG_ABBREVS = {
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
    "use_tkg_gate_up_proj_column_tiling": "gate_col",
    "use_tkg_down_proj_column_tiling": "down_col",
    "use_tkg_down_proj_optimized_layout": "down_opt",
    "transposed_in": "tin",
    "transposed_out": "tout",
}

# Transposed I/O configs (transposed_in=True, transposed_out=True)
# Input: [H0, n_prgs, H1_shard, BxS], Output: [H0, n_prgs*H1_shard*BxS]
# use_tkg_down_proj_column_tiling=False required for transposed_out
# fmt: off
nki_tkg_transposed_io_params = [
    # llama3_70b - NONE quant
    [2, 1, 1, 8192, 448, nl.bfloat16, None, QuantizationType.NONE, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False, True, True],
    [2, 1, 5, 8192, 448, nl.bfloat16, None, QuantizationType.NONE, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False, True, True],
    [2, 64, 1, 8192, 448, nl.bfloat16, None, QuantizationType.NONE, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False, True, True],
    # llama3_70b - STATIC quant
    [2, 1, 1, 8192, 448, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False, True, True],
    [2, 1, 5, 8192, 448, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False, True, True],
    [2, 64, 1, 8192, 448, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False, True, True],
    # llama3_70b - ROW quant
    [2, 1, 5, 8192, 448, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False, True, True],
    [2, 64, 1, 8192, 448, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False, True, True],
    # llama3_70b - smaller intermediate
    [2, 1, 1, 8192, 224, nl.bfloat16, None, QuantizationType.NONE, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False, True, True],
    [2, 1, 1, 8192, 224, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False, True, True],
    # qwen3_32b - NONE quant
    [2, 1, 1, 5120, 400, nl.bfloat16, None, QuantizationType.NONE, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False, True, True],
    [2, 64, 1, 5120, 400, nl.bfloat16, None, QuantizationType.NONE, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False, True, True],
    # qwen3_32b - ROW quant
    [2, 1, 1, 5120, 400, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False, True, True],
    [2, 64, 1, 5120, 400, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False, True, True],
    # gemma3_27b - NONE quant
    [2, 1, 1, 5376, 336, nl.bfloat16, None, QuantizationType.NONE, 0, NormType.RMS_NORM, False, False, ActFnType.GELU_Tanh_Approx, False, False, False, False, False, True, False, False, True, True],
    [2, 16, 1, 5376, 336, nl.bfloat16, None, QuantizationType.NONE, 0, NormType.RMS_NORM, False, False, ActFnType.GELU_Tanh_Approx, False, False, False, False, False, True, False, False, True, True],
    # gemma3_27b - ROW quant
    [2, 1, 1, 5376, 336, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 0, NormType.RMS_NORM, False, False, ActFnType.GELU_Tanh_Approx, False, False, False, False, False, True, False, False, True, True],
    [2, 16, 1, 5376, 336, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 0, NormType.RMS_NORM, False, False, ActFnType.GELU_Tanh_Approx, False, False, False, False, False, True, False, False, True, True],
    # llama3_70b - B=16 S=1, large I=1792 (NxDI model match)
    [2, 16, 1, 8192, 1792, nl.bfloat16, None, QuantizationType.NONE, None, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False, True, True],
    # transposed_in=True, transposed_out=False with down_col=True
    [2, 16, 1, 8192, 448, nl.bfloat16, None, QuantizationType.NONE, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, True, False, True, False],
    # transposed_in=True, transposed_out=True, NO_NORM (non-fused rmsnorm case)
    [2, 16, 1, 8192, 448, nl.bfloat16, None, QuantizationType.NONE, 0, NormType.NO_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, True, False, False, True, True],
]
# fmt: on

# Baseline (non-transposed) counterparts for the transposed configs above.
# Same model dims, quant, down_col=False — but transposed_in=False, transposed_out=False.
# fmt: off
nki_tkg_transposed_io_baseline_params = [
    # llama3_70b - NONE quant
    [2, 1, 1, 8192, 448, nl.bfloat16, None, QuantizationType.NONE, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    [2, 1, 5, 8192, 448, nl.bfloat16, None, QuantizationType.NONE, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    [2, 64, 1, 8192, 448, nl.bfloat16, None, QuantizationType.NONE, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # llama3_70b - STATIC quant
    [2, 1, 1, 8192, 448, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    [2, 1, 5, 8192, 448, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    [2, 64, 1, 8192, 448, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # llama3_70b - ROW quant
    [2, 1, 5, 8192, 448, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    [2, 64, 1, 8192, 448, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # llama3_70b - smaller intermediate
    [2, 1, 1, 8192, 224, nl.bfloat16, None, QuantizationType.NONE, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    [2, 1, 1, 8192, 224, nl.bfloat16, nl.float8_e4m3, QuantizationType.STATIC, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # qwen3_32b - NONE quant
    [2, 1, 1, 5120, 400, nl.bfloat16, None, QuantizationType.NONE, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    [2, 64, 1, 5120, 400, nl.bfloat16, None, QuantizationType.NONE, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # qwen3_32b - ROW quant
    [2, 1, 1, 5120, 400, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    [2, 64, 1, 5120, 400, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 0, NormType.RMS_NORM, False, False, ActFnType.SiLU, False, False, False, False, False, False, False, False, False, False],
    # gemma3_27b - NONE quant
    [2, 1, 1, 5376, 336, nl.bfloat16, None, QuantizationType.NONE, 0, NormType.RMS_NORM, False, False, ActFnType.GELU_Tanh_Approx, False, False, False, False, False, False, False, False, False, False],
    [2, 16, 1, 5376, 336, nl.bfloat16, None, QuantizationType.NONE, 0, NormType.RMS_NORM, False, False, ActFnType.GELU_Tanh_Approx, False, False, False, False, False, False, False, False, False, False],
    # gemma3_27b - ROW quant
    [2, 1, 1, 5376, 336, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 0, NormType.RMS_NORM, False, False, ActFnType.GELU_Tanh_Approx, False, False, False, False, False, False, False, False, False, False],
    [2, 16, 1, 5376, 336, nl.bfloat16, nl.float8_e4m3, QuantizationType.ROW, 0, NormType.RMS_NORM, False, False, ActFnType.GELU_Tanh_Approx, False, False, False, False, False, False, False, False, False, False],
]
# fmt: on

# Combined raw vectors for all TKG unit tests
_ALL_TKG_UNIT_RAW_VECTORS = (
    nki_tkg_fused_norm_mlp_kernel_spmd_vnc2_params
    + nki_tkg_fused_norm_mlp_kernel_spmd_vnc1_params
    + nki_tkg_fused_norm_mlp_kernel_spmd_vnc2_swap_perms
    + nki_tkg_fused_norm_mlp_kernel_spmd_skip_gate
    + nki_tkg_fused_norm_mlp_row_quant_kernel_params
    + nki_tkg_fused_norm_mlp_row_quant_kernel_layout_swap_perms
    + nki_tkg_fused_norm_mlp_static_quant_kernel_params
    + nki_tkg_fused_norm_mlp_static_quant_kernel_layout_swap_perms
    + nki_tkg_fused_norm_mlp_mx_quant_kernel_params
    + nki_tkg_fused_norm_mlp_static_mx_quant_kernel_params
    + nki_tkg_fused_norm_mlp_row_mx_quant_kernel_params
    + nki_tkg_transposed_io_params
    + nki_tkg_transposed_io_baseline_params
)


# Dedup ignoring tpbSgCyclesSum (index 8)
_ALL_TKG_UNIT_VECTORS_WITH_MODELS = dedup_test_vectors(_ALL_TKG_UNIT_RAW_VECTORS, ignore_indices={8})

# Separation pass vectors
_TKG_SEPARATION_PASS_RAW_VECTORS = nki_tkg_fused_norm_mlp_kernel_separation_pass_params


# ============================================================================
# TKG BF16 Sweep dimension values
# Derived from legacy sweep configs using RangeMonotonicGeneratorStrategy
# ============================================================================

# Main sweep (mlp_tkg_sweep_config):
# Trimmed from original ranges to keep sweep count under ~200 per test method.
# Retains boundary values and representative model sizes.
_TKG_SWEEP_BATCH = [1, 32, 48]
_TKG_SWEEP_SEQLEN = [1, 2]
_TKG_SWEEP_HIDDEN = [128, 512, 1024, 4096, 8192, 16384, 32768]
_TKG_SWEEP_INTERMEDIATE = [128, 512, 1024]

# I_non_multiple_of_128 sweep (mlp_tkg_I_non_multiple_of_128_sweep_config):
_TKG_I_NON_MULT_BATCH = [1, 2, 4, 8, 16]
_TKG_I_NON_MULT_SEQLEN = [1]
_TKG_I_NON_MULT_HIDDEN = [8192, 16384]
_TKG_I_NON_MULT_INTERMEDIATE = [412, 896, 1792, 2204, 3584, 4892, 5120]

# store_add_false sweep (mlp_tkg_store_add_false_sweep_config):
# All dimensions fixed: batch=4, seqlen=1, hidden=8192, intermediate=448
_TKG_STORE_ADD_FALSE_BATCH = [4]
_TKG_STORE_ADD_FALSE_SEQLEN = [1]
_TKG_STORE_ADD_FALSE_HIDDEN = [8192]
_TKG_STORE_ADD_FALSE_INTERMEDIATE = [448]

# basic sweep / feature_test (mlp_tkg_basic_sweep_config):
# All dimensions fixed: batch=4, seqlen=1, hidden=8192, intermediate=416
_TKG_BASIC_BATCH = [4]
_TKG_BASIC_SEQLEN = [1]
_TKG_BASIC_HIDDEN = [8192]
_TKG_BASIC_INTERMEDIATE = [416]

# Feature configs for main sweep and I_non_multiple cross-product (5 configs)
_TKG_SWEEP_5_FEATURE_CONFIGS = [
    ("column_tiling_basic", COLUMN_TILING_BASIC_CONFIG),
    ("column_tiling_full_features_rmsnorm", COLUMN_TILING_FULL_FEATURE_RMSNORM_CONFIG),
    ("column_tiling_full_features_layernorm", COLUMN_TILING_FULL_FEATURE_LAYERNORM_CONFIG),
    ("non_column_tiling_basic", NON_COLUMN_TILING_BASIC_CONFIG),
    ("non_column_tiling_full_features", NON_COLUMN_TILING_FULL_FEATURE_CONFIG),
]

# Feature configs for FP8 sweep cross-product (4 configs — no layernorm)
_TKG_SWEEP_4_FEATURE_CONFIGS = [
    ("column_tiling_basic", COLUMN_TILING_BASIC_CONFIG),
    ("column_tiling_full_features_rmsnorm", COLUMN_TILING_FULL_FEATURE_RMSNORM_CONFIG),
    ("non_column_tiling_basic", NON_COLUMN_TILING_BASIC_CONFIG),
    ("non_column_tiling_full_features", NON_COLUMN_TILING_FULL_FEATURE_CONFIG),
]

# Feature configs for store_add_false sweep cross-product (3 configs)
_TKG_STORE_ADD_FALSE_FEATURE_CONFIGS = [
    ("column_tiling_basic", COLUMN_TILING_BASIC_CONFIG),
    ("column_tiling_full_features_rmsnorm", COLUMN_TILING_FULL_FEATURE_RMSNORM_CONFIG),
    ("column_tiling_full_features_layernorm", COLUMN_TILING_FULL_FEATURE_LAYERNORM_CONFIG),
]

# Feature combos for basic sweep / feature_test (6 combos)
_TKG_FEATURE_TEST_COMBOS = [
    (True, True, True, False),
    (False, True, True, False),
    (True, True, True, True),
    (False, True, True, True),
    (True, True, False, True),
    (False, True, False, True),
]


def _tkg_sweep_filter(
    batch: int,
    seqlen: int,
    hidden: int,
    intermediate: int,
) -> FilterResult:
    """Filter function for TKG BF16 sweep dimension combinations.

    Encodes the legacy negative test logic from run_range_mlp_tkg_test.
    Feature configs are cross-producted separately via @pytest.mark.parametrize,
    so the filter only receives dimension params.

    The lnc_degree is always 2 (CompilerArgs default for TRN2).
    use_tkg_down_proj_column_tiling depends on the config, so psum bank limit
    checks that depend on it are handled inside the test method.
    """
    lnc_degree = 2  # CompilerArgs default for TRN2

    # BxS must not exceed 128 partitions
    if batch * seqlen > 128:
        return FilterResult.INVALID

    # H1 must be evenly divisible by lnc_degree for LNC2 sharding
    H1 = hidden // 128
    if H1 % lnc_degree != 0:
        return FilterResult.INVALID

    # Hidden for each core must be divisible by 128
    if hidden // lnc_degree % 128 != 0:
        return FilterResult.INVALID

    return FilterResult.VALID


def _mlp_tkg_sbuf_wrapper_kernel(
    hidden_tensor: nl.ndarray,
    gate_proj_weights_tensor: nl.ndarray,
    up_proj_weights_tensor: nl.ndarray,
    down_proj_weights_tensor: nl.ndarray,
    normalization_weights_tensor=None,
    gate_proj_bias_tensor=None,
    up_proj_bias_tensor=None,
    down_proj_bias_tensor=None,
    normalization_bias_tensor=None,
    fused_add_tensor=None,
    store_fused_add_result: bool = False,
    activation_fn: ActFnType = ActFnType.SiLU,
    normalization_type: NormType = NormType.NO_NORM,
    quantization_type: QuantizationType = QuantizationType.NONE,
    gate_w_scale=None,
    up_w_scale=None,
    down_w_scale=None,
    gate_up_in_scale=None,
    down_in_scale=None,
    quant_clipping_bound: float = 0.0,
    output_dtype=None,
    store_output_in_sbuf: bool = False,
    eps: float = 1e-6,
    skip_gate_proj: bool = False,
    use_tkg_gate_up_proj_column_tiling: bool = True,
    use_tkg_down_proj_column_tiling: bool = True,
    use_tkg_down_proj_optimized_layout: bool = False,
    gate_clamp_upper_limit=None,
    gate_clamp_lower_limit=None,
    up_clamp_upper_limit=None,
    up_clamp_lower_limit=None,
    force_cte_mode: bool = False,
    mode: ComputationMode = ComputationMode.DECODE,
    sbm=None,
    mx_dummy_scale_hbm=None,
    transposed_in: bool = False,
    transposed_out: bool = False,
) -> list[nl.ndarray]:
    """Wrapper for testing SBUF input or SBUF output paths.

    When store_output_in_sbuf=False: loads HBM input into SBUF, calls mlp with SBUF input.
    When store_output_in_sbuf=True: calls mlp with HBM input, copies SBUF output back to HBM.
    """
    if not store_output_in_sbuf:
        hidden_tensor, sbm = setup_sbuf_input(hidden_tensor)

    results = mlp_kernel(
        hidden_tensor=hidden_tensor,
        gate_proj_weights_tensor=gate_proj_weights_tensor,
        up_proj_weights_tensor=up_proj_weights_tensor,
        down_proj_weights_tensor=down_proj_weights_tensor,
        normalization_weights_tensor=normalization_weights_tensor,
        gate_proj_bias_tensor=gate_proj_bias_tensor,
        up_proj_bias_tensor=up_proj_bias_tensor,
        down_proj_bias_tensor=down_proj_bias_tensor,
        normalization_bias_tensor=normalization_bias_tensor,
        fused_add_tensor=fused_add_tensor,
        store_fused_add_result=store_fused_add_result,
        activation_fn=activation_fn,
        normalization_type=normalization_type,
        quantization_type=quantization_type,
        gate_w_scale=gate_w_scale,
        up_w_scale=up_w_scale,
        down_w_scale=down_w_scale,
        gate_up_in_scale=gate_up_in_scale,
        down_in_scale=down_in_scale,
        quant_clipping_bound=quant_clipping_bound,
        output_dtype=output_dtype,
        store_output_in_sbuf=store_output_in_sbuf,
        eps=eps,
        skip_gate_proj=skip_gate_proj,
        use_tkg_gate_up_proj_column_tiling=use_tkg_gate_up_proj_column_tiling,
        use_tkg_down_proj_column_tiling=use_tkg_down_proj_column_tiling,
        use_tkg_down_proj_optimized_layout=use_tkg_down_proj_optimized_layout,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        force_cte_mode=force_cte_mode,
        mode=mode,
        sbm=sbm,
        mx_dummy_scale_hbm=mx_dummy_scale_hbm,
        transposed_in=transposed_in,
        transposed_out=transposed_out,
    )

    if store_output_in_sbuf:
        return copy_sbuf_output_to_hbm(results[0], hidden_tensor, down_proj_weights_tensor)
    return results


@pytest_test_metadata(
    name="MLP TKG",
    tags=["model"],
)
@pytest_marks(["mlp", "tkg", "mx"])
@final
class TestMlpTkgKernel:
    def _kernel_input_generator(self, vec_dict):
        """Generate kernel inputs from a parsed TKG vector dict."""
        d = vec_dict
        lnc_degree = d["vnc_degree"]
        quant_type = d["quant_type"]
        use_tkg_down_proj_optimized_layout = d["use_tkg_down_proj_optimized_layout"]

        # Select tensor generator based on quant type and layout
        if quant_type == QuantizationType.NONE:
            if use_tkg_down_proj_optimized_layout:
                tensor_generator = gaussian_tensor_generator(
                    0,
                    241,
                    modifier_fn=modify_down_proj_lhs_rhs_swap_unit_stride_layout,
                    lnc=lnc_degree,
                )
            else:
                tensor_generator = gaussian_tensor_generator()
        elif quant_type in (
            QuantizationType.ROW,
            QuantizationType.STATIC,
            QuantizationType.STATIC_MX,
            QuantizationType.ROW_MX,
        ):
            if use_tkg_down_proj_optimized_layout:
                tensor_generator = random_lhs_and_random_bound_weight_tensor_generator(
                    0,
                    241,
                    modifier_fn=modify_down_proj_lhs_rhs_swap_unit_stride_layout,
                    lnc=lnc_degree,
                )
            else:
                tensor_generator = random_lhs_and_random_bound_weight_tensor_generator(0, 241)
        elif quant_type == QuantizationType.MX:
            # MX quant uses random_lhs_and_random_bound_weight_tensor_generator
            # (same as legacy run_mlp_tkg_test when quant_dtype is not None)
            if use_tkg_down_proj_optimized_layout:
                tensor_generator = random_lhs_and_random_bound_weight_tensor_generator(
                    0,
                    241,
                    modifier_fn=modify_down_proj_lhs_rhs_swap_unit_stride_layout,
                    lnc=lnc_degree,
                )
            else:
                tensor_generator = random_lhs_and_random_bound_weight_tensor_generator(0, 241)
        else:
            tensor_generator = gaussian_tensor_generator()

        kernel_input = build_fused_norm_mlp(
            batch=d["batch"],
            seqlen=d["seqlen"],
            hidden=d["hidden"],
            intermediate=d["intermediate"],
            dtype=d["dtype"],
            quantization_type=quant_type,
            quant_dtype=d["quant_dtype"],
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
            use_tkg_gate_up_proj_column_tiling=d["use_tkg_gate_up_proj_column_tiling"],
            use_tkg_down_proj_column_tiling=d["use_tkg_down_proj_column_tiling"],
            use_tkg_down_proj_optimized_layout=use_tkg_down_proj_optimized_layout,
            transposed_in=d.get("transposed_in", False),
            transposed_out=d.get("transposed_out", False),
            tensor_generator=tensor_generator,
            mode=ComputationMode.DECODE,
        )
        # Add missing params that mlp() kernel accepts but build_fused_norm_mlp doesn't produce
        kernel_input["quant_clipping_bound"] = 0.0
        kernel_input["force_cte_mode"] = False
        kernel_input["mode"] = ComputationMode.DECODE
        kernel_input["sbm"] = None
        return kernel_input

    def _run_mlp_tkg_test(
        self,
        test_manager,
        vec_dict,
        platform_target,
        compiler_args=None,
        rtol=2e-2,
        atol=1e-5,
        is_negative_test=False,
    ):
        """Run an MLP TKG unit test: builds kernel_input from vec_dict, then delegates to run_mlp_test."""
        lnc = vec_dict["vnc_degree"] if isinstance(vec_dict, dict) else compiler_args.logical_nc_config

        if compiler_args is None:
            compiler_args = CompilerArgs(logical_nc_config=lnc, platform_target=platform_target)

        _run_mlp_test(
            test_manager=test_manager,
            kernel_input=self._kernel_input_generator(vec_dict),
            compiler_args=compiler_args,
            output_tensor_descriptor=mlp_output_tensor_descriptor,
            rtol=rtol,
            atol=atol,
            is_negative_test=is_negative_test,
            inference_args=TKG_INFERENCE_ARGS,
        )

    def _build_and_run_tkg_sweep(
        self,
        test_manager,
        batch,
        seqlen,
        hidden,
        intermediate,
        config,
        is_negative_test_case,
        platform_target,
        quant_type=QuantizationType.NONE,
        quant_dtype=None,
        rtol=2e-2,
        fused_add_override=None,
        store_add_override=None,
        bias_override=None,
        skip_gate=False,
        gate_clamp_upper_limit=None,
        gate_clamp_lower_limit=None,
        up_clamp_upper_limit=None,
        up_clamp_lower_limit=None,
    ):
        """Shared helper for all TKG sweep tests.

        Handles psum bank limit check, optimized layout skip guard, tensor generator
        selection, kernel input construction, and test execution.

        Args:
            config: Feature config dict with norm_type, fused_add, store_add, etc.
            fused_add_override/store_add_override/bias_override: If not None, override
                the corresponding config values (used by store_add_false sweep).
        """
        compiler_args = CompilerArgs(platform_target=platform_target)
        lnc_degree = compiler_args.logical_nc_config

        use_tkg_down_proj_column_tiling = config["use_tkg_down_proj_column_tiling"]

        # Psum bank limit (when use_tkg_down_proj_column_tiling=False)
        if not use_tkg_down_proj_column_tiling:
            T = batch * seqlen
            H1_shard = hidden // 128 // lnc_degree
            perBankT = 512 // T if T > 0 else 0
            if perBankT > 0:
                num_required_down_psum_banks = math.ceil(H1_shard / perBankT)
                if num_required_down_psum_banks > 8:
                    is_negative_test_case = True

        # Resolve config values with optional overrides
        fused_add = fused_add_override if fused_add_override is not None else config["fused_add"]
        store_add = store_add_override if store_add_override is not None else config["store_add"]
        gate_bias = bias_override if bias_override is not None else config["gate_bias"]
        up_bias = bias_override if bias_override is not None else config["up_bias"]
        down_bias = bias_override if bias_override is not None else config["down_bias"]
        norm_type = config["norm_type"]
        act_fn_type = config["act_fn_type"]
        norm_bias = config["norm_bias"]
        use_tkg_gate_up_proj_column_tiling = config["use_tkg_gate_up_proj_column_tiling"]
        use_tkg_down_proj_optimized_layout = config["use_tkg_down_proj_optimized_layout"]

        # Optimized layout requires H//(128*lnc) > 0
        if use_tkg_down_proj_optimized_layout and hidden // (128 * lnc_degree) == 0:
            pytest.skip(f"hidden={hidden} too small for optimized layout with lnc={lnc_degree}")

        # Select tensor generator based on quant type and layout
        if quant_type in (QuantizationType.ROW, QuantizationType.STATIC):
            tensor_generator = random_lhs_and_random_bound_weight_tensor_generator(0, 241)
        elif use_tkg_down_proj_optimized_layout:
            tensor_generator = gaussian_tensor_generator(
                0,
                241,
                modifier_fn=modify_down_proj_lhs_rhs_swap_unit_stride_layout,
                lnc=lnc_degree,
            )
        else:
            tensor_generator = gaussian_tensor_generator()

        kernel_input = build_fused_norm_mlp(
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            intermediate=intermediate,
            dtype=nl.bfloat16,
            quantization_type=quant_type,
            quant_dtype=quant_dtype,
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
            use_tkg_gate_up_proj_column_tiling=use_tkg_gate_up_proj_column_tiling,
            use_tkg_down_proj_column_tiling=use_tkg_down_proj_column_tiling,
            use_tkg_down_proj_optimized_layout=use_tkg_down_proj_optimized_layout,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            tensor_generator=tensor_generator,
            mode=ComputationMode.DECODE,
        )
        kernel_input["quant_clipping_bound"] = 0.0
        kernel_input["force_cte_mode"] = False
        kernel_input["mode"] = ComputationMode.DECODE
        kernel_input["sbm"] = None

        _run_mlp_test(
            test_manager,
            kernel_input,
            compiler_args,
            mlp_output_tensor_descriptor,
            rtol=rtol,
            is_negative_test=is_negative_test_case,
        )

    def _run_validated_mlp_tkg(
        self,
        test_manager,
        platform_target,
        vnc_degree,
        batch,
        seqlen,
        hidden,
        intermediate,
        dtype,
        quant_dtype,
        quant_type,
        norm_type,
        fused_add,
        store_add,
        act_fn_type,
        skip_gate,
        gate_bias,
        up_bias,
        down_bias,
        norm_bias,
        use_tkg_gate_up_proj_column_tiling,
        use_tkg_down_proj_column_tiling,
        use_tkg_down_proj_optimized_layout,
        is_model_config=False,
        transposed_in=False,
        transposed_out=False,
    ):
        """Shared validation, rtol selection, and execution for TKG unit and model tests."""
        # MX and STATIC_MX quant are only supported on TRN3
        if (
            quant_type in (QuantizationType.MX, QuantizationType.STATIC_MX, QuantizationType.ROW_MX)
            and not platform_target.is_trn3()
        ):
            pytest.skip("MX/STATIC_MX/ROW_MX Quantization is only supported on TRN3.")

        # MX quantization requires H divisible by 512 (alignment for quantization groups: 128 * 4)
        if (
            quant_type in (QuantizationType.MX, QuantizationType.STATIC_MX, QuantizationType.ROW_MX)
            and hidden % 512 != 0
        ):
            pytest.skip("MX quantization requires H to be divisible by 512")

        # MX quantization requires I % 512 == 0 or (I < 512 and I % 32 == 0)
        if quant_type == QuantizationType.MX and not (
            intermediate % 512 == 0 or (intermediate < 512 and intermediate % 32 == 0)
        ):
            pytest.skip("MX quantization requires I to be I % 512 == 0 or (I < 512 and I % 32 ==0)")

        # MX quant kernels do not support BxS tiling (T > 128) yet
        if quant_type in (QuantizationType.MX,) and batch * seqlen > 128:
            pytest.skip("MX quant does not support T > 128 (BxS tiling) yet")

        # --- Negative test checks (from legacy run_range_mlp_tkg_test) ---
        is_negative_test = False

        # Psum bank limit when use_tkg_down_proj_column_tiling is False
        if not use_tkg_down_proj_column_tiling and not is_model_config:
            T = batch * seqlen
            H1_shard = hidden // 128 // vnc_degree
            perBankT = 512 // T if T > 0 else 0
            if perBankT > 0:
                num_required_down_psum_banks = math.ceil(H1_shard / perBankT)
                if num_required_down_psum_banks > 8:
                    is_negative_test = True

        # Determine rtol based on quant type
        if quant_type == QuantizationType.ROW:
            rtol = 4e-2
        elif quant_type == QuantizationType.STATIC:
            rtol = 3e-2
        elif quant_type in (QuantizationType.MX, QuantizationType.STATIC_MX, QuantizationType.ROW_MX):
            rtol = 5e-2
        else:
            rtol = 2e-2  # NONE quant default

        vec_dict = dict(
            vnc_degree=vnc_degree,
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            intermediate=intermediate,
            dtype=dtype,
            quant_dtype=quant_dtype,
            quant_type=quant_type,
            norm_type=norm_type,
            fused_add=fused_add,
            store_add=store_add,
            act_fn_type=act_fn_type,
            skip_gate=skip_gate,
            gate_bias=gate_bias,
            up_bias=up_bias,
            down_bias=down_bias,
            norm_bias=norm_bias,
            use_tkg_gate_up_proj_column_tiling=use_tkg_gate_up_proj_column_tiling,
            use_tkg_down_proj_column_tiling=use_tkg_down_proj_column_tiling,
            use_tkg_down_proj_optimized_layout=use_tkg_down_proj_optimized_layout,
            transposed_in=transposed_in,
            transposed_out=transposed_out,
            mode=ComputationMode.DECODE,
        )
        self._run_mlp_tkg_test(
            test_manager=test_manager,
            vec_dict=vec_dict,
            platform_target=platform_target,
            rtol=rtol,
            is_negative_test=is_negative_test,
        )

    @pytest.mark.fast
    @pytest_parametrize(TKG_UNIT_PARAM_NAMES, _ALL_TKG_UNIT_VECTORS_WITH_MODELS, abbrevs=_TKG_ABBREVS)
    def test_mlp_tkg_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden,
        intermediate,
        dtype,
        quant_dtype,
        quant_type,
        tpbSgCyclesSum,
        norm_type,
        fused_add,
        store_add,
        act_fn_type,
        skip_gate,
        gate_bias,
        up_bias,
        down_bias,
        norm_bias,
        use_tkg_gate_up_proj_column_tiling,
        use_tkg_down_proj_column_tiling,
        use_tkg_down_proj_optimized_layout,
        transposed_in,
        transposed_out,
    ):
        self._run_validated_mlp_tkg(
            test_manager=test_manager,
            platform_target=platform_target,
            vnc_degree=vnc_degree,
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            intermediate=intermediate,
            dtype=dtype,
            quant_dtype=quant_dtype,
            quant_type=quant_type,
            norm_type=norm_type,
            fused_add=fused_add,
            store_add=store_add,
            act_fn_type=act_fn_type,
            skip_gate=skip_gate,
            gate_bias=gate_bias,
            up_bias=up_bias,
            down_bias=down_bias,
            norm_bias=norm_bias,
            use_tkg_gate_up_proj_column_tiling=use_tkg_gate_up_proj_column_tiling,
            use_tkg_down_proj_column_tiling=use_tkg_down_proj_column_tiling,
            use_tkg_down_proj_optimized_layout=use_tkg_down_proj_optimized_layout,
            transposed_in=transposed_in,
            transposed_out=transposed_out,
            is_model_config=(tpbSgCyclesSum == 0),
        )

    @pytest_parametrize(TKG_UNIT_PARAM_NAMES, _TKG_SEPARATION_PASS_RAW_VECTORS, abbrevs=_TKG_ABBREVS)
    def test_mlp_tkg_separation_pass(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden,
        intermediate,
        dtype,
        quant_dtype,
        quant_type,
        tpbSgCyclesSum,
        norm_type,
        fused_add,
        store_add,
        act_fn_type,
        skip_gate,
        gate_bias,
        up_bias,
        down_bias,
        norm_bias,
        use_tkg_gate_up_proj_column_tiling,
        use_tkg_down_proj_column_tiling,
        use_tkg_down_proj_optimized_layout,
        transposed_in,
        transposed_out,
    ):
        vec_dict = dict(
            vnc_degree=vnc_degree,
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            intermediate=intermediate,
            dtype=dtype,
            quant_dtype=quant_dtype,
            quant_type=quant_type,
            norm_type=norm_type,
            fused_add=fused_add,
            store_add=store_add,
            act_fn_type=act_fn_type,
            skip_gate=skip_gate,
            gate_bias=gate_bias,
            up_bias=up_bias,
            down_bias=down_bias,
            norm_bias=norm_bias,
            use_tkg_gate_up_proj_column_tiling=use_tkg_gate_up_proj_column_tiling,
            use_tkg_down_proj_column_tiling=use_tkg_down_proj_column_tiling,
            use_tkg_down_proj_optimized_layout=use_tkg_down_proj_optimized_layout,
            transposed_in=transposed_in,
            transposed_out=transposed_out,
        )
        compiler_args = CompilerArgs(
            logical_nc_config=vnc_degree,
            platform_target=platform_target,
            separation_pass_mode=SeparationPassMode.INDIRECT,
        )
        self._run_mlp_tkg_test(
            test_manager=test_manager,
            vec_dict=vec_dict,
            platform_target=platform_target,
            compiler_args=compiler_args,
        )

    # ============================================================================
    # TKG BF16 Sweep Tests
    # ============================================================================

    # @IGNORE_FAST
    @pytest.mark.coverage_parametrize(
        batch=_TKG_SWEEP_BATCH,
        seqlen=_TKG_SWEEP_SEQLEN,
        hidden=_TKG_SWEEP_HIDDEN,
        intermediate=_TKG_SWEEP_INTERMEDIATE,
        filter=_tkg_sweep_filter,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    @pytest.mark.parametrize(
        "config_name,config",
        _TKG_SWEEP_5_FEATURE_CONFIGS,
    )
    def test_mlp_tkg_kernel_sweep(
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
        # NKILIB-848: SBUF address dependency issue with use_tkg_down_proj_optimized_layout=True
        if (
            batch == 1
            and seqlen == 1
            and hidden == 4096
            and intermediate == 128
            and config_name == "non_column_tiling_full_features"
        ):
            pytest.xfail("NKILIB-848: non-determinism from SBUF address dependency at b=1,s=1,h=4096,i=128")

        self._build_and_run_tkg_sweep(
            test_manager,
            batch,
            seqlen,
            hidden,
            intermediate,
            config,
            is_negative_test_case,
            platform_target,
        )

    # @IGNORE_FAST
    @pytest.mark.coverage_parametrize(
        batch=_TKG_I_NON_MULT_BATCH,
        seqlen=_TKG_I_NON_MULT_SEQLEN,
        hidden=_TKG_I_NON_MULT_HIDDEN,
        intermediate=_TKG_I_NON_MULT_INTERMEDIATE,
        filter=_tkg_sweep_filter,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    @pytest.mark.parametrize(
        "config_name,config",
        _TKG_SWEEP_5_FEATURE_CONFIGS,
    )
    def test_mlp_tkg_kernel_sweep_I_non_multiple_of_128(
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
        self._build_and_run_tkg_sweep(
            test_manager,
            batch,
            seqlen,
            hidden,
            intermediate,
            config,
            is_negative_test_case,
            platform_target,
        )

    # @IGNORE_FAST
    @pytest.mark.coverage_parametrize(
        batch=_TKG_STORE_ADD_FALSE_BATCH,
        seqlen=_TKG_STORE_ADD_FALSE_SEQLEN,
        hidden=_TKG_STORE_ADD_FALSE_HIDDEN,
        intermediate=_TKG_STORE_ADD_FALSE_INTERMEDIATE,
        filter=_tkg_sweep_filter,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    @pytest.mark.parametrize(
        "config_name,config",
        _TKG_STORE_ADD_FALSE_FEATURE_CONFIGS,
    )
    def test_mlp_tkg_kernel_sweep_store_add_false(
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
        self._build_and_run_tkg_sweep(
            test_manager,
            batch,
            seqlen,
            hidden,
            intermediate,
            config,
            is_negative_test_case,
            platform_target,
            fused_add_override=True,
            store_add_override=False,
            bias_override=False,
        )

    # @IGNORE_FAST
    @pytest.mark.coverage_parametrize(
        batch=_TKG_BASIC_BATCH,
        seqlen=_TKG_BASIC_SEQLEN,
        hidden=_TKG_BASIC_HIDDEN,
        intermediate=_TKG_BASIC_INTERMEDIATE,
        filter=_tkg_sweep_filter,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    @pytest.mark.parametrize(
        "use_tkg_gate_up_proj_column_tiling,use_tkg_down_proj_column_tiling,skip_gate_proj,clamp",
        _TKG_FEATURE_TEST_COMBOS,
    )
    def test_mlp_tkg_kernel_sweep_feature_test(
        self,
        test_manager: Orchestrator,
        batch: int,
        seqlen: int,
        hidden: int,
        intermediate: int,
        use_tkg_gate_up_proj_column_tiling: bool,
        use_tkg_down_proj_column_tiling: bool,
        skip_gate_proj: bool,
        clamp: bool,
        is_negative_test_case: bool,
        platform_target: Platforms,
    ):
        gate_clamp_upper_limit = float(8.0) if clamp else None
        gate_clamp_lower_limit = float(-6.0) if clamp else None
        up_clamp_upper_limit = float(8.0) if clamp else None
        up_clamp_lower_limit = float(-6.0) if clamp else None

        config = {
            "norm_type": NormType.NO_NORM,
            "fused_add": False,
            "store_add": False,
            "act_fn_type": ActFnType.SiLU,
            "gate_bias": False,
            "up_bias": False,
            "down_bias": False,
            "norm_bias": False,
            "use_tkg_gate_up_proj_column_tiling": use_tkg_gate_up_proj_column_tiling,
            "use_tkg_down_proj_column_tiling": use_tkg_down_proj_column_tiling,
            "use_tkg_down_proj_optimized_layout": False,
        }
        self._build_and_run_tkg_sweep(
            test_manager,
            batch,
            seqlen,
            hidden,
            intermediate,
            config,
            is_negative_test_case,
            platform_target,
            skip_gate=skip_gate_proj,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
        )

    # ============================================================================
    # TKG FP8 ROW Sweep Tests
    # ============================================================================

    # @IGNORE_FAST
    @pytest.mark.coverage_parametrize(
        batch=_TKG_SWEEP_BATCH,
        seqlen=_TKG_SWEEP_SEQLEN,
        hidden=_TKG_SWEEP_HIDDEN,
        intermediate=_TKG_SWEEP_INTERMEDIATE,
        filter=_tkg_sweep_filter,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    @pytest.mark.parametrize(
        "config_name,config",
        _TKG_SWEEP_4_FEATURE_CONFIGS,
    )
    def test_mlp_tkg_kernel_sweep_fp8_row_quant(
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
        # Pre-existing accuracy issue on this shape — not caused by UTF migration.
        # This test vector is new (not present on mainline) and fails validation consistently.
        # NKILIB-848: https://aws-neuron.atlassian.net/browse/NKILIB-848
        if (
            batch == 1
            and seqlen == 1
            and hidden == 4096
            and intermediate == 128
            and config_name == "non_column_tiling_full_features"
        ):
            pytest.xfail("Pre-existing FP8 row quant accuracy issue at b=1,s=1,h=4096,i=128 non-column-tiling")

        self._build_and_run_tkg_sweep(
            test_manager,
            batch,
            seqlen,
            hidden,
            intermediate,
            config,
            is_negative_test_case,
            platform_target,
            quant_type=QuantizationType.ROW,
            quant_dtype=nl.float8_e4m3,
            rtol=4e-2,
        )

    # @IGNORE_FAST
    @pytest.mark.coverage_parametrize(
        batch=_TKG_I_NON_MULT_BATCH,
        seqlen=_TKG_I_NON_MULT_SEQLEN,
        hidden=_TKG_I_NON_MULT_HIDDEN,
        intermediate=_TKG_I_NON_MULT_INTERMEDIATE,
        filter=_tkg_sweep_filter,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    @pytest.mark.parametrize(
        "config_name,config",
        _TKG_SWEEP_4_FEATURE_CONFIGS,
    )
    def test_mlp_tkg_kernel_sweep_I_non_multiple_of_128_fp8_row_quant(
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
        self._build_and_run_tkg_sweep(
            test_manager,
            batch,
            seqlen,
            hidden,
            intermediate,
            config,
            is_negative_test_case,
            platform_target,
            quant_type=QuantizationType.ROW,
            quant_dtype=nl.float8_e4m3,
            rtol=4e-2,
        )

    # ============================================================================
    # TKG FP8 STATIC Sweep Tests
    # ============================================================================

    # @IGNORE_FAST
    @pytest.mark.coverage_parametrize(
        batch=_TKG_SWEEP_BATCH,
        seqlen=_TKG_SWEEP_SEQLEN,
        hidden=_TKG_SWEEP_HIDDEN,
        intermediate=_TKG_SWEEP_INTERMEDIATE,
        filter=_tkg_sweep_filter,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    @pytest.mark.parametrize(
        "config_name,config",
        _TKG_SWEEP_4_FEATURE_CONFIGS,
    )
    def test_mlp_tkg_kernel_sweep_fp8_static_quant(
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
        # NKILIB-848: SBUF address dependency issue with use_tkg_down_proj_optimized_layout=True
        if (
            batch == 1
            and seqlen == 1
            and hidden == 4096
            and intermediate == 128
            and config_name == "non_column_tiling_full_features"
        ):
            pytest.xfail("NKILIB-848: non-determinism from SBUF address dependency at b=1,s=1,h=4096,i=128")

        self._build_and_run_tkg_sweep(
            test_manager,
            batch,
            seqlen,
            hidden,
            intermediate,
            config,
            is_negative_test_case,
            platform_target,
            quant_type=QuantizationType.STATIC,
            quant_dtype=nl.float8_e4m3,
            rtol=3e-2,
        )

    # @IGNORE_FAST
    @pytest.mark.coverage_parametrize(
        batch=_TKG_I_NON_MULT_BATCH,
        seqlen=_TKG_I_NON_MULT_SEQLEN,
        hidden=_TKG_I_NON_MULT_HIDDEN,
        intermediate=_TKG_I_NON_MULT_INTERMEDIATE,
        filter=_tkg_sweep_filter,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    @pytest.mark.parametrize(
        "config_name,config",
        _TKG_SWEEP_4_FEATURE_CONFIGS,
    )
    def test_mlp_tkg_kernel_sweep_I_non_multiple_of_128_fp8_static_quant(
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
        self._build_and_run_tkg_sweep(
            test_manager,
            batch,
            seqlen,
            hidden,
            intermediate,
            config,
            is_negative_test_case,
            platform_target,
            quant_type=QuantizationType.STATIC,
            quant_dtype=nl.float8_e4m3,
            rtol=3e-2,
        )

    # ============================================================================
    # TKG FP8 STATIC_MX Sweep Tests
    # ============================================================================

    @pytest_parametrize(
        TKG_UNIT_PARAM_NAMES, nki_tkg_fused_norm_mlp_static_mx_quant_kernel_params, abbrevs=_TKG_ABBREVS
    )
    def test_mlp_tkg_kernel_sweep_fp8_static_mx_quant(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden,
        intermediate,
        dtype,
        quant_dtype,
        quant_type,
        tpbSgCyclesSum,
        norm_type,
        fused_add,
        store_add,
        act_fn_type,
        skip_gate,
        gate_bias,
        up_bias,
        down_bias,
        norm_bias,
        use_tkg_gate_up_proj_column_tiling,
        use_tkg_down_proj_column_tiling,
        use_tkg_down_proj_optimized_layout,
        transposed_in,
        transposed_out,
    ):
        if not platform_target.is_trn3():
            pytest.skip("STATIC_MX uses MX matmul engine, only supported on TRN3.")

        vec_dict = dict(
            vnc_degree=vnc_degree,
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            intermediate=intermediate,
            dtype=dtype,
            quant_dtype=quant_dtype,
            quant_type=quant_type,
            norm_type=norm_type,
            fused_add=fused_add,
            store_add=store_add,
            act_fn_type=act_fn_type,
            skip_gate=skip_gate,
            gate_bias=gate_bias,
            up_bias=up_bias,
            down_bias=down_bias,
            norm_bias=norm_bias,
            use_tkg_gate_up_proj_column_tiling=use_tkg_gate_up_proj_column_tiling,
            use_tkg_down_proj_column_tiling=use_tkg_down_proj_column_tiling,
            use_tkg_down_proj_optimized_layout=use_tkg_down_proj_optimized_layout,
            transposed_in=transposed_in,
            transposed_out=transposed_out,
        )
        self._run_mlp_tkg_test(
            test_manager=test_manager,
            vec_dict=vec_dict,
            platform_target=platform_target,
            rtol=5e-2,
        )

    # ============================================================================
    # TKG FP8 ROW_MX Sweep Tests
    # ============================================================================

    @pytest_parametrize(TKG_UNIT_PARAM_NAMES, nki_tkg_fused_norm_mlp_row_mx_quant_kernel_params, abbrevs=_TKG_ABBREVS)
    def test_mlp_tkg_kernel_sweep_fp8_row_mx_quant(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden,
        intermediate,
        dtype,
        quant_dtype,
        quant_type,
        tpbSgCyclesSum,
        norm_type,
        fused_add,
        store_add,
        act_fn_type,
        skip_gate,
        gate_bias,
        up_bias,
        down_bias,
        norm_bias,
        use_tkg_gate_up_proj_column_tiling,
        use_tkg_down_proj_column_tiling,
        use_tkg_down_proj_optimized_layout,
        transposed_in,
        transposed_out,
    ):
        if not platform_target.is_trn3():
            pytest.skip("ROW_MX uses MX matmul engine, only supported on TRN3.")

        vec_dict = dict(
            vnc_degree=vnc_degree,
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            intermediate=intermediate,
            dtype=dtype,
            quant_dtype=quant_dtype,
            quant_type=quant_type,
            norm_type=norm_type,
            fused_add=fused_add,
            store_add=store_add,
            act_fn_type=act_fn_type,
            skip_gate=skip_gate,
            gate_bias=gate_bias,
            up_bias=up_bias,
            down_bias=down_bias,
            norm_bias=norm_bias,
            use_tkg_gate_up_proj_column_tiling=use_tkg_gate_up_proj_column_tiling,
            use_tkg_down_proj_column_tiling=use_tkg_down_proj_column_tiling,
            use_tkg_down_proj_optimized_layout=use_tkg_down_proj_optimized_layout,
            transposed_in=transposed_in,
            transposed_out=transposed_out,
        )
        self._run_mlp_tkg_test(
            test_manager=test_manager,
            vec_dict=vec_dict,
            platform_target=platform_target,
            rtol=5e-2,
        )

    # ============================================================================
    # TKG SBUF Input / Output Sweep Tests
    # ============================================================================

    _TKG_SBUF_SWEEP_BATCH = [1, 4, 16, 32, 128]
    _TKG_SBUF_SWEEP_SEQLEN = [1]
    _TKG_SBUF_SWEEP_HIDDEN = [512, 1024, 4096]
    _TKG_SBUF_SWEEP_INTERMEDIATE = [128, 512, 1024]

    _TKG_SBUF_SWEEP_FEATURE_CONFIGS = [
        (
            "sbuf_input",
            {
                "wrapper": "sbuf_input",
                "fused_add": False,
                "norm_type": NormType.RMS_NORM,
            },
        ),
        (
            "sbuf_output",
            {
                "wrapper": "sbuf_output",
                "fused_add": False,
                "norm_type": NormType.RMS_NORM,
            },
        ),
    ]

    @pytest.mark.coverage_parametrize(
        batch=_TKG_SBUF_SWEEP_BATCH,
        seqlen=_TKG_SBUF_SWEEP_SEQLEN,
        hidden=_TKG_SBUF_SWEEP_HIDDEN,
        intermediate=_TKG_SBUF_SWEEP_INTERMEDIATE,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    @pytest.mark.parametrize(
        "config_name,config",
        _TKG_SBUF_SWEEP_FEATURE_CONFIGS,
    )
    def test_mlp_tkg_kernel_sweep_sbuf(
        self,
        test_manager: Orchestrator,
        batch: int,
        seqlen: int,
        hidden: int,
        intermediate: int,
        config_name: str,
        config: dict,
        is_negative_test_case: bool,
        platform_target: Platforms,
    ):
        """Sweep test for MLP TKG SBUF input and SBUF output paths (column tiling)."""
        compiler_args = CompilerArgs(platform_target=platform_target)

        is_sbuf_input = config["wrapper"] == "sbuf_input"

        kernel_input = build_fused_norm_mlp(
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            intermediate=intermediate,
            dtype=nl.bfloat16,
            norm_type=config["norm_type"],
            fused_add=config["fused_add"],
            use_tkg_gate_up_proj_column_tiling=True,
            use_tkg_down_proj_column_tiling=True,
        )
        kernel_input["quant_clipping_bound"] = 0.0
        kernel_input["force_cte_mode"] = False
        kernel_input["mode"] = ComputationMode.DECODE
        kernel_input["sbm"] = None
        kernel_input["store_output_in_sbuf"] = not is_sbuf_input

        _run_mlp_test(
            test_manager=test_manager,
            kernel_input=kernel_input,
            compiler_args=compiler_args,
            output_tensor_descriptor=mlp_output_tensor_descriptor,
            is_negative_test=is_negative_test_case,
            inference_args=TKG_INFERENCE_ARGS,
            kernel_entry=_mlp_tkg_sbuf_wrapper_kernel,
        )


@pytest_marks(["mlp", "tkg", "model", "mx"])
@final
class TestMlpTkgModel:
    """Model regression tests for MLP TKG kernel.

    Separate test methods per tier for cleaner pytest discovery:
    - test_tier0: Critical model configs (high priority)
    - test_optimal: Optimal performance configs
    - test_generality: Generality/coverage configs
    """

    # Tier params resolved at class definition time (lazy loading would require conditional imports)
    _TIER0_PARAMS, _TIER0_IDS = (
        prepare_model_parametrize({ModelTestType.TIER0: mlp_tkg_model_configs.get(ModelTestType.TIER0, [])})
        if mlp_tkg_model_configs
        else ([], [])
    )
    _OPTIMAL_PARAMS, _OPTIMAL_IDS = (
        prepare_model_parametrize({ModelTestType.OPTIMAL: mlp_tkg_model_configs.get(ModelTestType.OPTIMAL, [])})
        if mlp_tkg_model_configs
        else ([], [])
    )
    _GENERALITY_PARAMS, _GENERALITY_IDS = (
        prepare_model_parametrize({ModelTestType.GENERALITY: mlp_tkg_model_configs.get(ModelTestType.GENERALITY, [])})
        if mlp_tkg_model_configs
        else ([], [])
    )

    def _run_model_test(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden,
        intermediate,
        dtype,
        quant_dtype,
        quant_type,
        tpbSgCyclesSum,
        norm_type,
        fused_add,
        store_add,
        act_fn_type,
        skip_gate,
        gate_bias,
        up_bias,
        down_bias,
        norm_bias,
        use_tkg_gate_up_proj_column_tiling,
        use_tkg_down_proj_column_tiling,
        use_tkg_down_proj_optimized_layout,
        transposed_in,
        transposed_out,
    ):
        q_dt = str(quant_dtype) if quant_dtype is not None else None
        metadata_key = {
            "vnc": vnc_degree,
            "b": batch,
            "s": seqlen,
            "h": hidden,
            "i": intermediate,
            "dt": str(dtype),
            "q_dt": q_dt,
            "q_t": quant_type,
            "norm_type": norm_type,
            "fa": fused_add,
            "sa": store_add,
            "act_fn_type": act_fn_type,
            "skip_gate": skip_gate,
            "gb": gate_bias,
            "ub": up_bias,
            "db": down_bias,
            "nb": norm_bias,
            "gate_col": use_tkg_gate_up_proj_column_tiling,
            "down_col": use_tkg_down_proj_column_tiling,
            "down_opt": use_tkg_down_proj_optimized_layout,
        }
        metadata_list = load_model_configs("test_mlp_tkg")
        collector.match_and_add_metadata_dimensions(metadata_key, metadata_list)
        TestMlpTkgKernel()._run_validated_mlp_tkg(
            test_manager=test_manager,
            platform_target=platform_target,
            vnc_degree=vnc_degree,
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            intermediate=intermediate,
            dtype=dtype,
            quant_dtype=quant_dtype,
            quant_type=quant_type,
            norm_type=norm_type,
            fused_add=fused_add,
            store_add=store_add,
            act_fn_type=act_fn_type,
            skip_gate=skip_gate,
            gate_bias=gate_bias,
            up_bias=up_bias,
            down_bias=down_bias,
            norm_bias=norm_bias,
            use_tkg_gate_up_proj_column_tiling=use_tkg_gate_up_proj_column_tiling,
            use_tkg_down_proj_column_tiling=use_tkg_down_proj_column_tiling,
            use_tkg_down_proj_optimized_layout=use_tkg_down_proj_optimized_layout,
            transposed_in=transposed_in,
            transposed_out=transposed_out,
            is_model_config=True,
        )

    @pytest.mark.tier0
    @pytest.mark.parametrize(TKG_UNIT_PARAM_NAMES, _TIER0_PARAMS, ids=_TIER0_IDS)
    def test_tier0(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden,
        intermediate,
        dtype,
        quant_dtype,
        quant_type,
        tpbSgCyclesSum,
        norm_type,
        fused_add,
        store_add,
        act_fn_type,
        skip_gate,
        gate_bias,
        up_bias,
        down_bias,
        norm_bias,
        use_tkg_gate_up_proj_column_tiling,
        use_tkg_down_proj_column_tiling,
        use_tkg_down_proj_optimized_layout,
        transposed_in,
        transposed_out,
    ):
        """TIER0: Critical model configs - highest priority for model validation."""
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        self._run_model_test(**kwargs)

    @pytest.mark.optimal
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(TKG_UNIT_PARAM_NAMES, _OPTIMAL_PARAMS, ids=_OPTIMAL_IDS)
    def test_optimal(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden,
        intermediate,
        dtype,
        quant_dtype,
        quant_type,
        tpbSgCyclesSum,
        norm_type,
        fused_add,
        store_add,
        act_fn_type,
        skip_gate,
        gate_bias,
        up_bias,
        down_bias,
        norm_bias,
        use_tkg_gate_up_proj_column_tiling,
        use_tkg_down_proj_column_tiling,
        use_tkg_down_proj_optimized_layout,
        transposed_in,
        transposed_out,
    ):
        """OPTIMAL: Performance-optimized model configs."""
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        self._run_model_test(**kwargs)

    @pytest.mark.generality
    @pytest.mark.parametrize(TKG_UNIT_PARAM_NAMES, _GENERALITY_PARAMS, ids=_GENERALITY_IDS)
    def test_generality(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden,
        intermediate,
        dtype,
        quant_dtype,
        quant_type,
        tpbSgCyclesSum,
        norm_type,
        fused_add,
        store_add,
        act_fn_type,
        skip_gate,
        gate_bias,
        up_bias,
        down_bias,
        norm_bias,
        use_tkg_gate_up_proj_column_tiling,
        use_tkg_down_proj_column_tiling,
        use_tkg_down_proj_optimized_layout,
        transposed_in,
        transposed_out,
    ):
        """GENERALITY: Broad coverage model configs."""
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        self._run_model_test(**kwargs)
