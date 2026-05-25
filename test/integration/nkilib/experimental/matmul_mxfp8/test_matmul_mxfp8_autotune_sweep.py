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

"""Auto-tuning sweep for MXFP8 matmul kernel config optimization.

Generates candidate configs for user-specified shapes and runs each on
hardware to collect performance data for offline cache updates.

"""

import os

import numpy as np
import pytest

from nkilib_src.nkilib.experimental.matmul_mxfp8 import matmul_mxfp8_generic_kernel
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import (
    MatmulMxfp8KernelConfig,
    generate_autotune_candidates,
)
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_torch import matmul_mxfp8_torch_ref
from test.integration.nkilib.experimental.matmul_mxfp8 import config_helper, constants
from test.integration.nkilib.experimental.matmul_mxfp8.test_matmul_mxfp8_generic_kernel import (
    _mxfp8_comparator,
    build_matmul_inputs,
    get_output_dtype,
)
from test.utils import common_dataclasses
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework

# ── Shape parsing ─────────────────────────────────────────────────────


def _parse_shapes():
    """Parse shapes from env vars.

    Supports two modes:
      1. AUTOTUNE_SHAPES="MxKxN,MxKxN,..." — explicit shapes
      2. AUTOTUNE_MODEL_CONFIG="/path/to/model.json" — generate shapes from model config
         Optional: AUTOTUNE_TP (default 1), AUTOTUNE_CP (default 1)

    Both can be combined.
    """
    shapes = []

    # Mode 1: explicit shapes
    raw = os.environ.get("AUTOTUNE_SHAPES", "")
    if raw.strip():
        for token in raw.split(","):
            token = token.strip()
            parts = token.split("x")
            if len(parts) != 3:
                raise ValueError(f"Invalid shape '{token}', expected MxKxN")
            M, K, N = int(parts[0]), int(parts[1]), int(parts[2])
            shapes.append((f"{M}x{K}x{N}", M, K, N))

    # Mode 2: model config
    model_config = os.environ.get("AUTOTUNE_MODEL_CONFIG", "")
    if model_config.strip():
        from test.integration.nkilib.experimental.matmul_mxfp8.model_config_reader import (
            generate_transformer_block_shapes,
        )

        tp = int(os.environ.get("AUTOTUNE_TP", "1"))
        cp = int(os.environ.get("AUTOTUNE_CP", "1"))
        model_shapes = generate_transformer_block_shapes(model_config, TP=tp, CP=cp)
        seen = set()
        for name, M, K, N in model_shapes:
            M, K, N = int(M), int(K), int(N)
            key = f"{M}x{K}x{N}"
            if key not in seen:
                seen.add(key)
                shapes.append((key, M, K, N))

    return shapes


def _is_prequant_only():
    return os.environ.get("AUTOTUNE_PREQUANT_ONLY", "0") not in ("0", "", "false")


# ── Candidate generation ─────────────────────────────────────────────


def _make_test_config(kc, dtype_mode, idx, shape_label):
    """Wrap a MatmulMxfp8KernelConfig into a TestConfig for the test harness."""
    if dtype_mode == "prequant":
        return config_helper.TestConfig(
            M=kc.M,
            K=kc.K,
            N=kc.N,
            tile_m=kc.tile_m,
            tile_k=kc.tile_k,
            tile_n=kc.tile_n,
            TILES_IN_BLOCK_M=kc.TILES_IN_BLOCK_M,
            TILES_IN_BLOCK_N=kc.TILES_IN_BLOCK_N,
            TILES_IN_BLOCK_K=kc.TILES_IN_BLOCK_K,
            TILES_IN_LOAD_M=kc.TILES_IN_LOAD_M,
            TILES_IN_LOAD_N=kc.TILES_IN_LOAD_N,
            run_with_lnc2=kc.run_with_lnc2,
            lnc_2_shard_rhs=kc.lnc_2_shard_rhs,
            description=f"pq_{shape_label}_c{idx}",
            seed=42,
            lhs_dtype=constants.MatrixPrecision.MXFP8_X4,
            rhs_dtype=constants.MatrixPrecision.MXFP8_X4,
            output_dtype=constants.MatrixPrecision.FP32,
            dists=["uniform", "uniform"],
            params=[{"a": -1.0, "b": 1.0}, {"a": -1.0, "b": 1.0}],
        )
    else:
        return config_helper.TestConfig(
            M=kc.M,
            K=kc.K,
            N=kc.N,
            tile_m=kc.tile_m,
            tile_k=kc.tile_k,
            tile_n=kc.tile_n,
            TILES_IN_BLOCK_M=kc.TILES_IN_BLOCK_M,
            TILES_IN_BLOCK_N=kc.TILES_IN_BLOCK_N,
            TILES_IN_BLOCK_K=kc.TILES_IN_BLOCK_K,
            TILES_IN_LOAD_M=kc.TILES_IN_LOAD_M,
            TILES_IN_LOAD_N=kc.TILES_IN_LOAD_N,
            run_with_lnc2=True,
            lnc_2_shard_rhs=kc.lnc_2_shard_rhs,
            description=f"bf16_{shape_label}_c{idx}",
            seed=42,
            lhs_dtype=constants.MatrixPrecision.BFLOAT16,
            rhs_dtype=constants.MatrixPrecision.BFLOAT16,
            output_dtype=constants.MatrixPrecision.FP32,
            enable_scale_packing=False,
            lhs_is_swizzled=True,
            rhs_is_swizzled=True,
            spill_reload=True,
            dists=["uniform", "uniform"],
            params=[{"a": -1.0, "b": 1.0}, {"a": -1.0, "b": 1.0}],
        )


def _generate_all_configs():
    """Generate TestConfig list from AUTOTUNE_SHAPES env var."""
    shapes = _parse_shapes()
    prequant_only = _is_prequant_only()
    modes = ["prequant"] if prequant_only else ["prequant", "bf16"]
    configs = []
    for label, M, K, N in shapes:
        base = MatmulMxfp8KernelConfig(M=M, K=K, N=N)
        candidates = generate_autotune_candidates(base)
        for mode in modes:
            for idx, kc in enumerate(candidates):
                configs.append(_make_test_config(kc, mode, idx, label))
    return configs


AUTOTUNE_CONFIGS = _generate_all_configs()


# ── Test class ────────────────────────────────────────────────────────


@pytest_test_metadata(name="Matmul MXFP8 Autotune Sweep")
@pytest_marks(["matmul_mxfp8", "mx", "mxfp8", "autotune"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
class TestMatmulMxfp8AutotuneSweep:
    @pytest.mark.parametrize(
        "conf",
        AUTOTUNE_CONFIGS,
        ids=[c.description for c in AUTOTUNE_CONFIGS],
    )
    def test_autotune_sweep(self, test_manager, conf, platform_target):
        """Sweep autotune candidates for user-specified shapes."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")

        output_dtype = get_output_dtype(conf)

        def input_generator(test_config):
            return build_matmul_inputs(conf)

        def output_tensors(kernel_input):
            return {"out": np.zeros((conf.M, conf.N), dtype=output_dtype)}

        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.run_with_lnc2 else 1,
            platform_target=platform_target,
            additional_cmd_args=["--internal-backend-options=--enable-mx-alternative-emax"],
        )

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=matmul_mxfp8_generic_kernel.matmul_mxfp8,
            torch_ref=matmul_mxfp8_torch_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            custom_comparator=_mxfp8_comparator(conf, output_dtype, gpu_golden_enabled=False),
        )
