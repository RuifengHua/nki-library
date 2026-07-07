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
"""Integration tests for tutorials in test/docs/neurotile/examples/03_tile_ops/."""

import ml_dtypes
import numpy as np
import pytest

from nkilib_src.nkilib.experimental.neurotile.examples._03_tile_ops import (
    _01_reshape_permute as reshape_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._03_tile_ops import (
    _01_reshape_permute_torch as reshape_refs,
)
from nkilib_src.nkilib.experimental.neurotile.examples._03_tile_ops import (
    _02_fold_pattern_override as fold_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._03_tile_ops import (
    _02_fold_pattern_override_torch as fold_refs,
)
from nkilib_src.nkilib.experimental.neurotile.examples._03_tile_ops import (
    _03_tensor_view as view_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._03_tile_ops import (
    _03_tensor_view_torch as view_refs,
)
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _run(test_manager, platform_target, kernel, ref, inputs, outputs, rtol=1e-2, atol=1e-2):
    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=kernel,
        torch_ref=torch_ref_wrapper(ref),
        kernel_input_generator=inputs,
        output_tensor_descriptor=outputs,
    )
    framework.run_test(
        test_config=None,
        compiler_args=CompilerArgs(platform_target=platform_target),
        rtol=rtol,
        atol=atol,
    )


@pytest_marks(["neurotile"])
class TestNeurotileReshapePermute:
    """Tutorials in 01_reshape_permute.py."""

    @pytest.mark.fast
    def test_norm_reshape_transpose(self, test_manager: Orchestrator, platform_target: Platforms):
        # src: [4, 1024] fp32, out: [128, 32]
        _run(
            test_manager,
            platform_target,
            reshape_mod.norm_reshape_transpose,
            reshape_refs.norm_reshape_transpose_torch_ref,
            lambda _: {"src_hbm": np.random.RandomState(42).randn(4, 1024).astype(np.float32)},
            lambda _: {"out": np.zeros((128, 32), dtype=np.float32)},
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    def test_reshape_permute_load(self, test_manager, platform_target):
        # src: [4, 1024] fp32, out: [128, 4, 8]
        _run(
            test_manager,
            platform_target,
            reshape_mod.reshape_permute_load,
            reshape_refs.reshape_permute_load_torch_ref,
            lambda _: {"src_hbm": np.random.RandomState(42).randn(4, 1024).astype(np.float32)},
            lambda _: {"out": np.zeros((128, 4, 8), dtype=np.float32)},
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    def test_bsh_to_h0_bs_h1(self, test_manager, platform_target):
        # B=4, S=32, H=128 => out: [H0=8, B*S=128, H1=16]
        _run(
            test_manager,
            platform_target,
            reshape_mod.bsh_to_h0_bs_h1,
            reshape_refs.bsh_to_h0_bs_h1_torch_ref,
            lambda _: {"src_hbm": np.random.RandomState(42).randn(4, 32, 128).astype(np.float32)},
            lambda _: {"out": np.zeros((8, 128, 16), dtype=np.float32)},
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    def test_attention_qk_layout(self, test_manager, platform_target):
        # q: [B=4, S=32, H=128], out: [head_dim=16, B*S=128]
        _run(
            test_manager,
            platform_target,
            reshape_mod.attention_qk_layout,
            reshape_refs.attention_qk_layout_torch_ref,
            lambda _: {"q_hbm": np.random.RandomState(42).randn(4, 32, 128).astype(np.float32)},
            lambda _: {"out": np.zeros((16, 128), dtype=np.float32)},
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "kernel,ref,shape",
        [
            (reshape_mod.tile_reshape_elementwise, reshape_refs.tile_reshape_elementwise_torch_ref, (128, 512)),
            (reshape_mod.block_reshape_elementwise, reshape_refs.block_reshape_elementwise_torch_ref, (256, 512)),
            (reshape_mod.block_reshape_chunked, reshape_refs.block_reshape_chunked_torch_ref, (256, 512)),
        ],
    )
    def test_reshape_elementwise(self, test_manager, platform_target, kernel, ref, shape):
        _run(
            test_manager,
            platform_target,
            kernel,
            ref,
            lambda _: {"src": np.random.RandomState(42).randn(*shape).astype(ml_dtypes.bfloat16)},
            lambda ki: {"out": np.zeros_like(ki["src"])},
        )


@pytest_marks(["neurotile"])
class TestNeurotileFoldPattern:
    """Tutorials in 02_fold_pattern_override.py."""

    @pytest.mark.fast
    def test_fold_into_partition(self, test_manager, platform_target):
        # src: [P=32, F=512, K=4] -> out: [K*P=128, F=512]
        _run(
            test_manager,
            platform_target,
            fold_mod.fold_into_partition,
            fold_refs.fold_into_partition_torch_ref,
            lambda _: {"src_hbm": np.random.RandomState(42).randn(32, 512, 4).astype(np.float32)},
            lambda _: {"out": np.zeros((128, 512), dtype=np.float32)},
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    def test_fold_free_dim(self, test_manager, platform_target):
        # src: [P=32, F=128, K=4] -> out: [P, F*K] = [32, 512]
        _run(
            test_manager,
            platform_target,
            fold_mod.fold_free_dim,
            fold_refs.fold_free_dim_torch_ref,
            lambda _: {"src_hbm": np.random.RandomState(42).randn(32, 128, 4).astype(np.float32)},
            lambda _: {"out": np.zeros((32, 512), dtype=np.float32)},
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "kernel,ref",
        [
            (fold_mod.fold_partition_roundtrip, fold_refs.fold_partition_roundtrip_torch_ref),
            (fold_mod.fold_free_dim_roundtrip, fold_refs.fold_free_dim_roundtrip_torch_ref),
        ],
    )
    def test_fold_roundtrip(self, test_manager, platform_target, kernel, ref):
        # src: [P=32, F=128, K=4]; out shape == in shape
        _run(
            test_manager,
            platform_target,
            kernel,
            ref,
            lambda _: {"src_hbm": np.random.RandomState(42).randn(32, 128, 4).astype(np.float32)},
            lambda ki: {"out": np.zeros_like(ki["src_hbm"])},
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    def test_fold_chain_4d(self, test_manager, platform_target):
        # src: [4, 8, 64, 8] -> out: [32, 512]
        _run(
            test_manager,
            platform_target,
            fold_mod.fold_chain_4d,
            fold_refs.fold_chain_4d_torch_ref,
            lambda _: {"src_hbm": np.random.RandomState(42).randn(4, 8, 64, 8).astype(np.float32)},
            lambda _: {"out": np.zeros((32, 512), dtype=np.float32)},
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    def test_pattern_override_load(self, test_manager, platform_target):
        # src: [P=32, F_src=1024] -> out: [P, F_dst=512] strided.
        _run(
            test_manager,
            platform_target,
            fold_mod.pattern_override_load,
            fold_refs.pattern_override_load_torch_ref,
            lambda _: {"src_hbm": np.random.RandomState(42).randn(32, 1024).astype(np.float32)},
            lambda _: {"out": np.zeros((32, 512), dtype=np.float32)},
            rtol=1e-5,
            atol=1e-5,
        )


@pytest_marks(["neurotile"])
class TestNeurotileTensorView:
    """Tutorials in 03_tensor_view.py."""

    @pytest.mark.fast
    def test_tensor_view_chain(self, test_manager, platform_target):
        # src: [B=1, S=4, H=256], out: [H0=128, B*S*H1=8]
        _run(
            test_manager,
            platform_target,
            view_mod.tensor_view_chain,
            view_refs.tensor_view_chain_torch_ref,
            lambda _: {"src": np.random.RandomState(42).randn(1, 4, 256).astype(ml_dtypes.bfloat16)},
            lambda _: {"out": np.zeros((128, 8), dtype=ml_dtypes.bfloat16)},
        )

    @pytest.mark.fast
    def test_tensor_view_select(self, test_manager, platform_target):
        # src: [4, 128, 16] bf16 -> out: src[0] * 3
        _run(
            test_manager,
            platform_target,
            view_mod.tensor_view_select,
            view_refs.tensor_view_select_torch_ref,
            lambda _: {"src": np.random.RandomState(42).randn(4, 128, 16).astype(ml_dtypes.bfloat16)},
            lambda _: {"out": np.zeros((128, 16), dtype=ml_dtypes.bfloat16)},
        )

    @pytest.mark.fast
    def test_tensor_view_3d_direct(self, test_manager, platform_target):
        # src: [128, 4, 32] fp32 -> out: same shape
        _run(
            test_manager,
            platform_target,
            view_mod.tensor_view_3d_direct,
            view_refs.tensor_view_3d_direct_torch_ref,
            lambda _: {"src": np.random.RandomState(42).randn(128, 4, 32).astype(np.float32)},
            lambda ki: {"out": np.zeros_like(ki["src"])},
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    def test_nd_tile_alloc_sbuf(self, test_manager, platform_target):
        # src: [128, 4, 32] fp32 -> out: [128, 128]
        _run(
            test_manager,
            platform_target,
            view_mod.nd_tile_alloc_sbuf,
            view_refs.nd_tile_alloc_sbuf_torch_ref,
            lambda _: {"src": np.random.RandomState(42).randn(128, 4, 32).astype(np.float32)},
            lambda _: {"out": np.zeros((128, 128), dtype=np.float32)},
            rtol=1e-5,
            atol=1e-5,
        )
