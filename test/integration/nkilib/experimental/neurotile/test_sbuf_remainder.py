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

"""NKI integration tests for SBUF remainder tile handling.

Tests alloc_tiles with element_shape, SBUF tile indexing/data shapes,
retile of remainder tiles, and end-to-end transpose with remainder.

Note: NeuroTile's mlp_cte_remainder_* tests imported from a fgcc_mlp
example kernel that is not part of this migration. Those tests are
deliberately omitted from this port.
"""

import ml_dtypes
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
import torch

from nkilib_src.nkilib.experimental import neurotile as nt
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


@nki.jit
def _alloc_tiles_remainder_kernel(src):
    """Allocate tiled SBUF with remainder, verify shapes via trace-time asserts."""
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    gated = nt.alloc_tiles(
        tile_size=(128, 512),
        buffer_type=nl.sbuf,
        dtype=nl.bfloat16,
        element_shape=(128, 1792),
    )
    assert gated.shape == (1, 4), f"Expected (1,4), got {gated.shape}"
    assert gated.data.shape == (128, 1792), f"Expected (128,1792), got {gated.data.shape}"

    for i in range(3):
        t = gated[0, i]
        assert t.element_shape == (128, 512), f"Tile [0,{i}] expected (128,512)"
        assert t.data.shape == (128, 512), f"Tile [0,{i}] data expected (128,512)"
        assert not t.is_remainder

    rem = gated[0, 3]
    assert rem.element_shape == (128, 256), "Remainder expected (128,256)"
    assert rem.data.shape == (128, 256), "Remainder data expected (128,256)"
    assert rem.is_remainder

    nisa.dma_copy(dst, src)
    return dst


@nki.jit
def _alloc_tiles_multirow_kernel(src):
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    gated = nt.alloc_tiles(
        tile_size=(128, 512),
        buffer_type=nl.sbuf,
        dtype=nl.bfloat16,
        element_shape=(256, 1792),
    )
    assert gated.shape == (2, 4)

    for row in range(2):
        assert gated[row, 0].data.shape == (128, 512)
        assert gated[row, 3].data.shape == (128, 256)
        assert gated[row, 3].is_remainder

    nisa.dma_copy(dst, src)
    return dst


@nki.jit
def _alloc_tiles_exact_kernel(src):
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    gated = nt.alloc_tiles(
        tile_size=(128, 512),
        buffer_type=nl.sbuf,
        dtype=nl.bfloat16,
        element_shape=(128, 2048),
    )
    assert gated.shape == (1, 4)
    assert gated.data.shape == (128, 2048)
    assert not gated.is_remainder

    for i in range(4):
        assert gated[0, i].data.shape == (128, 512)
        assert not gated[0, i].is_remainder

    nisa.dma_copy(dst, src)
    return dst


@nki.jit
def _retile_remainder_kernel(src):
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    gated = nt.alloc_tiles(
        tile_size=(128, 512),
        buffer_type=nl.sbuf,
        dtype=nl.bfloat16,
        element_shape=(128, 1792),
    )

    sub_full = nt.tiles(gated[0, 0], tile_size=(128, 128))
    assert sub_full.shape == (1, 4)
    for j in range(4):
        assert sub_full[0, j].data.shape == (128, 128)

    sub_rem = nt.tiles(gated[0, 3], tile_size=(128, 128))
    assert sub_rem.shape == (1, 2)
    for j in range(2):
        assert sub_rem[0, j].data.shape == (128, 128)

    nisa.dma_copy(dst, src)
    return dst


@nki.jit
def _transpose_remainder_kernel(src):
    M, I = src.shape
    dst = nl.ndarray((M, I), dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(128, 512))

    gated = nt.alloc_tiles(
        tile_size=(128, 512),
        buffer_type=nl.sbuf,
        dtype=src.dtype,
        element_shape=(128, I),
    )

    for i in range(src_tiles.shape[1]):
        t = src_tiles[0, i].load()
        actual_f = t.data.shape[1]
        nisa.tensor_copy(gated[0, i].data[:, :actual_f], t.data)

    for i in range(gated.shape[1]):
        block = gated[0, i]
        subtiles = nt.tiles(block, tile_size=(128, 128))
        for j in range(subtiles.shape[1]):
            st = subtiles[0, j]
            psum_tmp = nl.ndarray((128, 128), dtype=src.dtype, buffer=nl.psum)
            nisa.nc_transpose(psum_tmp, st.data)
            nisa.tensor_copy(st.data, psum_tmp)

    for i in range(gated.shape[1]):
        actual_f = gated[0, i].data.shape[1]
        nisa.dma_copy(dst[:128, nl.ds(i * 512, actual_f)], gated[0, i].data)

    return dst


def _small_src_inputs(_):
    np.random.seed(42)
    return {"src": np.random.randn(128, 128).astype(ml_dtypes.bfloat16)}


def _small_src_output(kernel_input):
    return {"out": np.zeros_like(kernel_input["src"])}


def _passthrough_ref(src: torch.Tensor) -> torch.Tensor:
    return src.clone()


def _transpose_inputs(_):
    np.random.seed(42)
    return {"src": np.random.randn(128, 1792).astype(ml_dtypes.bfloat16)}


def _transpose_output(kernel_input):
    return {"out": np.zeros_like(kernel_input["src"])}


def _transpose_ref(src: torch.Tensor) -> torch.Tensor:
    """Per-tile in-place transpose: each (128, 128) sub-tile gets transposed.

    The kernel transposes within each tile_size=(128, 128) sub-tile. For
    a (128, 1792) source: 14 sub-tiles per row, each transposed.
    """
    M, I = src.shape
    out = src.clone()
    sub_w = 128
    for i in range(I // sub_w):
        s = i * sub_w
        e = s + sub_w
        out[:, s:e] = src[:, s:e].t()
    # Remainder columns (1792 % 128 = 0 here, so no extras)
    return out


@pytest_marks(["neurotile"])
class TestSBUFRemainderShapes:
    """alloc_tiles with element_shape: shape and data shape pinning."""

    @pytest.mark.fast
    def test_alloc_tiles_remainder_shapes(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_alloc_tiles_remainder_kernel,
            torch_ref=torch_ref_wrapper(_passthrough_ref),
            kernel_input_generator=_small_src_inputs,
            output_tensor_descriptor=_small_src_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_alloc_tiles_multirow_remainder(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_alloc_tiles_multirow_kernel,
            torch_ref=torch_ref_wrapper(_passthrough_ref),
            kernel_input_generator=_small_src_inputs,
            output_tensor_descriptor=_small_src_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_alloc_tiles_exact_fit(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_alloc_tiles_exact_kernel,
            torch_ref=torch_ref_wrapper(_passthrough_ref),
            kernel_input_generator=_small_src_inputs,
            output_tensor_descriptor=_small_src_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_retile_remainder(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_retile_remainder_kernel,
            torch_ref=torch_ref_wrapper(_passthrough_ref),
            kernel_input_generator=_small_src_inputs,
            output_tensor_descriptor=_small_src_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )


@pytest_marks(["neurotile"])
class TestSBUFRemainderTranspose:
    """End-to-end transpose with remainder-allocated SBUF."""

    @pytest.mark.fast
    def test_transpose_remainder(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_transpose_remainder_kernel,
            torch_ref=torch_ref_wrapper(_transpose_ref),
            kernel_input_generator=_transpose_inputs,
            output_tensor_descriptor=_transpose_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )
