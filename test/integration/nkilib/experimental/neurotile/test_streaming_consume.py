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

"""Behavioral tests for the post-stream-load view state.

`stream(dim=).load(k)` is a "consume one step on dim" operation. These
kernels assert the post-consume Grid/Layout state inside the kernel body
(via Python assert / NKI tracer), then dma_copy(out, src) so we can
validate end-to-end with a passthrough torch reference.
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

# Tile-grid streaming kernels: src=(512, 1024), tile_size=(128, 128) -> (4, 8) grid


@nki.jit
def _kernel_stream_tiles_default_dim0(src):
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.tiles(src, tile_size=(128, 128))
    s = v.stream(buffer_count=2)
    assert s.count == 4, str(s.count)
    slot = s.load(0)
    assert slot.element_shape == (128, 1024), str(slot.element_shape)
    assert slot.grid.cursor == 1, str(slot.grid.cursor)
    assert slot.shape == (8,), str(slot.shape)
    nisa.dma_copy(out, src)
    return out


@nki.jit
def _kernel_stream_tiles_explicit_dim1(src):
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.tiles(src, tile_size=(128, 128))
    s = v.stream(dim=1, buffer_count=2)
    assert s.count == 8, str(s.count)
    slot = s.load(0)
    assert slot.element_shape == (512, 128), str(slot.element_shape)
    assert slot.grid.cursor == 0, str(slot.grid.cursor)
    assert slot.shape == (4, 1), str(slot.shape)
    nisa.dma_copy(out, src)
    return out


@nki.jit
def _kernel_stream_tile_row_via_int(src):
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.tiles(src, tile_size=(128, 128))
    row = v[2]
    assert row.grid.cursor == 1, str(row.grid.cursor)
    s = row.stream(buffer_count=2)
    assert s.count == 8, str(s.count)
    slot = s.load(0)
    assert slot.element_shape == (128, 128), str(slot.element_shape)
    assert slot.grid.cursor == slot.ndim, str(slot.grid.cursor)
    nisa.dma_copy(out, src)
    return out


@nki.jit
def _kernel_stream_tile_col_via_int(src):
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.tiles(src, tile_size=(128, 128))
    col = v[:, 3]
    assert col.grid.cursor == 0, str(col.grid.cursor)
    s = col.stream(buffer_count=2)
    assert s.count == 4, str(s.count)
    nisa.dma_copy(out, src)
    return out


@nki.jit
def _kernel_stream_tile_subgrid(src):
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.tiles(src, tile_size=(128, 128))
    sub = v[1:3, :]
    s = sub.stream(buffer_count=2)
    assert s.count == 2, str(s.count)
    nisa.dma_copy(out, src)
    return out


# Block-grid streaming: src=(512, 1024), tile_size=(128, 256), block_size=(2, 2) -> (2, 2) blocks


@nki.jit
def _kernel_stream_blocks_default_dim0(src):
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.blocks(src, tile_size=(128, 256), block_size=(2, 2))
    s = v.stream(buffer_count=2)
    assert s.count == 2, str(s.count)
    nisa.dma_copy(out, src)
    return out


@nki.jit
def _kernel_stream_blocks_explicit_dim1(src):
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.blocks(src, tile_size=(128, 256), block_size=(2, 2))
    s = v.stream(dim=1, buffer_count=2)
    assert s.count == 2, str(s.count)
    nisa.dma_copy(out, src)
    return out


@nki.jit
def _kernel_stream_block_row_via_int(src):
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.blocks(src, tile_size=(128, 256), block_size=(2, 2))
    row = v[0]
    assert row.grid.cursor == 1, str(row.grid.cursor)
    s = row.stream(buffer_count=2)
    assert s.count == 2, str(s.count)
    nisa.dma_copy(out, src)
    return out


@nki.jit
def _kernel_stream_block_col_via_int(src):
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.blocks(src, tile_size=(128, 256), block_size=(2, 2))
    col = v[:, 1]
    assert col.grid.cursor == 0, str(col.grid.cursor)
    s = col.stream(buffer_count=2)
    assert s.count == 2, str(s.count)
    nisa.dma_copy(out, src)
    return out


# Common fixture: passthrough kernel takes src, returns src.clone() unmodified


def _src_inputs(_):
    np.random.seed(42)
    return {"src": np.random.randn(512, 1024).astype(ml_dtypes.bfloat16)}


def _src_output(kernel_input):
    return {"out": np.zeros_like(kernel_input["src"])}


def _passthrough_ref(src: torch.Tensor) -> torch.Tensor:
    return src.clone()


_KERNELS = [
    _kernel_stream_tiles_default_dim0,
    _kernel_stream_tiles_explicit_dim1,
    _kernel_stream_tile_row_via_int,
    _kernel_stream_tile_col_via_int,
    _kernel_stream_tile_subgrid,
    _kernel_stream_blocks_default_dim0,
    _kernel_stream_blocks_explicit_dim1,
    _kernel_stream_block_row_via_int,
    _kernel_stream_block_col_via_int,
]


@pytest_marks(["neurotile"])
class TestStreamingConsume:
    """Pin the post-stream view state across pre-narrow / stream-dim variants."""

    @pytest.mark.fast
    @pytest.mark.parametrize("kernel", _KERNELS, ids=lambda k: k.__name__.lstrip("_"))
    def test_stream_consume(self, test_manager: Orchestrator, platform_target: Platforms, kernel):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref_wrapper(_passthrough_ref),
            kernel_input_generator=_src_inputs,
            output_tensor_descriptor=_src_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )
