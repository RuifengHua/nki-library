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

"""SBUF / HBM AP correctness on partial-trailing-tile loads.

Two failure modes that were caught at production-kernel scale:

1. Partition stride (`_build_ap`): when the rotating buffer slot is
   wider than one walked block (e.g., partial trailing I-tile on a
   sliced view), the AP partition stride must equal the SBUF ndarray's
   allocated F width, not the walked F.

2. HBM AP merge clamp (`ap_emitter`): on a sharded view, a gappy
   tile-walk axis (step > nested walk) merged with an outer block axis
   must clamp the merged count using inner-walk-aware arithmetic.
"""

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
def _stream_partial_tile_kernel(src):
    """Stream-load a partial trailing I-tile from a wider rotating buffer."""
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    src_blocks = nt.blocks(src, tile_size=(128, 256), block_size=(4, 1))
    src_partial = src_blocks[:, 1:2]
    src_stream = src_partial.stream(buffer_count=2)

    for k_block_idx in range(src_partial.shape[0]):
        loaded = src_stream.load(k_block_idx)
        tiles = nt.tiles(loaded)
        for k in range(tiles.shape[0]):
            psum = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(psum, 0.0)

    nisa.dma_copy(dst, src)
    return dst


@nki.jit
def _interleaved_block_load_kernel(src):
    """Sharded interleaved + multi-block load with tolist iteration."""
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    shard_id = nl.program_id(0)
    num_shards = nl.num_programs(0)

    src_flat = nt.tensor_view(src).flatten_dims(0, 1)
    r = nt.interleaved_range(rank=shard_id, num_shards=num_shards, total=8)
    tiles_view = nt.tiles(src_flat, tile_size=(128, 128))[r, :]
    blocks = nt.blocks(tiles_view, block_size=(2, 1))

    items = blocks.tolist()
    for i in range(len(items)):
        block_row = items[i]
        sb_block = block_row.load()  # noqa: F841

    nisa.dma_copy(dst, src)
    return dst


def _stream_partial_inputs(_):
    np.random.seed(42)
    return {
        "src": np.random.randn(1024, 384)
        .astype(np.float32)
        .astype(np.dtype("bfloat16") if hasattr(np, "bfloat16") else np.float32)
    }


def _stream_partial_inputs_simple(_):
    np.random.seed(42)
    import ml_dtypes

    return {"src": np.random.randn(1024, 384).astype(ml_dtypes.bfloat16)}


def _stream_partial_output(kernel_input):
    return {"out": np.zeros(kernel_input["src"].shape, dtype=kernel_input["src"].dtype)}


def _stream_partial_ref(src: torch.Tensor) -> torch.Tensor:
    """The kernel's only observable behavior is `nisa.dma_copy(dst, src)`."""
    return src.clone()


def _interleaved_block_inputs(_):
    np.random.seed(42)
    import ml_dtypes

    return {"src": np.random.randn(1, 1024, 8192).astype(ml_dtypes.bfloat16)}


def _interleaved_block_output(kernel_input):
    return {"out": np.zeros(kernel_input["src"].shape, dtype=kernel_input["src"].dtype)}


def _interleaved_block_ref(src: torch.Tensor) -> torch.Tensor:
    """The kernel's only observable behavior is `nisa.dma_copy(dst, src)`."""
    return src.clone()


@pytest_marks(["neurotile"])
class TestSBUFAPPartitionStride:
    """Validation that AP construction emits correct strides for partial tiles."""

    @pytest.mark.fast
    def test_stream_partial_tile_partition_stride(self, test_manager: Orchestrator, platform_target: Platforms):
        """SBUFLayout._build_ap: partition stride uses allocated F, not walked F."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_stream_partial_tile_kernel,
            torch_ref=torch_ref_wrapper(_stream_partial_ref),
            kernel_input_generator=_stream_partial_inputs_simple,
            output_tensor_descriptor=_stream_partial_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.fast
    def test_interleaved_block_load_merge_clamp(self, test_manager: Orchestrator, platform_target: Platforms):
        """APEmitter._merge_contiguous: clamp accounts for inner-walk on gappy axes."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_interleaved_block_load_kernel,
            torch_ref=torch_ref_wrapper(_interleaved_block_ref),
            kernel_input_generator=_interleaved_block_inputs,
            output_tensor_descriptor=_interleaved_block_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=2),
            rtol=1e-2,
            atol=1e-2,
        )
