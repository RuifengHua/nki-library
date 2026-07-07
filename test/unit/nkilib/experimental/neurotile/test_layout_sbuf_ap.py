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

"""Tests for SBUFLayout AP pattern construction.

Locks the partition-stride invariant: level-0 stride of every emitted
AP pattern must equal the underlying SBUF ndarray's free-dim flat width
(``product(sbuf.shape[1:])``), regardless of how narrow the walked region
is. The compiler enforces this rule in
``nki/isa/validation.py::validate_contiguous_partition_access``; emitting a
narrower stride causes a compile-time assertion.

Pure Python: stubs out ``nl.ndarray.ap()`` so we can introspect the pattern
the layout would have submitted.
"""

import pytest

from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout
from test.utils.pytest_test_metadata import pytest_marks


class _StubSBUF:
    """Minimal stand-in for ``nl.ndarray`` that captures the AP pattern.

    ``_build_default_ap`` reads ``.shape``, optionally ``._storage_shape``
    (NKI views preserve it through slicing), and calls ``.ap(pattern,
    offset?)``. Pass ``storage_shape=`` to model a sliced view of a larger
    underlying buffer.
    """

    def __init__(self, shape, storage_shape=None):
        self.shape = shape
        self._storage_shape = storage_shape if storage_shape is not None else shape
        self.captured_pattern = None
        self.captured_offset = None

    def ap(self, pattern, offset=None):
        self.captured_pattern = pattern
        self.captured_offset = offset
        return self  # value unused by tests


@pytest_marks(["neurotile"])
class TestPartitionRowStride:
    """``_partition_row_stride`` returns the underlying-buffer free width."""

    @pytest.mark.fast
    def test_2d_buffer(self):
        sbuf = _StubSBUF(shape=(128, 512))
        assert SBUFLayout._partition_row_stride(sbuf) == 512

    @pytest.mark.fast
    def test_3d_buffer(self):
        sbuf = _StubSBUF(shape=(128, 32, 4))
        assert SBUFLayout._partition_row_stride(sbuf) == 128

    @pytest.mark.fast
    def test_independent_of_walked_remaining(self):
        """Stride is a property of the ndarray, not the caller's walk."""
        wide = _StubSBUF(shape=(128, 1024))
        assert SBUFLayout._partition_row_stride(wide) == 1024


@pytest_marks(["neurotile"])
class TestBuildDefaultAp:
    """``_build_default_ap`` produces compiler-valid AP patterns.

    Level-0 always pins to the underlying ndarray's free width (read from
    ``_storage_shape``), satisfying the new compiler's
    ``partition_step == tensor.free_dim_size`` invariant.
    """

    @pytest.mark.fast
    def test_single_p_tile_full_walk(self):
        sbuf = _StubSBUF(shape=(128, 512))
        SBUFLayout._build_default_ap(sbuf, remaining=(128, 512), tile_p=128, p_tiles=1)
        # [[partition_stride, tile_p], [1, remaining[1]]]
        # partition_stride = product((128, 512)[1:]) = 512
        assert sbuf.captured_pattern == [[512, 128], [1, 512]]

    @pytest.mark.fast
    def test_single_p_tile_subtile_walk(self):
        """Sub-tile (1x1) walk on a wider buffer: level-0 still pins to underlying free width."""
        sbuf = _StubSBUF(shape=(128, 64))
        SBUFLayout._build_default_ap(sbuf, remaining=(1, 1), tile_p=1, p_tiles=1)
        # [[partition_stride=64, tile_p=1], [1, remaining[1]=1]]
        assert sbuf.captured_pattern == [[64, 1], [1, 1]]

    @pytest.mark.fast
    def test_single_p_tile_partial_f_walk(self):
        """Partial F walk over a wider buffer (the original bug case)."""
        sbuf = _StubSBUF(shape=(128, 512))
        SBUFLayout._build_default_ap(sbuf, remaining=(128, 128), tile_p=128, p_tiles=1)
        # partition_stride = 512 (parent), inner walks remaining[1] = 128
        assert sbuf.captured_pattern == [[512, 128], [1, 128]]

    @pytest.mark.fast
    def test_multi_p_tile_uses_allocated_free_width_as_partition_stride(self):
        sbuf = _StubSBUF(shape=(128, 1024))
        SBUFLayout._build_default_ap(sbuf, remaining=(256, 512), tile_p=128, p_tiles=2)
        # [[partition_stride, tile_p], [1, total_f]]
        # partition_stride = product((128, 1024)[1:]) = 1024
        # total_f = f_extent((256, 512), 128) = 512 * 2 = 1024
        assert sbuf.captured_pattern == [[1024, 128], [1, 1024]]
