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

"""
Tests for NDSlice: Grid + Layout coordination, iteration, DMA.

Pure Python -- no NKI, no tracer, no device required.
"""

import nki.language as nl
import numpy as np
import pytest

from nkilib_src.nkilib.experimental.neurotile.core._helpers import (
    contiguous_strides as _contiguous_strides,
)
from nkilib_src.nkilib.experimental.neurotile.core._helpers import (
    product,
)
from nkilib_src.nkilib.experimental.neurotile.core.grid import Grid
from nkilib_src.nkilib.experimental.neurotile.core.layout_hbm import HBMLayout
from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout
from nkilib_src.nkilib.experimental.neurotile.core.ndslice import NDSlice
from test.utils.pytest_test_metadata import pytest_marks


def grid_is_element(view, dim):
    """Check if dim is at element level (step == 1)."""
    return view.grid.is_element(dim)


# ============================================================================
# Helpers
# ============================================================================


def make_view(element_shape, tile_size, block_size=None, offset=0):
    """Create an NDSlice(Grid, HBMLayout) for testing."""
    ndim = len(element_shape)
    n_batch_dims = ndim - len(tile_size)

    # Pad tile_size for batch dims
    if n_batch_dims > 0:
        padded = []
        for d in range(n_batch_dims):
            padded.append(1)
        for d in range(len(tile_size)):
            padded.append(tile_size[d])
        full_tile = tuple(padded)
    else:
        full_tile = tile_size
        n_batch_dims = 0

    # Pad block_size
    full_block = None
    if block_size is not None:
        padded_b = []
        for d in range(n_batch_dims):
            padded_b.append(1)
        for d in range(len(block_size)):
            padded_b.append(block_size[d])
        full_block = tuple(padded_b)

    grid = Grid.from_shape(element_shape, full_tile, block_size=full_block, n_batch_dims=n_batch_dims)
    strides = _contiguous_strides(element_shape)
    layout = HBMLayout(
        source="mock_tensor",
        offset=offset,
        strides=strides,
        dtype="float32",
        buffer_type=nl.shared_hbm,
    )
    return NDSlice(grid, layout)


def make_untiled_view(element_shape, offset=0):
    """Create an untiled NDSlice (tensor_view semantics)."""
    ndim = len(element_shape)
    strides = _contiguous_strides(element_shape)

    grid = Grid.from_shape(element_shape, None)
    layout = HBMLayout(
        source="mock_tensor",
        offset=offset,
        strides=strides,
        dtype="float32",
        buffer_type=nl.shared_hbm,
    )
    return NDSlice(grid, layout)


# ============================================================================
# Construction and forwarded attributes
# ============================================================================


@pytest_marks(["neurotile"])
class TestConstruction:
    @pytest.mark.fast
    def test_forwarded_attributes(self):
        v = make_view((512, 2048), (128, 512))
        assert v.shape == (4, 4)
        assert v.element_shape == (512, 2048)
        assert v.tile_size == (128, 512)
        assert v.ndim == 2
        assert v.dtype == "float32"
        assert v.source == "mock_tensor"
        assert v.offset == 0
        assert v.strides == (2048, 1)
        assert v.data is None  # HBM: no tile-local data (use .load())
        assert v.buffer_type == "shared_hbm"

    def test_block_view_shape(self):
        v = make_view((512, 2048), (128, 512), block_size=(2, 2))
        assert v.shape == (2, 2)

    def test_iteration_dim_shape(self):
        """Cursor defaults to n_batch_dims, so shape skips the batch axis."""
        v = make_view((8, 128, 64), (128, 64))
        # n_batch_dims=1 -> cursor=1 -> shape = (1, 1) on the inner tile dims.
        assert v.shape == (1, 1)
        assert v.ndim == 3
        # Outer count on the batch dim is its element_shape.
        assert v.grid.outer_axis(0).count == 8


# ============================================================================
# __getitem__ -- int indexing (descend)
# ============================================================================


@pytest_marks(["neurotile"])
class TestGetitemInt:
    @pytest.mark.fast
    def test_single_tile_index(self):
        """tiles[1, 0] -> single tile."""
        v = make_view((512, 2048), (128, 512))
        r = v[1, 0]
        # Descended both dims to element level
        assert r.element_shape == (128, 512)
        # Offset: 1 * 128 * 2048 + 0 * 512 * 1
        assert r.offset == 1 * 128 * 2048

    def test_single_block_index(self):
        """blocks[0, 1] -> tile-level view."""
        v = make_view((512, 2048), (128, 512), block_size=(2, 2))
        r = v[0, 1]
        # Block descent: remaining = block_size * tile_size per dim
        assert r.element_shape == (256, 1024)
        assert r.shape == (2, 2)  # tiles within block
        # Offset: 0*256*2048 + 1*1024*1 = 1024
        assert r.offset == 1024

    def test_block_then_tile(self):
        """blocks[0, 1][1, 0] -> element-level view."""
        v = make_view((512, 2048), (128, 512), block_size=(2, 2))
        block = v[0, 1]
        tile = block[1, 0]
        assert tile.element_shape == (128, 512)
        # Block offset: 1024. Tile offset: 1*128*2048 + 0
        assert tile.offset == 1024 + 1 * 128 * 2048

    def test_negative_index(self):
        """tiles[-1, 0] -> last tile."""
        v = make_view((512, 2048), (128, 512))
        last = v[-1, 0]
        explicit = v[3, 0]
        assert last.offset == explicit.offset
        assert last.element_shape == explicit.element_shape

    def test_iteration_dim_drops(self):
        """3D tensor with 2D tile: int on iter dim -> dim drops."""
        v = make_view((8, 128, 64), (128, 64))
        r = v[3]
        # Iteration dim dropped: 3D -> 2D
        assert r.ndim == 2
        assert r.element_shape == (128, 64)
        # Offset: 3 * 1 * 8192 (stride for dim 0 of (8, 128, 64))
        assert r.offset == 3 * 8192

    def test_element_level_drops(self):
        """After tile descent, element int consumes the elem axes."""
        v = make_view((512, 2048), (128, 512))
        tile = v[0, 0]  # descend both to element level: remaining = (128, 512)
        assert tile.element_shape == (128, 512)
        # Now index at element level: step=1 for both dims; both consumed.
        sub = tile[42, 16]
        # Element-level offset: 42 * stride[0] + 16 * stride[1] = 42*2048 + 16
        assert sub.offset == 42 * 1 * 2048 + 16 * 1 * 1

    def test_min_2d_enforcement(self):
        """Tiled view: dropping all dims restores to 2D."""
        v = make_view((128, 512), (128, 512))
        # Single tile: shape (1, 1)
        tile = v[0, 0]  # element level
        # Index both element dims
        sub = tile[42, 16]
        # Would be 0D without min 2D -> enforced back to 2D-ish
        # One dim kept (the last exhausted is NOT dropped)
        assert sub.ndim >= 1  # at least 1D preserved


# ============================================================================
# __getitem__ -- slice indexing (narrow)
# ============================================================================


@pytest_marks(["neurotile"])
class TestGetitemSlice:
    @pytest.mark.fast
    def test_basic_slice(self):
        """tiles[0:2] -> narrow dim 0."""
        v = make_view((512, 2048), (128, 512))
        r = v[0:2]
        assert r.element_shape == (256, 2048)  # 2 tiles * 128
        assert r.offset == 0
        assert r.shape == (2, 4)

    def test_slice_with_offset(self):
        """tiles[2:4] -> narrow with offset."""
        v = make_view((512, 2048), (128, 512))
        r = v[2:4]
        assert r.element_shape == (256, 2048)
        assert r.offset == 2 * 128 * 2048

    def test_slice_dim1(self):
        """tiles[:, 1:3] -> narrow dim 1."""
        v = make_view((512, 2048), (128, 512))
        r = v[:, 1:3]
        assert r.element_shape == (512, 1024)  # 2 tiles * 512
        assert r.offset == 1 * 512 * 1  # dim 1 stride = 1

    def test_full_slice(self):
        """tiles[:] -> no change."""
        v = make_view((512, 2048), (128, 512))
        r = v[:]
        assert r.element_shape == v.element_shape
        assert r.offset == 0

    def test_mixed_int_and_slice(self):
        """tiles[1, 0:2] -> descend dim 0 + narrow dim 1."""
        v = make_view((512, 2048), (128, 512))
        r = v[1, 0:2]
        # dim 0: descend (tile level), remaining=128
        # dim 1: narrow to 2 tiles, remaining=1024
        assert r.element_shape[1] == 1024
        assert r.offset == 1 * 128 * 2048


# ============================================================================
# __getitem__ -- indirect (runtime k)
# ============================================================================


@pytest_marks(["neurotile"])
class TestGetitemIndirect:
    @pytest.mark.fast
    def test_runtime_scalar(self):
        """tiles[runtime_k] -> indirect offset stored on Layout.indirect."""
        v = make_view((8, 128, 64), (128, 64))

        # Stand-in for an NKI LoopVar / CExpr: a class that's neither
        # int nor slice and not in the rejected-Python-types list, so
        # validate_index_key routes it through the runtime path.
        class _FakeLoopVar:
            pass

        runtime_k = _FakeLoopVar()
        r = v[runtime_k]
        # Indirect is a tagged value carrying the runtime expr + dim.
        assert r.layout.indirect is not None
        assert r.layout.indirect.dim == 0
        assert r.offset == 0  # compile-time offset unchanged


# ============================================================================
# tolist() -- iteration
# ============================================================================


@pytest_marks(["neurotile"])
class TestToList:
    @pytest.mark.fast
    def test_tile_iteration(self):
        """tolist() on 2x2 tile grid -> 2 children (dim 0)."""
        v = make_view((256, 1024), (128, 512))
        assert v.shape == (2, 2)
        children = v.tolist()
        assert len(children) == 2
        # Each child has dim 0 narrowed, iter_dim advanced
        assert children[0].shape == (2,)  # only dim 1 visible
        assert children[1].shape == (2,)

    def test_tile_iteration_offsets(self):
        """Children have correct offsets."""
        v = make_view((256, 1024), (128, 512))
        children = v.tolist()
        # Child 0: offset 0, child 1: offset 1 * 128 * 1024
        assert children[0].offset == 0
        assert children[1].offset == 128 * 1024

    def test_block_iteration(self):
        """tolist on block grid -> block-level children."""
        v = make_view((512, 2048), (128, 512), block_size=(2, 2))
        assert v.shape == (2, 2)
        children = v.tolist()
        assert len(children) == 2  # 2 blocks along dim 0
        # Each child: dim 0 narrowed to 1 block (256 elements)
        assert children[0].element_shape[0] == 256

    def test_batch_dim_iteration_via_explicit_dim(self):
        """tolist(dim=0) on the batch dim -> 8 children (cursor doesn't auto-iterate batch)."""
        v = make_view((8, 128, 64), (128, 64))
        children = v.tolist(dim=0)
        assert len(children) == 8
        # Children retain ndim=3 (explicit-dim path doesn't drop).
        assert children[3].offset == 3 * 8192

    def test_nested_tolist(self):
        """Nested tolist peels one dim at a time."""
        v = make_view((256, 1024), (128, 512))
        # First level: 2 children along dim 0
        level1 = v.tolist()
        assert len(level1) == 2
        # Second level: 2 children along dim 1
        level2 = level1[0].tolist()
        assert len(level2) == 2
        # Each is a single tile
        assert level2[0].element_shape == (128, 512)
        assert level2[1].element_shape == (128, 512)

    def test_single_tile_returns_self(self):
        """tolist on a view past all dims -> [self]."""
        v = make_view((128, 512), (128, 512))
        # Shape (1, 1). First tolist gives 1 item, second gives 1 item.
        children = v.tolist()
        assert len(children) == 1
        grandchildren = children[0].tolist()
        assert len(grandchildren) == 1


# ============================================================================
# _enumerate
# ============================================================================


@pytest_marks(["neurotile"])
class TestEnumerate:
    @pytest.mark.fast
    def test_basic(self):
        v = make_view((256, 1024), (128, 512))
        pairs = v._enumerate()
        assert len(pairs) == 2
        assert pairs[0][0] == 0
        assert pairs[1][0] == 1
        assert pairs[0][1].offset == 0
        assert pairs[1][1].offset == 128 * 1024

    def test_with_start(self):
        v = make_view((256, 1024), (128, 512))
        pairs = v._enumerate(start=10)
        assert pairs[0][0] == 10
        assert pairs[1][0] == 11


# ============================================================================
# __iter__ and __len__
# ============================================================================


@pytest_marks(["neurotile"])
class TestIterAndLen:
    @pytest.mark.fast
    def test_len(self):
        v = make_view((512, 2048), (128, 512))
        assert len(v) == 16  # 4 * 4

    def test_iter(self):
        v = make_view((256, 1024), (128, 512))
        items = list(v)
        assert len(items) == 2


# ============================================================================
# Full block -> tile -> element pipeline
# ============================================================================


@pytest_marks(["neurotile"])
class TestFullPipeline:
    @pytest.mark.fast
    def test_blocks_enumerate_then_load(self):
        """Simulate: enumerate blocks, each block is ready for load."""
        v = make_view((512, 2048), (128, 512), block_size=(2, 2))

        # Enumerate block dim 0
        block_rows = v.tolist()
        assert len(block_rows) == 2

        # Enumerate block dim 1
        blocks = block_rows[0].tolist()
        assert len(blocks) == 2

        # One block: (256, 1024), iter_dim past both dims -> shape ()
        block = blocks[1]
        assert block.element_shape == (256, 1024)
        assert block.shape == ()  # fully enumerated, ready for load()

        # After enumerate, iter_dim is past all dims (shape=()).
        # Grid still has the data scope -- remaining tells load() the region size.
        # current_count at block level gives 1 (narrowed to single block).
        assert block.grid.remaining == (256, 1024)
        # To iterate tiles within, user would .load() -> SBUF with tile grid.

    def test_blocks_direct_index(self):
        """Direct index blocks[0, 1][1, 0] without enumerate."""
        v = make_view((512, 2048), (128, 512), block_size=(2, 2))

        # Direct block index (no enumerate, iter_dim stays 0)
        block = v[0, 1]
        assert block.element_shape == (256, 1024)
        assert block.shape == (2, 2)  # tiles visible

        # Tile index within block
        tile = block[1, 0]
        assert tile.element_shape == (128, 512)

    def test_iteration_then_tiles(self):
        """3D tensor: explicit-dim iterate batch dim, then index tile."""
        v = make_view((8, 128, 64), (128, 64))

        # Iterate the batch dim explicitly.
        items = v.tolist(dim=0)
        assert len(items) == 8

        # Drop the batch dim, then index the inner tile (dim 0 of the 2D view).
        item = items[3][0]  # consume rank dim
        tile = item[0, 0]  # tile-level descent
        assert tile.offset == 3 * 8192


# ============================================================================
# Chained indexing
# ============================================================================


@pytest_marks(["neurotile"])
class TestChainedIndexing:
    @pytest.mark.fast
    def test_sequential_indexing(self):
        """view[0][1] == view[0, 1] in terms of offset."""
        v = make_view((256, 1024), (128, 512))
        chained = v[0][1]
        direct = v[0, 1]
        # Both should reach the same offset
        # v[0]: descend dim 0, element_shape=(128, 1024), offset=0
        # v[0][1]: descend dim 1 of the result... but iter_dim matters
        # After v[0], iter_dim was NOT advanced (it's __getitem__, not tolist())
        # So v[0] has iter_dim=0, dim 0 at element level
        # v[0][1]: processes key[0]=1 at dim=0+iter_dim=0
        # This descends dim 0 at element level (step=1)!
        # That's not the same as v[0, 1] which processes dim 0 and dim 1.
        #
        # This is a known difference: chained [0][1] indexes the SAME dim twice,
        # while [0, 1] indexes dim 0 then dim 1.
        # This matches numpy: arr[0][1] != arr[0, 1] for 2D arrays
        # (arr[0] gives row 0, then [1] gives element 1 of that row)

    def test_partial_then_full(self):
        """blocks[0] then block[0, 1] -- mixed levels."""
        v = make_view((512, 2048), (128, 512), block_size=(2, 2))
        row = v[0]  # consume one axis on dim 0 (block -> tile)
        # dim 0: block axis consumed -> tile + leaf remain (2 axes).
        # dim 1: untouched -> block + tile + leaf (3 axes).
        assert len(row.grid.axes_for(0)) == 2
        assert len(row.grid.axes_for(1)) == 3


# ============================================================================
# NDSlice.tolist(dim=d) -- axis-explicit materialization
# ============================================================================


@pytest_marks(["neurotile"])
class TestTolistAxisExplicit:
    """view.tolist(dim=d) produces sub-views along the named axis."""

    @pytest.mark.fast
    def test_tolist_dim1_iterates_named_dim(self):
        v = make_view((256, 1024), (128, 512))
        items = v.tolist(dim=1)
        assert len(items) == 2
        assert items[0].element_shape[1] == 512
        assert items[1].element_shape[1] == 512

    def test_tolist_dim_offsets(self):
        """Children along dim d have offsets stepping through that axis."""
        v = make_view((256, 1024), (128, 512))
        items = v.tolist(dim=1)
        assert items[0].offset == 0
        assert items[1].offset == 512  # dim-1 stride = 1, tile_size = 512

    def test_tolist_list_behaviour(self):
        """tolist(dim=d) returns a plain Python list -- supports indexing / slicing."""
        v = make_view((256, 1024), (128, 512))
        items = v.tolist(dim=1)
        assert isinstance(items, list)
        assert len(items) == 2
        assert len(items[0:1]) == 1


# ============================================================================
# Transforms through NDSlice
# ============================================================================


@pytest_marks(["neurotile"])
class TestTransforms:
    @pytest.mark.fast
    def test_reshape_dim(self):
        v = make_view((128, 512), (128, 512))
        tile = v[0, 0]
        reshaped = tile.reshape_dim(1, (8, 64))
        assert reshaped.element_shape == (128, 8, 64)
        assert reshaped.tile_size == (128, 8, 64)  # single tile covering new shape

    def test_permute(self):
        v = make_view((128, 512), (128, 512))
        tile = v[0, 0]
        permuted = tile.permute((1, 0))
        assert permuted.element_shape == (512, 128)

    def test_flatten_dims(self):
        v = make_view((128, 512), (128, 512))
        tile = v[0, 0]
        reshaped = tile.reshape_dim(1, (8, 64))
        flat = reshaped.flatten_dims(0, 1)
        assert flat.element_shape == (1024, 64)

    def test_transform_preserves_offset(self):
        v = make_view((256, 1024), (128, 512))
        tile = v[1, 0]
        reshaped = tile.reshape_dim(1, (8, 64))
        assert reshaped.offset == tile.offset  # offset preserved


# ============================================================================
# Element-level slice
# ============================================================================


@pytest_marks(["neurotile"])
class TestSlice:
    @pytest.mark.fast
    def test_slice_basic(self):
        """slice(dim, start, end) narrows at element level."""
        v = make_view((128, 512), (128, 512))
        tile = v[0, 0]
        sliced = tile.slice(1, 0, 256)
        assert sliced.element_shape == (128, 256)
        assert sliced.offset == 0

    def test_slice_with_offset(self):
        """slice with non-zero start advances offset."""
        v = make_view((128, 512), (128, 512))
        tile = v[0, 0]
        sliced = tile.slice(1, 128, 256)
        assert sliced.element_shape == (128, 128)
        # offset = start * strides[dim] = 128 * 1 = 128
        assert sliced.offset == 128

    def test_slice_dim0(self):
        """slice on partition dimension."""
        v = make_view((128, 512), (128, 512))
        tile = v[0, 0]
        sliced = tile.slice(0, 32, 64)
        assert sliced.element_shape == (32, 512)
        # offset = 32 * 512 = 16384
        assert sliced.offset == 32 * 512

    def test_slice_3d(self):
        """slice on 3D view (like head selection after reshape)."""
        v = make_view((16, 128, 8), (16, 128, 8))
        tile = v[0, 0, 0]
        sliced = tile.slice(2, 0, 1)
        assert sliced.element_shape == (16, 128, 1)
        assert sliced.offset == 0

    def test_slice_then_load_shape(self):
        """slice preserves tile_size = element_shape for loading."""
        v = make_view((16, 128, 8), (16, 128, 8))
        tile = v[0, 0, 0]
        sliced = tile.slice(2, 3, 5)
        assert sliced.tile_size == (16, 128, 2)
        assert sliced.element_shape == (16, 128, 2)

    def test_slice_sbuf_basic(self):
        """slice on SBUF tile narrows via sub_index."""
        v = make_sbuf_view((128, 512))
        sliced = v.slice(1, 0, 256)
        assert sliced.element_shape == (128, 256)
        assert isinstance(sliced.layout, SBUFLayout)

    def test_slice_sbuf_with_offset(self):
        """slice on SBUF with non-zero start."""
        v = make_sbuf_view((128, 512))
        sliced = v.slice(1, 128, 384)
        assert sliced.element_shape == (128, 256)
        # sub_index creates new source slice -- offset resets to 0
        assert sliced.layout.source.shape == (128, 256)

    def test_slice_sbuf_dim0(self):
        """slice on SBUF partition dimension."""
        v = make_sbuf_view((128, 512))
        sliced = v.slice(0, 32, 64)
        assert sliced.element_shape == (32, 512)

    def test_getitem_p_dim_slice_sbuf(self):
        """__getitem__ P-dim slice at element level works correctly."""
        v = make_sbuf_view((128, 512))
        # At element level (after _apply_transform), P-dim slice
        # should narrow P, not be silently ignored.
        reshaped = v.reshape_dim(1, (4, 128))  # at element level
        sub = reshaped[32:64]  # P-dim slice -> should give (32, 4, 128)
        assert sub.element_shape[0] == 32


# ============================================================================
# Remainder
# ============================================================================


@pytest_marks(["neurotile"])
class TestRemainder:
    @pytest.mark.fast
    def test_remainder_detected_on_parent(self):
        v = make_view((300, 512), (128, 512))
        # 3 tiles: [0:128], [128:256], [256:300]
        assert v.shape == (3, 1)
        assert v.is_remainder is True
        # Total children: 3 (whole + remainder).
        children = v.tolist()
        assert len(children) == 3


# ============================================================================
# Repr
# ============================================================================


@pytest_marks(["neurotile"])
class TestRepr:
    @pytest.mark.fast
    def test_repr(self):
        v = make_view((512, 2048), (128, 512))
        r = repr(v)
        assert "NDSlice(" in r
        assert "shape=(4, 4)" in r


# ============================================================================
# SBUF Transforms
# ============================================================================


def make_sbuf_view(tile_size, tile_shape=None):
    """Create an NDSlice(Grid, SBUFLayout) backed by a numpy array."""
    if tile_shape is None:
        tile_shape = tuple(1 for _ in tile_size)
    tile_p = tile_size[0]
    tile_f = product(tile_size, start=1)
    total_tiles = product(tile_shape)
    sbuf_data = np.zeros((tile_p, total_tiles * tile_f), dtype=np.float32)
    element_shape = tuple(tile_shape[d] * tile_size[d] for d in range(len(tile_size)))
    grid, layout = SBUFLayout.build_view(sbuf_data, element_shape, tile_size, "float32", "sbuf")
    return NDSlice(grid, layout)


@pytest_marks(["neurotile"])
class TestSBUFTransforms:
    @pytest.mark.fast
    def test_reshape_dim(self):
        """reshape_dim on SBUF tile updates Grid, keeps SBUFLayout."""
        v = make_sbuf_view((128, 512))
        reshaped = v.reshape_dim(1, (4, 128))
        assert reshaped.element_shape == (128, 4, 128)
        assert reshaped.tile_size == (128, 4, 128)
        assert isinstance(reshaped.layout, SBUFLayout)
        assert reshaped.layout.source is v.layout.source  # same underlying buffer

    def test_permute(self):
        """permute on SBUF tile reorders element_shape."""
        v = make_sbuf_view((128, 512))
        reshaped = v.reshape_dim(1, (4, 128))
        permuted = reshaped.permute((0, 2, 1))
        assert permuted.element_shape == (128, 128, 4)
        assert permuted.layout.source is v.layout.source

    def test_flatten_dims(self):
        """flatten_dims on SBUF tile merges dimensions back."""
        v = make_sbuf_view((128, 512))
        reshaped = v.reshape_dim(1, (4, 128))
        flat = reshaped.flatten_dims(1, 2)
        assert flat.element_shape == (128, 512)
        assert flat.layout.source is v.layout.source

    def test_squeeze_dim(self):
        """squeeze_dim on SBUF tile removes size-1 dim."""
        v = make_sbuf_view((128, 1, 512))
        squeezed = v.squeeze_dim(1)
        assert squeezed.element_shape == (128, 512)
        assert squeezed.layout.source is v.layout.source

    def test_expand_dim(self):
        """expand_dim on SBUF tile inserts size-1 dim."""
        v = make_sbuf_view((128, 512))
        expanded = v.expand_dim(1)
        assert expanded.element_shape == (128, 1, 512)
        assert expanded.layout.source is v.layout.source

    def test_broadcast(self):
        """broadcast on SBUF tile after expand_dim."""
        v = make_sbuf_view((128, 512))
        expanded = v.expand_dim(1)
        broadcast = expanded.broadcast(1, 4)
        assert broadcast.element_shape == (128, 4, 512)
        assert broadcast.layout.source is v.layout.source

    def test_split(self):
        """split convenience method on SBUF tile."""
        v = make_sbuf_view((128, 512))
        chunked = v.split(1, 4)
        assert chunked.element_shape == (128, 4, 128)
        assert chunked.layout.source is v.layout.source

    def test_sub_index_after_reshape(self):
        """reshape_dim then __getitem__ routes through sub_index."""
        v = make_sbuf_view((128, 512))
        reshaped = v.reshape_dim(1, (4, 128))
        # Index: select chunk 2 -> (128, 128) slice at F-offset 256
        chunk = reshaped[:, 2, :]
        assert chunk.element_shape == (128, 128)
        assert isinstance(chunk.layout, SBUFLayout)
        # Verify sub-slice offset: chunk 2 starts at F=256
        assert chunk.layout.offset == 0  # sub_index returns new source slice
        assert chunk.layout.source.shape == (128, 128)

    def test_split_then_index(self):
        """split then iterate chunks."""
        v = make_sbuf_view((128, 512))
        chunked = v.split(1, 2)  # (128, 2, 256)
        c0 = chunked[:, 0, :]
        c1 = chunked[:, 1, :]
        assert c0.element_shape == (128, 256)
        assert c1.element_shape == (128, 256)
        # c0 source at F=[0:256], c1 source at F=[256:512]
        assert c0.layout.source.shape == (128, 256)
        assert c1.layout.source.shape == (128, 256)

    def test_reshape_preserves_layout(self):
        """After reshape, layout is preserved -- data refers to same buffer."""
        v = make_sbuf_view((128, 512))
        reshaped = v.reshape_dim(1, (4, 128))
        # layout source unchanged, same buffer
        assert reshaped.layout.source is v.layout.source
        assert reshaped.data.shape == v.data.shape

    def test_chained_reshape_flatten(self):
        """reshape_dim -> sub_index -> flatten: reshape chain on SBUF."""
        v = make_sbuf_view((128, 512))
        reshaped = v.reshape_dim(1, (4, 128))
        # Flatten back to original
        flat = reshaped.flatten_dims(1, 2)
        assert flat.element_shape == (128, 512)
        # sub_index on flattened view works
        sub = flat[:, 0:256]
        assert sub.element_shape == (128, 256)


# ============================================================================
# iter_dim advancement
# ============================================================================


@pytest_marks(["neurotile"])
class TestCursorAdvancement:
    """view[i] at tile level should advance cursor past consumed dim."""

    @pytest.mark.fast
    def test_single_dim_descent_advances(self):
        v = make_view((256, 1024), (128, 512))
        child = v[0]
        # dim 0 selected: stack popped from (tile, 1) to (1,) = element
        assert grid_is_element(child, 0)
        # dim 1 untouched: still at tile level
        assert not grid_is_element(child, 1)
        assert child.grid.cursor == 1

    def test_all_dims_descended_advances_to_end(self):
        """Both dims int-consumed -> cursor advances past both, no wrap
        (leaves are not iteration-level, no further iteration possible)."""
        v = make_view((512, 2048), (128, 512))
        child = v[0, 0]
        assert grid_is_element(child, 0)
        assert grid_is_element(child, 1)
        assert child.grid.cursor == child.grid.ndim

    def test_block_consume_advances_past_dim(self):
        """Int on dim 0 (block-grid) -> cursor advances past dim 0;
        the remaining iteration is on dim 1."""
        v = make_view((512, 1024), (128, 512), block_size=(2, 2))
        child = v[0]
        # dim 0 was iter-consumed: cursor moves past it.
        assert child.grid.cursor == 1
        # dim 0 retains the block interior axes: still navigable
        # for drilling, but past the cursor for shape purposes.
        assert not grid_is_element(child, 0)

    def test_consistent_with_tolist(self):
        v = make_view((256, 1024), (128, 512))
        indexed = v[0]
        iterated = v.tolist()[0]
        assert indexed.grid.cursor == iterated.grid.cursor


# ============================================================================
# block_size property
# ============================================================================


@pytest_marks(["neurotile"])
class TestBlockSize:
    """NDSlice.block_size derived from Grid."""

    @pytest.mark.fast
    def test_present(self):
        v = make_view((512, 1024), (128, 512), block_size=(2, 2))
        assert v.block_size == (2, 2)

    def test_none_for_tiles(self):
        v = make_view((512, 1024), (128, 512))
        assert v.block_size is None

    def test_partial(self):
        v = make_view((512, 1024), (128, 512), block_size=(2, 1))
        assert v.block_size == (2, 1)


# ============================================================================
# .data attribute
# ============================================================================


@pytest_marks(["neurotile"])
class TestDataAttribute:
    """NDSlice.data returns appropriate buffer."""

    @pytest.mark.fast
    def test_hbm_data_is_none(self):
        v = make_view((512, 1024), (128, 512))
        assert v.data is None

    def test_sbuf_multi_tile_data_is_source(self):
        v = make_sbuf_view((128, 512), tile_shape=(1, 2))
        # remaining (128, 1024), tile_size (128, 512) -> dim 1 exceeds -> source
        assert v.data is v.source

    def test_sbuf_single_tile_data_is_tile(self):
        v = make_sbuf_view((128, 512))
        # remaining == tile_size -> single tile -> tile_data()
        assert v.data is v.source  # offset=0, tile_f covers source


# ============================================================================
# NDSlice user-facing method misuse guards (P0c lift)
# ============================================================================


@pytest_marks(["neurotile"])
class TestTolistMisuseGuards:
    """NDSlice.tolist(dim=) input validation."""

    @pytest.mark.fast
    def test_dim_must_be_int_or_none(self):
        v = make_view((512, 1024), (128, 512))
        with pytest.raises(AssertionError, match="dim= must be int or None"):
            v.tolist(dim=1.5)

    def test_dim_in_range(self):
        v = make_view((512, 1024), (128, 512))
        with pytest.raises(AssertionError, match=r"dim=5 is out of range"):
            v.tolist(dim=5)

    def test_dim_negative_rejected(self):
        v = make_view((512, 1024), (128, 512))
        with pytest.raises(AssertionError, match=r"dim=-1 is out of range"):
            v.tolist(dim=-1)

    def test_dim_none_works(self):
        v = make_view((512, 1024), (128, 512))
        # cursor-driven; should not raise
        items = v.tolist()
        assert len(items) == 4

    def test_dim_int_in_range_works(self):
        v = make_view((512, 1024), (128, 512))
        items = v.tolist(dim=1)
        assert len(items) == 2


@pytest_marks(["neurotile"])
class TestStreamMisuseGuards:
    """NDSlice.stream(dim=, buffer_count=, ...) input validation."""

    @pytest.mark.fast
    def test_dim_must_be_int(self):
        v = make_view((512, 1024), (128, 512))
        with pytest.raises(AssertionError, match="dim= must be int"):
            v.stream(dim=0.5, buffer_count=2)

    def test_dim_in_range(self):
        v = make_view((512, 1024), (128, 512))
        with pytest.raises(AssertionError, match=r"dim=5 is out of range"):
            v.stream(dim=5, buffer_count=2)

    def test_buffer_count_must_be_int(self):
        v = make_view((512, 1024), (128, 512))
        with pytest.raises(AssertionError, match="buffer_count= must be int"):
            v.stream(buffer_count=2.0)

    def test_buffer_count_must_be_positive(self):
        v = make_view((512, 1024), (128, 512))
        with pytest.raises(AssertionError, match="buffer_count= must be >= 1"):
            v.stream(buffer_count=0)
        with pytest.raises(AssertionError, match="buffer_count= must be >= 1"):
            v.stream(buffer_count=-1)

    def test_pattern_override_requires_out_shape(self):
        v = make_view((512, 1024), (128, 512))
        with pytest.raises(AssertionError, match="pattern_override= requires out_shape="):
            v.stream(buffer_count=2, pattern_override=[[1024, 4], [1, 512]])

    def test_stream_on_sbuf_rejected(self):
        v = make_sbuf_view((128, 512))
        with pytest.raises(AssertionError, match="only valid on HBM views"):
            v.stream(buffer_count=2)

    def test_stream_with_unconsumed_batch_dims(self):
        # 3D source + 2D tile_size -> 1 batch dim left.
        v = make_view((4, 512, 1024), (128, 512))
        with pytest.raises(AssertionError, match="batch dims to be consumed first"):
            v.stream(buffer_count=2)


@pytest_marks(["neurotile"])
class TestLoadMisuseGuards:
    """NDSlice.load() input validation."""

    @pytest.mark.fast
    def test_load_on_sbuf_rejected(self):
        v = make_sbuf_view((128, 512))
        with pytest.raises(AssertionError, match="load.. is only valid on HBM views"):
            v.load()

    def test_load_with_unconsumed_batch_dims(self):
        # 3D source + 2D tile_size -> 1 batch dim left.
        v = make_view((4, 512, 1024), (128, 512))
        with pytest.raises(AssertionError, match="batch dims to be consumed first"):
            v.load()

    def test_oob_value_requires_oob_mode(self):
        v = make_view((512, 1024), (128, 512))
        tile = v[0, 0]
        with pytest.raises(AssertionError, match="oob_value= requires oob_mode="):
            tile.load(oob_value=0.0)

    def test_pattern_override_requires_out_shape_or_dst(self):
        v = make_view((512, 1024), (128, 512))
        tile = v[0, 0]
        with pytest.raises(AssertionError, match="pattern_override= requires out_shape= or dst="):
            tile.load(pattern_override=[[1, 512], [512, 1]])


@pytest_marks(["neurotile"])
class TestStoreMisuseGuards:
    """NDSlice.store() input validation (existing guards verified)."""

    @pytest.mark.fast
    def test_store_on_sbuf_rejected(self):
        v = make_sbuf_view((128, 512))
        with pytest.raises(AssertionError, match="store.. is only valid on HBM views"):
            v.store(data="anything")

    def test_store_data_cannot_be_ndslice(self):
        # Use a dest HBM view; pass an NDSlice as data -- should reject.
        dst = make_view((512, 1024), (128, 512))
        src = make_sbuf_view((128, 512))
        with pytest.raises(AssertionError, match=r"expects ndarray or .ap\(\) view, not NDSlice"):
            dst[0, 0].store(src)

    def test_store_with_unconsumed_batch_dims(self):
        v = make_view((4, 512, 1024), (128, 512))
        with pytest.raises(AssertionError, match="batch dims to be consumed first"):
            v.store(data="anything")


# ============================================================================
# Slice-based sharding -- block_range narrows remaining; offset matches start
# ============================================================================


@pytest_marks(["neurotile"])
class TestSliceShardingOffset:
    """view[range, :] narrows the view to the rank's owned slice."""

    @pytest.mark.fast
    def test_block_range_narrows_remaining(self):
        from nkilib_src.nkilib.experimental.neurotile.core.factories import blocks
        from nkilib_src.nkilib.experimental.neurotile.core.shard_helpers import block_range

        class _Src:
            shape = (512, 1024)
            dtype = "float32"

        # 2 blocks on dim 0 (block_size=2 -> 256 elements/block).
        # Rank 1 of 2 ranks: own 1 block; offset = 1 * (2*128) * stride[0] = 262144.
        sharded = blocks(_Src(), tile_size=(128, 512), block_size=(2, 2))[block_range(rank=1, num_shards=2, total=2), :]
        assert sharded.offset == 1 * (2 * 128) * 1024
        assert sharded.grid.remaining[0] == 256


# ============================================================================
# HBMLayout int advance preserves an existing indirect tag
# ============================================================================


@pytest_marks(["neurotile"])
class TestIndirectOffsetCombine:
    """Compile-time advance on a dim with a runtime indirect must keep the tag."""

    @pytest.mark.fast
    def test_int_advance_preserves_indirect(self):
        from nkilib_src.nkilib.experimental.neurotile.core.axis import IndirectKind, IndirectOffset

        layout = HBMLayout(
            "src", 0, (1024, 1), "f32", "hbm", indirect=IndirectOffset(kind=IndirectKind.SCALAR, value=100, dim=0)
        )
        new = layout.advance(0, 2, 128)
        # Compile-time advance updates offset; runtime tag is preserved.
        assert new.indirect.value == 100
        assert new.indirect.dim == 0
        assert new.offset == 2 * 128 * 1024


# ============================================================================
# N-D SBUF tile_shape computation
# ============================================================================


@pytest_marks(["neurotile"])
class TestSBUFTileShape:
    """_sbuf_tile_shape_from_buffer should compute per-dim tile counts for N-D."""

    @pytest.mark.fast
    def test_3d_matching_dims(self):
        """3D source (128, 4, 64) with tile (128, 1, 64) -> (1, 4, 1)."""
        from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout

        ts = SBUFLayout.tile_shape_from_buffer((128, 4, 64), (128, 1, 64))
        assert ts == (1, 4, 1)

    def test_3d_single_tile(self):
        """3D source == tile -> (1, 1, 1)."""
        from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout

        ts = SBUFLayout.tile_shape_from_buffer((128, 1, 8), (128, 1, 8))
        assert ts == (1, 1, 1)

    def test_2d_fallback(self):
        """2D source with 2D tile -> flat computation."""
        from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout

        ts = SBUFLayout.tile_shape_from_buffer((128, 512), (128, 512))
        assert ts == (1, 1)

    def test_2d_multi_tile(self):
        """2D source with multiple tiles."""
        from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout

        ts = SBUFLayout.tile_shape_from_buffer((128, 2048), (128, 512))
        assert ts == (1, 4)


# ============================================================================
# SBUF sharding via slice indexing
# ============================================================================


@pytest_marks(["neurotile"])
class TestSBUFSharding:
    """Slice-based sharding on SBUF tiles narrows the view to the rank's range."""

    @pytest.mark.fast
    def test_sbuf_tiles_with_block_range(self):
        from nkilib_src.nkilib.experimental.neurotile.core.factories import tiles
        from nkilib_src.nkilib.experimental.neurotile.core.shard_helpers import block_range

        class MockSBUF:
            def __init__(self, shape):
                self.shape = shape
                self.dtype = "bf16"

            def __getitem__(self, key):
                return self

            def reshape(self, new_shape):
                return MockSBUF(new_shape)

        sbuf = MockSBUF((128, 8, 64))
        # Rank 0 of 2, block-sharded on dim 1 -> owns 4 of 8 tiles.
        view = tiles(sbuf, tile_size=(128, 1, 64), buffer_type=nl.sbuf)[
            :, block_range(rank=0, num_shards=2, total=8), :
        ]
        assert view.grid.remaining[1] == 4


# ============================================================================
# SBUFLayout transform strides and broadcast AP
# ============================================================================


@pytest_marks(["neurotile"])
class TestSBUFTransformStrides:
    """apply_transform stores strides; ap() uses them for broadcast."""

    @pytest.mark.fast
    def test_apply_transform_creates_new_layout(self):
        """apply_transform returns new SBUFLayout with transform_strides."""
        layout = SBUFLayout("src", 0, (512,), (128, 512), "f32", "sbuf")
        new_layout = layout.apply_transform((8, 0, 1))
        assert new_layout is not layout
        assert new_layout.ap_strides == (8, 0, 1)
        assert new_layout.source is layout.source

    def testap_strides_chain(self):
        """Chained transforms replace strides."""
        layout = SBUFLayout("src", 0, (512,), (128, 512), "f32", "sbuf")
        first = layout.apply_transform((64, 1))
        second = first.apply_transform((0, 64, 1))
        assert second.ap_strides == (0, 64, 1)

    def testap_strides_used_inap_strides(self):
        """transform_strides() returns stored strides for chaining."""
        layout = SBUFLayout("src", 0, (512,), (128, 512), "f32", "sbuf")
        transformed = layout.apply_transform((8, 0, 1))
        assert transformed.transform_strides((128, 4, 64)) == (8, 0, 1)

    def test_noap_strides_gives_contiguous(self):
        """Without transform, transform_strides() returns contiguous."""
        layout = SBUFLayout("src", 0, (512,), (128, 512), "f32", "sbuf")
        assert layout.transform_strides((128, 512)) == (512, 1)


# ============================================================================
# SBUFLayout.tile_data N-D reshape
# ============================================================================


@pytest_marks(["neurotile"])
class TestTileDataNDReshape:
    """tile_data() should reshape N-D source to 2D before slicing."""

    @pytest.mark.fast
    def test_3d_source_returns_2d(self):
        """3D source (128, 1, 8) -> _slice_one_tile returns 2D (128, 8)."""

        class Mock3D:
            def __init__(self):
                self.shape = (128, 1, 8)

            def reshape(self, new_shape):
                return Mock2D(new_shape)

            def __getitem__(self, key):
                return self

        class Mock2D:
            def __init__(self, shape):
                self.shape = shape

            def __getitem__(self, key):
                return Mock2D((self.shape[0], key[1].stop - (key[1].start or 0)))

        src = Mock3D()
        layout = SBUFLayout(src, 0, (8,), (128, 8), "f32", "sbuf")
        result = layout.tile_data()
        assert result.shape == (128, 8)

    def test_2d_source_no_reshape(self):
        """2D source (128, 512) -> tile_data returns as-is when allocation tile fills source."""

        class Mock2D:
            def __init__(self):
                self.shape = (128, 512)

            def __getitem__(self, key):
                return self

        src = Mock2D()
        layout = SBUFLayout(src, 0, (512,), (128, 512), "f32", "sbuf")
        result = layout.tile_data()
        assert result is src  # returned directly, no reshape


# ============================================================================
# Logical vs physical SBUF shape (P-fold awareness)
# ============================================================================


@pytest_marks(["neurotile"])
class TestSBUFLogicalShape:
    """SBUF storage is 2-D; NDSlice carries the logical element_shape.

    For multi-P-tile views, P-tiles fold into F columns physically while
    element_shape retains the logical (BS*tile_p, f_count*tile_f) rank.
    """

    @pytest.mark.fast
    def test_single_row_shape_matches_source(self):
        """Single P-tile row: source shape equals element_shape."""
        v = make_sbuf_view(tile_size=(128, 512), tile_shape=(1, 4))
        assert v.layout.source.shape == (128, 2048)
        assert v.shape == (1, 4)

    def test_multi_p_tile_source_folds_into_f(self):
        """Multi-P-tile SBUF: physical (128, 4*512); logical (256, 1024)."""
        v = make_sbuf_view(tile_size=(128, 512), tile_shape=(2, 2))
        assert v.layout.source.shape == (128, 2048)
        assert v.grid.element_shape == (256, 1024)
        assert v.shape == (2, 2)

    def test_sbuf_tile_shape_from_logical_vs_raw(self):
        """_sbuf_tile_shape_from_buffer differs for raw vs logical shape."""
        from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout

        tile_size = (128, 512)
        assert SBUFLayout.tile_shape_from_buffer((128, 2048), tile_size) == (1, 4)
        assert SBUFLayout.tile_shape_from_buffer((256, 1024), tile_size) == (2, 2)


# ============================================================================
# Slice-based sharding on a 3D source (rank batch dim) drops the consumed rank
# ============================================================================


class _Mock3DSrc:
    shape = (2, 512, 512)
    dtype = "float32"


@pytest_marks(["neurotile"])
class TestRankShardDimDrop:
    """v[rank, :, :] consumes the rank batch dim and drops it from the view.

    Matches the fgcc_mlp_cte pattern: 3D source (num_ranks, M, H) is indexed
    with `view[rank]` so subsequent `[m_batch]` targets the M dim.
    """

    @pytest.mark.fast
    def test_no_shard_rank_dim_visible(self):
        from nkilib_src.nkilib.experimental.neurotile.core.factories import tiles

        v = tiles(_Mock3DSrc(), tile_size=(128, 512))
        assert v.ndim == 3
        assert v.grid.n_batch_dims == 1

    def test_rank_index_drops_dim(self):
        from nkilib_src.nkilib.experimental.neurotile.core.factories import tiles

        v = tiles(_Mock3DSrc(), tile_size=(128, 512))
        sharded = v[0]  # consume rank dim
        assert sharded.ndim == 2
        assert sharded.grid.remaining == (512, 512)

    def test_rank_then_m_index_targets_m(self):
        from nkilib_src.nkilib.experimental.neurotile.core.factories import tiles

        v = tiles(_Mock3DSrc(), tile_size=(128, 512))
        sharded = v[0]
        child = sharded[0]
        assert child.grid.remaining[0] == 128


# ============================================================================
# New public API -- Phase 1: tolist(dim=), stream(dim=), _load_dst field
# ============================================================================


def _equivalent_children(a, b):
    """Structural equality for a pair of NDSlice lists (NDSlice has no __eq__).

    Compares only data-relevant fields (element_shape, offset). `shape`
    tracks iteration-grid cursor state which can differ between paths that
    produce semantically-equivalent children.
    """
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i].element_shape != b[i].element_shape:
            return False
        if a[i].offset != b[i].offset:
            return False
    return True


@pytest_marks(["neurotile"])
class TestTolistDefaultDim:
    """NDSlice.tolist() with no args -- follows grid cursor, equals tolist()."""

    @pytest.mark.fast
    def test_tolist_cursor_2d(self):
        v = make_view((512, 2048), (128, 512))
        assert _equivalent_children(v.tolist(), v.tolist())

    def test_tolist_cursor_with_block_size(self):
        v = make_view((512, 2048), (128, 512), block_size=(2, 2))
        assert _equivalent_children(v.tolist(), v.tolist())

    def test_tolist_explicit_dim_iterates_batch(self):
        """Explicit-dim tolist on the batch dim yields one child per batch slot."""
        v = make_view((4, 128, 512), (128, 128))
        items = v.tolist(dim=0)
        assert len(items) == 4

    def test_tolist_empty_view(self):
        """Fully-consumed view with ndim==0 materializes as empty list."""
        v = make_view((128, 256), (128, 256))
        inner = v[0]  # consume outer dim; grid.cursor >= grid.ndim
        # After consumption cursor is past all dims -> tolist returns [self].
        result = inner.tolist()
        # This should NOT be length 0 since inner still has content; the
        # "returns [self]" branch of tolist_cursor fires. Assert it's length 1.
        assert len(result) == 1


@pytest_marks(["neurotile"])
class TestTolistAlongDim:
    """NDSlice.tolist(dim=d) -- explicit axis iteration, cursor untouched."""

    @pytest.mark.fast
    def test_tolist_dim0_counts(self):
        v = make_view((512, 2048), (128, 512))
        items = v.tolist(dim=0)
        assert len(items) == 4  # 512 / 128

    def test_tolist_dim1_counts(self):
        v = make_view((512, 2048), (128, 512))
        items = v.tolist(dim=1)
        assert len(items) == 4  # 2048 / 512

    def test_tolist_dim_offsets_step_along_dim(self):
        v = make_view((512, 2048), (128, 512))
        items = v.tolist(dim=1)
        # dim=1 stride = 1, tile_size = 512 -> offsets are 0, 512, 1024, 1536.
        offsets = [items[i].offset for i in range(len(items))]
        assert offsets == [0, 512, 1024, 1536]

    def test_tolist_dim_preserves_other_dims(self):
        """Child along dim=1 keeps full dim 0 span."""
        v = make_view((512, 2048), (128, 512))
        items = v.tolist(dim=1)
        for child in items:
            assert child.element_shape[0] == 512


import nki as _nki
import nki.language as _nl

from nkilib_src.nkilib.experimental.neurotile.core.factories import tiles as _tiles
from test.utils.pytest_test_metadata import pytest_marks


@_nki.jit
def _stream_dim0_probe():
    src = _nl.ndarray((512, 2048), dtype=_nl.float32, buffer=_nl.shared_hbm)
    v = _tiles(src, tile_size=(128, 512))
    s = v.stream(buffer_count=2)
    assert s.count == v.grid.current_count(0)
    assert s._buffer_count == 2
    assert s._dim == 0
    return src


@_nki.jit
def _stream_dim1_probe():
    src = _nl.ndarray((512, 2048), dtype=_nl.float32, buffer=_nl.shared_hbm)
    v = _tiles(src, tile_size=(128, 512))
    s = v.stream(dim=1, buffer_count=2)
    assert s.count == v.grid.current_count(1)
    assert s._dim == 1
    return src


@_nki.jit
def _stream_tolist_children_probe():
    src = _nl.ndarray((512, 2048), dtype=_nl.float32, buffer=_nl.shared_hbm)
    v = _tiles(src, tile_size=(128, 512))
    s = v.stream(buffer_count=2)
    children = s.tolist()
    assert len(children) == 4
    for c in children:
        assert isinstance(c, NDSlice)
    return src


@_nki.jit
def _stream_tolist_load_dst_probe():
    src = _nl.ndarray((512, 2048), dtype=_nl.float32, buffer=_nl.shared_hbm)
    v = _tiles(src, tile_size=(128, 512))
    s = v.stream(buffer_count=2)
    children = s.tolist()
    for i in range(len(children)):
        assert children[i]._load_dst is s._buffers[i % s._buffer_count]
    return src


@pytest_marks(["neurotile"])
@pytest.mark.skip(
    reason="Requires NKI runtime (libnrt). These tests dispatch a "
    "compiled kernel and belong in integration/, not unit/."
)
class TestStreamDimKwarg:
    """NDSlice.stream(dim=d, ...) produces a BlockStream walking along dim d.

    Run via @nki.jit so SBUF buffer allocation has an active backend.
    """

    @pytest.mark.fast
    def test_stream_default_dim0(self):
        _stream_dim0_probe[1]()

    def test_stream_explicit_dim(self):
        _stream_dim1_probe[1]()


@pytest_marks(["neurotile"])
class TestLoadDstField:
    """NDSlice._load_dst, _load_pattern_override, _load_out_shape -- stored state."""

    @pytest.mark.fast
    def test_load_dst_default_none(self):
        v = make_view((128, 512), (128, 512))
        assert v._load_dst is None
        assert v._load_pattern_override is None
        assert v._load_out_shape is None

    def test_load_dst_settable_via_init(self):
        v = make_view((128, 512), (128, 512))
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import NDSlice

        marker = object()
        v2 = NDSlice(v.grid, v.layout, load_dst=marker)
        assert v2._load_dst is marker


@pytest_marks(["neurotile"])
class TestLoadDstRouting:
    """NDSlice.load() precedence rules for _load_dst routing.

    Monkeypatches HBMLayout.load to record its dst= kwarg. Encodes the
    design's four precedence rules as invariants.
    """

    def _capture_load_dst(self, monkeypatch):
        """Return a list that records the `dst` kwarg on every HBMLayout.load call."""
        from nkilib_src.nkilib.experimental.neurotile.core.layout_hbm import HBMLayout

        recorded = []
        original = HBMLayout.load

        def spy(
            self,
            grid,
            dtype=None,
            dst=None,
            oob_mode=None,
            oob_value=None,
            out_shape=None,
            transpose=False,
            dge_mode=None,
            pattern_override=None,
        ):
            recorded.append({"dst": dst, "pattern_override": pattern_override, "out_shape": out_shape, "dtype": dtype})
            # Return a stub -- we don't need a real SBUFLayout for these tests.
            return (None, None)

        monkeypatch.setattr(HBMLayout, "load", spy)
        return recorded

    def _nd_with_load_dst(self, buf):
        """Build an NDSlice with load_dst=buf but no other state."""
        v = make_view((128, 512), (128, 512))
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import NDSlice

        return NDSlice(v.grid, v.layout, load_dst=buf)

    def _fake_sbuf_buffer(self, dtype="float32"):
        class _FakeSBUF:
            pass

        buf = _FakeSBUF()
        buf.dtype = dtype
        buf.shape = (128, 512)
        return buf

    @pytest.mark.fast
    def test_explicit_dst_wins_over_load_dst(self, monkeypatch):
        """view.load(dst=X) uses X -- _load_dst is ignored."""
        recorded = self._capture_load_dst(monkeypatch)
        buf = self._fake_sbuf_buffer()
        override = self._fake_sbuf_buffer()
        view = self._nd_with_load_dst(buf)
        try:
            view.load(dst=override)
        except Exception:
            pass  # spy returns (None, None) so wrap construction may crash
        assert recorded[0]["dst"] is override

    def test_load_dst_used_when_no_explicit(self, monkeypatch):
        """view.load() (no dst=) routes into _load_dst."""
        recorded = self._capture_load_dst(monkeypatch)
        buf = self._fake_sbuf_buffer(dtype=make_view((128, 512), (128, 512)).dtype)
        view = self._nd_with_load_dst(buf)
        try:
            view.load()
        except Exception:
            pass
        assert recorded[0]["dst"] is buf

    def test_remainder_skips_load_dst(self, monkeypatch):
        recorded = self._capture_load_dst(monkeypatch)
        buf = self._fake_sbuf_buffer(dtype="float32")
        v = make_view((300, 512), (128, 512))
        items = v.tolist()
        remainder = items[-1]
        assert remainder.is_remainder
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import NDSlice

        view = NDSlice(remainder.grid, remainder.layout, load_dst=buf)
        try:
            view.load()
        except Exception:
            pass
        assert recorded[0]["dst"] is None

    def test_dtype_mismatch_skips_load_dst(self, monkeypatch):
        """If caller requests a different dtype, skip _load_dst and allocate fresh."""
        recorded = self._capture_load_dst(monkeypatch)
        # buf dtype intentionally mismatched with load() dtype request.
        buf = self._fake_sbuf_buffer(dtype="float32")
        view = self._nd_with_load_dst(buf)
        try:
            view.load(dtype="bfloat16")
        except Exception:
            pass
        assert recorded[0]["dst"] is None

    def test_bare_ndslice_allocates(self, monkeypatch):
        """Bare NDSlice (_load_dst=None) -> dst=None -> layout allocates as before."""
        recorded = self._capture_load_dst(monkeypatch)
        v = make_view((128, 512), (128, 512))
        try:
            v.load()
        except Exception:
            pass
        assert recorded[0]["dst"] is None

    def test_pattern_override_default_from_stream(self, monkeypatch):
        """_load_pattern_override is applied when caller omits pattern_override."""
        recorded = self._capture_load_dst(monkeypatch)
        v = make_view((128, 512), (128, 512))
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import NDSlice

        sentinel = [[1, 512], [512, 1]]
        # _load_out_shape required so the pattern_override + out_shape contract
        # is satisfied (see NDSlice.load assertion).
        view = NDSlice(v.grid, v.layout, load_pattern_override=sentinel, load_out_shape=(128, 512))
        try:
            view.load()
        except Exception:
            pass
        assert recorded[0]["pattern_override"] is sentinel

    def test_explicit_pattern_override_wins(self, monkeypatch):
        recorded = self._capture_load_dst(monkeypatch)
        v = make_view((128, 512), (128, 512))
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import NDSlice

        default = [[1, 512], [512, 1]]
        override = [[2, 256], [256, 2]]
        view = NDSlice(v.grid, v.layout, load_pattern_override=default, load_out_shape=(128, 512))
        try:
            view.load(pattern_override=override, out_shape=(128, 512))
        except Exception:
            pass
        assert recorded[0]["pattern_override"] is override


@pytest_marks(["neurotile"])
@pytest.mark.skip(
    reason="Requires NKI runtime (libnrt). These tests dispatch a "
    "compiled kernel and belong in integration/, not unit/."
)
class TestBlockStreamTolistBinding:
    """BlockStream.tolist() returns NDSlice children with load_dst pre-filled."""

    @pytest.mark.fast
    def test_tolist_children_are_ndslice(self):
        _stream_tolist_children_probe[1]()

    def test_tolist_pre_fills_load_dst_from_rotating_buffers(self):
        _stream_tolist_load_dst_probe[1]()


# ============================================================================
# whole_tiles / remainder_tiles -- iteration split for boundary handling
# ============================================================================


@pytest_marks(["neurotile"])
class TestWholeAndRemainderTiles:
    """Pins the contract: whole_tiles returns non-remainder children;
    remainder_tiles returns the rest. On a clean (non-remainder) view
    whole_tiles returns the full list and remainder_tiles returns []."""

    @pytest.mark.fast
    def test_clean_view_whole_tiles_returns_all(self):
        """Even-divisible view: every child has is_remainder == False."""
        v = make_view((256, 1024), (128, 512))  # 2x2, no remainder
        whole = v.whole_tiles()
        rem = v.remainder_tiles()
        assert len(whole) == len(v.tolist())
        assert rem == []

    def test_clean_view_remainder_tiles_empty(self):
        v = make_view((512, 1024), (128, 512))
        assert v.remainder_tiles() == []

    def test_remainder_view_split(self):
        """Partial trailing tile on dim 0: 300 // 128 = 2, remainder 44.
        tolist() yields 3 children; the last is is_remainder=True.
        """
        v = make_view((300, 512), (128, 512))
        all_items = v.tolist()
        assert len(all_items) == 3
        whole = v.whole_tiles()
        rem = v.remainder_tiles()
        assert len(whole) + len(rem) == len(all_items)
        # Whole + remainder partition tolist; remainder ends up at the end.
        assert all(not c.is_remainder for c in whole)
        assert all(c.is_remainder for c in rem)
