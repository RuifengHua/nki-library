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
Tests for HBMLayout: offset tracking, stride transforms, indirect state.

Pure Python -- no NKI, no tracer, no device required.
"""

import pytest

from nkilib_src.nkilib.experimental.neurotile.core._helpers import contiguous_ap_pattern as _contiguous_ap_pattern
from nkilib_src.nkilib.experimental.neurotile.core._helpers import contiguous_strides as _contiguous_strides
from nkilib_src.nkilib.experimental.neurotile.core.axis import IndirectKind, IndirectOffset
from nkilib_src.nkilib.experimental.neurotile.core.layout_hbm import HBMLayout
from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout


def _scalar_indirect(value, dim):
    return IndirectOffset(kind=IndirectKind.SCALAR, value=value, dim=dim)


def _vector_indirect(value, dim):
    return IndirectOffset(kind=IndirectKind.VECTOR, value=value, dim=dim)


import nki.language as nl

from nkilib_src.nkilib.experimental.neurotile.core.transforms import (
    compute_broadcast,
    compute_expand_dim,
    compute_flatten_dims,
    compute_fold,
    compute_permute,
    compute_reshape,
    compute_reshape_dim,
    compute_squeeze_dim,
)
from test.utils.pytest_test_metadata import pytest_marks

# ============================================================================
# Helpers
# ============================================================================


def make_layout(shape, offset=0):
    """Create HBMLayout with contiguous strides for given shape."""
    strides = _contiguous_strides(shape)
    return HBMLayout(
        source=None,  # not needed for offset/stride tests
        offset=offset,
        strides=strides,
        dtype="float32",
        buffer_type=nl.shared_hbm,
    )


# ============================================================================
# Contiguous strides helper
# ============================================================================


@pytest_marks(["neurotile"])
class TestContiguousStrides:
    @pytest.mark.fast
    def test_2d(self):
        assert _contiguous_strides((128, 512)) == (512, 1)

    def test_3d(self):
        assert _contiguous_strides((8, 128, 64)) == (8192, 64, 1)

    def test_1d(self):
        assert _contiguous_strides((256,)) == (1,)

    def test_4d(self):
        s = _contiguous_strides((2, 4, 8, 16))
        assert s == (512, 128, 16, 1)


# ============================================================================
# Advance -- concrete (int k)
# ============================================================================


@pytest_marks(["neurotile"])
class TestAdvanceConcrete:
    @pytest.mark.fast
    def test_tile_level(self):
        """Advance by 1 tile along dim 0, step=128."""
        layout = make_layout((512, 2048))
        # strides = (2048, 1)
        result = layout.advance(dim=0, k=1, step=128)
        # offset += 1 * 128 * 2048 = 262144
        assert result.offset == 262144
        assert result.strides == layout.strides  # unchanged

    def test_block_level(self):
        """Advance by 1 block along dim 0, step=256 (bs=2, ts=128)."""
        layout = make_layout((512, 2048))
        result = layout.advance(dim=0, k=1, step=256)
        # offset += 1 * 256 * 2048 = 524288
        assert result.offset == 524288

    def test_element_level(self):
        """Advance by 42 elements along dim 0, step=1."""
        layout = make_layout((128, 64))
        result = layout.advance(dim=0, k=42, step=1)
        # offset += 42 * 1 * 64 = 2688
        assert result.offset == 2688

    def test_dim1_advance(self):
        """Advance along dim 1."""
        layout = make_layout((512, 2048))
        result = layout.advance(dim=1, k=2, step=512)
        # offset += 2 * 512 * 1 = 1024
        assert result.offset == 1024

    def test_cumulative_offset(self):
        """Multiple advances accumulate offset."""
        layout = make_layout((512, 2048))
        r1 = layout.advance(dim=0, k=1, step=128)  # +262144
        r2 = r1.advance(dim=1, k=2, step=512)  # +1024
        assert r2.offset == 262144 + 1024

    def test_preserves_indirect(self):
        """Concrete advance preserves an existing IndirectOffset tag."""
        layout = HBMLayout(
            source=None,
            offset=0,
            strides=(2048, 1),
            dtype="float32",
            buffer_type=nl.shared_hbm,
            indirect=_scalar_indirect("some_scalar", 0),
        )
        result = layout.advance(dim=1, k=1, step=512)
        assert result.indirect.value == "some_scalar"
        assert result.indirect.dim == 0


# ============================================================================
# Advance -- indirect (non-int k)
# ============================================================================


@pytest_marks(["neurotile"])
class TestAdvanceIndirect:
    @pytest.mark.fast
    def test_step_1_no_scaling(self):
        """Runtime k with step=1: stored as-is, no scaling."""
        layout = make_layout((8, 128, 64))
        runtime_k = 5.0  # non-int triggers indirect path
        result = layout.advance(dim=0, k=runtime_k, step=1)
        assert result.indirect.value == 5.0
        assert result.indirect.dim == 0
        assert result.offset == 0

    def test_step_gt_1_scaled(self):
        """Runtime k with step>1: scaled by step (not strides)."""
        layout = make_layout((512, 2048))
        runtime_k = 3.0
        result = layout.advance(dim=0, k=runtime_k, step=128)
        # 3.0 * 128 = 384.0 (in source-element units; stride applied later).
        assert result.indirect.value == 384.0
        assert result.indirect.dim == 0
        assert result.offset == 0

    def test_indirect_dim_is_surviving_dim(self):
        """IndirectOffset dim is the dim parameter."""
        layout = make_layout((8, 128, 64))
        result = layout.advance(dim=1, k=2.0, step=128)
        assert result.indirect.dim == 1

    def test_indirect_replaces_vector(self):
        """Scalar advance replaces any pre-existing vector indirect."""
        layout = HBMLayout(
            source=None,
            offset=0,
            strides=(8192, 64, 1),
            dtype="float32",
            indirect=_vector_indirect("old_vector", 0),
        )
        result = layout.advance(dim=0, k=2.0, step=1)
        assert result.indirect.kind == IndirectKind.SCALAR
        assert result.indirect.value == 2.0


# ============================================================================
# Advance vector
# ============================================================================


@pytest_marks(["neurotile"])
class TestSetIndirectVector:
    @pytest.mark.fast
    def test_basic(self):
        layout = make_layout((512, 2048))
        vec = "mock_vector_tensor"
        result = layout.set_indirect(IndirectKind.VECTOR, vec, dim=0)
        assert result.indirect.kind == IndirectKind.VECTOR
        assert result.indirect.value == vec
        assert result.indirect.dim == 0


# ============================================================================
# Drop dim
# ============================================================================


@pytest_marks(["neurotile"])
class TestDropDim:
    @pytest.mark.fast
    def test_drop_dim0(self):
        """Drop dim 0: strides shrink, offset preserved."""
        layout = make_layout((8, 128, 64))
        # strides = (8192, 64, 1)
        result = layout.drop_dim(0)
        assert result.strides == (64, 1)
        assert result.offset == 0
        assert result.source is layout.source

    def test_drop_dim1(self):
        layout = make_layout((8, 128, 64))
        result = layout.drop_dim(1)
        assert result.strides == (8192, 1)

    def test_drop_preserves_indirect_dim(self):
        """IndirectOffset dim is source-relative -- drop_dim doesn't shift it."""
        layout = HBMLayout(
            source=None,
            offset=0,
            strides=(8192, 64, 1),
            dtype="float32",
            indirect=_scalar_indirect("eid", 0),
        )
        result = layout.drop_dim(0)
        assert result.indirect.dim == 0
        assert result.strides == (64, 1)

    def test_advance_then_drop(self):
        """Full iteration dim flow: advance(runtime) then drop."""
        layout = make_layout((8, 128, 64))
        advanced = layout.advance(dim=0, k=5.0, step=1)
        dropped = advanced.drop_dim(0)
        assert dropped.strides == (64, 1)
        assert dropped.indirect.value == 5.0
        assert dropped.indirect.dim == 0


# ============================================================================
# With strides (for transforms)
# ============================================================================


@pytest_marks(["neurotile"])
class TestApplyTransform:
    @pytest.mark.fast
    def test_basic(self):
        layout = make_layout((128, 512))
        new_layout = layout.apply_transform((1, 128))
        assert new_layout.strides == (1, 128)
        assert new_layout.offset == layout.offset
        assert new_layout.source is layout.source


# ============================================================================
# Stride transforms (standalone functions)
# ============================================================================


@pytest_marks(["neurotile"])
class TestComputeReshapeDim:
    @pytest.mark.fast
    def test_split_last_dim(self):
        """Split dim 1 of (128, 512) into (128, 8, 64)."""
        shape = (128, 512)
        strides = (512, 1)
        new_shape, new_strides = compute_reshape_dim(shape, strides, dim=1, new_sub_shape=(8, 64))
        assert new_shape == (128, 8, 64)
        assert new_strides == (512, 64, 1)

    def test_split_first_dim(self):
        """Split dim 0 of (128, 512) into (4, 32, 512)."""
        shape = (128, 512)
        strides = (512, 1)
        new_shape, new_strides = compute_reshape_dim(shape, strides, dim=0, new_sub_shape=(4, 32))
        assert new_shape == (4, 32, 512)
        assert new_strides == (16384, 512, 1)

    def test_split_middle_dim(self):
        """Split dim 1 of (4, 128, 64) into (4, 2, 64, 64)."""
        shape = (4, 128, 64)
        strides = (8192, 64, 1)
        new_shape, new_strides = compute_reshape_dim(shape, strides, dim=1, new_sub_shape=(2, 64))
        assert new_shape == (4, 2, 64, 64)
        assert new_strides == (8192, 4096, 64, 1)


@pytest_marks(["neurotile"])
class TestComputeReshape:
    @pytest.mark.fast
    def test_basic(self):
        """Full reshape (128, 512) -> (64, 1024)."""
        shape = (128, 512)
        strides = (512, 1)
        new_shape, new_strides = compute_reshape(shape, strides, (64, 1024))
        assert new_shape == (64, 1024)
        assert new_strides == (1024, 1)

    def test_preserves_base_stride(self):
        """Reshape on permuted view preserves non-unit base stride."""
        shape = (512, 128)
        strides = (1, 512)  # permuted: dim 0 is contiguous
        new_shape, new_strides = compute_reshape(shape, strides, (256, 256))
        # Base stride = strides[-1] = 512
        assert new_strides == (256 * 512, 512)


@pytest_marks(["neurotile"])
class TestComputePermute:
    @pytest.mark.fast
    def test_transpose_2d(self):
        shape = (128, 512)
        strides = (512, 1)
        new_shape, new_strides = compute_permute(shape, strides, (1, 0))
        assert new_shape == (512, 128)
        assert new_strides == (1, 512)

    def test_permute_3d(self):
        shape = (4, 128, 64)
        strides = (8192, 64, 1)
        new_shape, new_strides = compute_permute(shape, strides, (2, 0, 1))
        assert new_shape == (64, 4, 128)
        assert new_strides == (1, 8192, 64)


@pytest_marks(["neurotile"])
class TestComputeFlattenDims:
    @pytest.mark.fast
    def test_flatten_last_two(self):
        """Flatten (4, 128, 64) dims [1,2] -> (4, 8192)."""
        shape = (4, 128, 64)
        strides = (8192, 64, 1)
        new_shape, new_strides = compute_flatten_dims(shape, strides, 1, 2)
        assert new_shape == (4, 8192)
        assert new_strides == (8192, 1)  # innermost stride of merged range

    def test_flatten_first_two(self):
        """Flatten (4, 128, 64) dims [0,1] -> (512, 64)."""
        shape = (4, 128, 64)
        strides = (8192, 64, 1)
        new_shape, new_strides = compute_flatten_dims(shape, strides, 0, 1)
        assert new_shape == (512, 64)
        assert new_strides == (64, 1)


@pytest_marks(["neurotile"])
class TestComputeSqueezeDim:
    @pytest.mark.fast
    def test_basic(self):
        shape = (1, 128, 64)
        strides = (8192, 64, 1)
        new_shape, new_strides = compute_squeeze_dim(shape, strides, 0)
        assert new_shape == (128, 64)
        assert new_strides == (64, 1)


@pytest_marks(["neurotile"])
class TestComputeExpandDim:
    @pytest.mark.fast
    def test_basic(self):
        shape = (128, 64)
        strides = (64, 1)
        new_shape, new_strides = compute_expand_dim(shape, strides, 1)
        assert new_shape == (128, 1, 64)
        assert new_strides == (64, 0, 1)


@pytest_marks(["neurotile"])
class TestComputeBroadcast:
    @pytest.mark.fast
    def test_basic(self):
        shape = (128, 1, 64)
        strides = (64, 0, 1)
        new_shape, new_strides = compute_broadcast(shape, strides, 1, 16)
        assert new_shape == (128, 16, 64)
        assert new_strides == (64, 0, 1)


@pytest_marks(["neurotile"])
class TestComputeFold:
    @pytest.mark.fast
    def test_fold_outer(self):
        """Fold dim 2 into dim 0 (outer)."""
        shape = (4, 8, 16)
        strides = (128, 16, 1)
        new_shape, new_strides = compute_fold(shape, strides, src_dim=2, into_dim=0)
        assert new_shape == (64, 8)  # 4*16=64, dim 2 removed
        assert new_strides == (128, 16)  # into_dim gets its own stride

    def test_fold_inner(self):
        shape = (4, 8, 16)
        strides = (128, 16, 1)
        new_shape, new_strides = compute_fold(shape, strides, src_dim=2, into_dim=0, position="inner")
        assert new_shape == (64, 8)
        assert new_strides == (1, 16)  # src_dim's stride


# ============================================================================
# Chained transforms -- realistic pipeline
# ============================================================================


@pytest_marks(["neurotile"])
class TestTransformPipeline:
    @pytest.mark.fast
    def test_flatten_reshape_permute(self):
        """Realistic: (B, S, H) -> flatten(0,1) -> reshape_dim -> permute."""
        shape = (4, 128, 512)
        strides = _contiguous_strides(shape)
        assert strides == (65536, 512, 1)

        # Flatten B and S: (4, 128, 512) -> (512, 512)
        s1, st1 = compute_flatten_dims(shape, strides, 0, 1)
        assert s1 == (512, 512)
        assert st1 == (512, 1)

        # Reshape H into (H0, H1): (512, 512) -> (512, 8, 64)
        s2, st2 = compute_reshape_dim(s1, st1, dim=1, new_sub_shape=(8, 64))
        assert s2 == (512, 8, 64)
        assert st2 == (512, 64, 1)

        # Permute to (H0, BS, H1): (512, 8, 64) -> (8, 512, 64)
        s3, st3 = compute_permute(s2, st2, (1, 0, 2))
        assert s3 == (8, 512, 64)
        assert st3 == (64, 512, 1)

    def test_layout_apply_transform_pipeline(self):
        """Apply transform pipeline through HBMLayout.apply_transform."""
        layout = make_layout((4, 128, 512))

        # Pipeline: flatten -> reshape -> permute (from above)
        _, st1 = compute_flatten_dims((4, 128, 512), layout.strides, 0, 1)
        _, st2 = compute_reshape_dim((512, 512), st1, 1, (8, 64))
        _, st3 = compute_permute((512, 8, 64), st2, (1, 0, 2))

        result = layout.apply_transform(st3)
        assert result.strides == (64, 512, 1)
        assert result.offset == 0
        assert result.source is layout.source


# ============================================================================
# Repr
# ============================================================================


@pytest_marks(["neurotile"])
class TestRepr:
    @pytest.mark.fast
    def test_basic(self):
        layout = make_layout((128, 512))
        r = repr(layout)
        assert "HBMLayout(" in r
        assert "offset=0" in r

    def test_with_indirect(self):
        layout = HBMLayout(
            source=None,
            offset=100,
            strides=(64, 1),
            dtype="float32",
            indirect=_scalar_indirect("eid", 0),
        )
        r = repr(layout)
        assert "indirect=" in r


# ============================================================================
# compute_effective_tiles: clamping and tile count
# Regression: incorrect clamping caused wrong SBUF allocation shapes
# ============================================================================


@pytest_marks(["neurotile"])
class TestComputeEffectiveTiles:
    @pytest.mark.fast
    def test_exact_fit(self):
        eff, shape = HBMLayout.compute_effective_tiles((128, 512), (128, 512))
        assert eff == (128, 512)
        assert shape == (1, 1)

    def test_multi_tiles(self):
        eff, shape = HBMLayout.compute_effective_tiles((128, 512), (128, 128))
        assert eff == (128, 128)
        assert shape == (1, 4)

    def test_sub_tile_clamp(self):
        eff, shape = HBMLayout.compute_effective_tiles((64, 256), (128, 512))
        assert eff == (64, 256)
        assert shape == (1, 1)

    def test_remainder_ceiling(self):
        eff, shape = HBMLayout.compute_effective_tiles((128, 300), (128, 128))
        assert eff == (128, 128)
        assert shape == (1, 3)

    def test_3d(self):
        eff, shape = HBMLayout.compute_effective_tiles((8, 128, 256), (1, 128, 128))
        assert eff == (1, 128, 128)
        assert shape == (8, 1, 2)


# ============================================================================
# compute_sbuf_alloc: N-D vs flat 2D
# Regression: wrong allocation shape caused SBUF AP mismatch
# ============================================================================


@pytest_marks(["neurotile"])
class TestComputeSbufAlloc:
    @pytest.mark.fast
    def test_single_p_tile_keeps_nd(self):
        shape, p_tiles = HBMLayout.compute_sbuf_alloc((128, 512), (128, 512))
        assert p_tiles == 1
        assert shape == (128, 512)

    def test_multi_p_tile_flattens(self):
        shape, p_tiles = HBMLayout.compute_sbuf_alloc((256, 512), (128, 512))
        assert p_tiles == 2
        assert shape == (128, 1024)

    def test_3d_single_p_tile(self):
        shape, p_tiles = HBMLayout.compute_sbuf_alloc((128, 8, 64), (128, 8, 64))
        assert p_tiles == 1
        assert shape == (128, 8, 64)


# ============================================================================
# sbuf_f_extent and _contiguous_ap_pattern
# ============================================================================


@pytest_marks(["neurotile"])
class TestSbufFExtent:
    @pytest.mark.fast
    def test_single_p_tile(self):
        assert SBUFLayout.f_extent((128, 512), 128) == 512

    def test_multi_p_tile(self):
        assert SBUFLayout.f_extent((256, 512), 128) == 1024

    def test_3d(self):
        assert SBUFLayout.f_extent((128, 8, 64), 128) == 512


@pytest_marks(["neurotile"])
class TestContiguousApPattern:
    @pytest.mark.fast
    def test_2d(self):
        assert _contiguous_ap_pattern((128, 512)) == [[512, 128], [1, 512]]

    def test_3d(self):
        assert _contiguous_ap_pattern((128, 8, 64)) == [[512, 128], [64, 8], [1, 64]]


# ============================================================================
# indirect_dim preservation across multiple batch dim drops
# Regression: inline drop shifted dim indices but indirect_dim is source-relative
# ============================================================================


@pytest_marks(["neurotile"])
class TestIndirectDimBatchDrops:
    @pytest.mark.fast
    def test_drop_before_indirect(self):
        """Drop dim 0 with IndirectOffset.dim=1 keeps IndirectOffset.dim=1."""
        layout = HBMLayout(
            source=None,
            offset=0,
            strides=(512, 64, 1),
            dtype="float32",
            indirect=_scalar_indirect("pos", 1),
        )
        dropped = layout.drop_dim(0)
        assert dropped.indirect.dim == 1

    def test_drop_after_indirect(self):
        layout = HBMLayout(
            source=None,
            offset=0,
            strides=(512, 64, 1),
            dtype="float32",
            indirect=_scalar_indirect("pos", 1),
        )
        dropped = layout.drop_dim(2)
        assert dropped.indirect.dim == 1

    def test_two_drops_preserve_indirect(self):
        """IndirectOffset.dim is source-relative -- repeated drops don't shift it."""
        layout = HBMLayout(
            source=None,
            offset=0,
            strides=(4096, 512, 64, 1),
            dtype="float32",
            indirect=_scalar_indirect("pos", 2),
        )
        d1 = layout.drop_dim(0)
        assert d1.indirect.dim == 2
        d2 = d1.drop_dim(0)
        assert d2.indirect.dim == 2


# ============================================================================
# _apply_ap: pattern + offset, optional indirect dispatch
# ============================================================================


@pytest_marks(["neurotile"])
class TestBuildOverrideAp:
    @pytest.mark.fast
    def test_no_indirect(self):
        """No indirect: plain AP with pattern + offset."""

        class MockSource:
            def ap(self, **kwargs):
                return kwargs

        result = HBMLayout._apply_ap(
            MockSource(),
            offset=100,
            pattern=[[128, 4], [1, 64]],
            indirect=None,
        )
        assert result["pattern"] == [[128, 4], [1, 64]]
        assert result["offset"] == 100
        assert "scalar_offset" not in result
        assert "vector_offset" not in result

    def test_scalar_indirect(self):
        """Scalar IndirectOffset: passes scalar_offset + indirect_dim."""

        class MockSource:
            def ap(self, **kwargs):
                return kwargs

        result = HBMLayout._apply_ap(
            MockSource(),
            offset=0,
            pattern=[[128, 16], [2, 64]],
            indirect=_scalar_indirect("eid_scalar", 0),
        )
        assert result["scalar_offset"] == "eid_scalar"
        assert result["indirect_dim"] == 0
        assert "vector_offset" not in result

    def test_vector_indirect(self):
        """Vector IndirectOffset: passes vector_offset + indirect_dim."""

        class MockSource:
            def ap(self, **kwargs):
                return kwargs

        result = HBMLayout._apply_ap(
            MockSource(),
            offset=0,
            pattern=[[128, 16], [2, 64]],
            indirect=_vector_indirect("vec_tensor", 0),
        )
        assert result["vector_offset"] == "vec_tensor"
        assert result["indirect_dim"] == 0
        assert "scalar_offset" not in result
