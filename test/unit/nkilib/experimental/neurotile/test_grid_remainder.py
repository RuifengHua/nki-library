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

"""Coverage for ``Grid.remainder_dims`` + partial-trailing-tile semantics.

Pins three behaviors:

1. ``Grid.remainder_dims`` is per-dim -- a remainder on dim 1 must not
   trigger ``_compute_remaining``'s clamped-leaf path on other dims.
2. ``truncate_to_source`` finds the elem-leaf (innermost ``step==1``
   axis) on the dim it clamps, whether that is the outermost axis
   (post-``consume``) or sits inside a TILE / BLOCK wrapper
   (post-``narrow``).
3. ``view[:, last_tile:last_tile+1]`` over a partial trailing tile
   reports the partial-aware ``element_shape`` -- this is what unblocks
   the FGCC MLP CTE remainder kernels.

The PSUM new-API tests live in ``test_psum_pool_api.py``.
"""

import pytest

from nkilib_src.nkilib.experimental.neurotile.core.axis import Axis, AxisLabel
from nkilib_src.nkilib.experimental.neurotile.core.grid import Grid
from test.utils.pytest_test_metadata import pytest_marks

# ============================================================================
# remainder_dims: per-dim flag tracking
# ============================================================================


@pytest_marks(["neurotile"])
class TestRemainderDimsPerDim:
    """``remainder_dims`` is a tuple of dim indices; ``is_remainder`` is a
    convenience bool (``len(remainder_dims) > 0``)."""

    @pytest.mark.fast
    def test_constructor_default_empty(self):
        # No clamping: an exactly-tile-aligned grid surfaces no remainder.
        g = Grid.from_shape(element_shape=(128, 512), tile_size=(128, 128))
        assert g.remainder_dims == ()
        assert g.is_remainder is False

    def test_construction_derives_remainder_from_overshoot(self):
        # Source 1792 with tile_size 512 -> ceil 4 tiles, last partial.
        # Outer axis count*step (= 4*512=2048) overshoots element_shape (1792).
        g = Grid.from_shape(element_shape=(128, 1792), tile_size=(128, 512))
        assert 1 in g.remainder_dims
        assert g.is_remainder is True

    def test_explicit_remainder_dims_carries_through(self):
        # Direct constructor: ``remainder_dims=`` records dims clamped by
        # truncate_to_source even when overshoot detection would not fire.
        es = (128, 256)
        ax_p = Axis(count=128, step=1, dim=0, label=AxisLabel.PARTITION)
        ax_e = Axis(count=256, step=1, dim=1, label=AxisLabel.ELEM)
        g = Grid(
            element_shape=es,
            axes=(ax_p, ax_e),
            cursor=0,
            n_batch_dims=0,
            remainder_dims=(1,),
        )
        assert 1 in g.remainder_dims

    def test_remainder_on_one_dim_does_not_pollute_other_dims(self):
        """Regression: a single-bool ``is_remainder`` made
        ``_compute_remaining`` mis-clamp every dim once any dim was a
        remainder."""
        # Dim 0 fully tiles (4096 / 128 = 32); dim 1 has remainder
        # (1792 / 512 = 3 + partial).
        g = Grid.from_shape(
            element_shape=(4096, 1792),
            tile_size=(128, 512),
            block_size=(32, 1),
        )
        # Dim 0 should report the full source extent.
        assert g.remaining[0] == 4096
        # Dim 1 reports the source extent (overshoot clamped to source).
        assert g.remaining[1] == 1792

    def test_dims_walking_past_source_detects_per_dim(self):
        g = Grid.from_shape(
            element_shape=(4096, 1792),
            tile_size=(128, 512),
            block_size=(32, 1),
        )
        derived = g._dims_walking_past_source()
        # Dim 0 fits exactly; only dim 1 walks past.
        assert derived == (1,)


# ============================================================================
# truncate_to_source: finds the innermost step==1 axis and clamps
# ============================================================================


@pytest_marks(["neurotile"])
class TestTruncateToSource:
    @pytest.mark.fast
    def test_clamps_outer_when_outer_is_elem_leaf(self):
        """Single-axis dim: outer IS the elem-leaf -- clamp via narrow."""
        # Single elem axis count=512, source extent=1024. 100 elements
        # consumed -> addressable=924; leaf 512 fits, no clamp. Bump
        # consumed to 700: addressable=324 < 512 -> clamps to 324.
        es = (128, 1024)
        ax_p = Axis(count=128, step=1, dim=0, label=AxisLabel.PARTITION)
        ax_e = Axis(count=512, step=1, dim=1, label=AxisLabel.ELEM)
        g = Grid(
            element_shape=es,
            axes=(ax_p, ax_e),
            cursor=0,
            n_batch_dims=0,
        )
        clamped = g.truncate_to_source(dim=1, elements_consumed=700)
        elem = [ax for ax in clamped.axes if ax.dim == 1 and ax.step == 1][0]
        assert elem.count == 324
        assert 1 in clamped.remainder_dims

    def test_clamps_elem_leaf_inside_tile_block_wrapper(self):
        """Multi-axis dim: BLOCK -> TILE -> ELEM. Clamp the inner ELEM."""
        es = (128, 1792)
        ax_p = Axis(count=128, step=1, dim=0, label=AxisLabel.PARTITION)
        # Slice on dim 1 narrowed BLOCK to count=1; ELEM still 512.
        ax_b = Axis(count=1, step=512, dim=1, label=AxisLabel.BLOCK)
        ax_t = Axis(count=1, step=512, dim=1, label=AxisLabel.TILE)
        ax_e = Axis(count=512, step=1, dim=1, label=AxisLabel.ELEM)
        g = Grid(
            element_shape=es,
            axes=(ax_p, ax_b, ax_t, ax_e),
            cursor=0,
            n_batch_dims=0,
        )
        # 1536 elements consumed (slice start 3 * 512). Addressable = 256.
        clamped = g.truncate_to_source(dim=1, elements_consumed=1536)
        elem = [ax for ax in clamped.axes if ax.dim == 1 and ax.step == 1][0]
        assert elem.count == 256
        # BLOCK / TILE wrappers untouched.
        block = [ax for ax in clamped.axes if ax.label == AxisLabel.BLOCK][0]
        assert block.count == 1 and block.step == 512

    def test_no_op_when_addressable_fits(self):
        """When the leaf already fits the addressable remainder, no change."""
        es = (128, 1024)
        ax_p = Axis(count=128, step=1, dim=0, label=AxisLabel.PARTITION)
        ax_e = Axis(count=128, step=1, dim=1, label=AxisLabel.ELEM)
        g = Grid(
            element_shape=es,
            axes=(ax_p, ax_e),
            cursor=0,
            n_batch_dims=0,
        )
        # Addressable=1024-100=924 >= leaf.count=128 -> no-op.
        clamped = g.truncate_to_source(dim=1, elements_consumed=100)
        assert clamped is g or clamped.axes == g.axes
        assert 1 not in clamped.remainder_dims

    def test_no_op_for_non_int_elements_consumed(self):
        """Runtime ``elements_consumed`` (CExpr / LoopVar) -> no-op."""
        es = (128, 256)
        ax_p = Axis(count=128, step=1, dim=0, label=AxisLabel.PARTITION)
        ax_e = Axis(count=512, step=1, dim=1, label=AxisLabel.ELEM)
        g = Grid(
            element_shape=es,
            axes=(ax_p, ax_e),
            cursor=0,
            n_batch_dims=0,
        )

        class FakeCExpr:
            pass

        clamped = g.truncate_to_source(dim=1, elements_consumed=FakeCExpr())
        assert clamped is g

    def test_no_op_when_no_elem_leaf(self):
        """Dim has only TILE / BLOCK axes, no step==1 axis."""
        es = (128, 256)
        ax_p = Axis(count=128, step=1, dim=0, label=AxisLabel.PARTITION)
        ax_b = Axis(count=4, step=64, dim=1, label=AxisLabel.BLOCK)
        g = Grid(
            element_shape=es,
            axes=(ax_p, ax_b),
            cursor=0,
            n_batch_dims=0,
        )
        clamped = g.truncate_to_source(dim=1, elements_consumed=100)
        assert clamped is g


# ============================================================================
# _compute_remaining: partial-trailing-tile formula on flagged dims only
# ============================================================================


@pytest_marks(["neurotile"])
class TestComputeRemainingPartialTile:
    @pytest.mark.fast
    def test_normal_walk_uses_count_times_step(self):
        # 4 tiles of 128 each -> walked 512.
        g = Grid.from_shape(element_shape=(128, 512), tile_size=(128, 128))
        assert g.remaining == (128, 512)

    def test_partial_trailing_tile_uses_clamped_leaf(self):
        # 1792 / 512 -> 3 full + 1 partial of 256.
        # _compute_remaining for dim 1 takes the partial-aware path:
        #   walked = (4-1)*512 + 512 = 2048; min(2048, 1792) = 1792.
        # Dim 0: 32 * 128 = 4096 == element_shape -> normal path.
        g = Grid.from_shape(
            element_shape=(4096, 1792),
            tile_size=(128, 512),
            block_size=(32, 1),
        )
        assert g.remaining[0] == 4096
        assert g.remaining[1] == 1792

    def test_post_truncate_partial_clamps_dim_remaining(self):
        """After ``truncate_to_source`` clamps the elem-leaf, dim's
        ``remaining`` reflects the clamped count."""
        # gate[:, 3:4] over (128, 1792) with tile_size=(128, 512):
        #   BLOCK count=1 step=512, TILE count=1 step=512, ELEM count=256
        #   (clamped from 512). remainder_dims=(1,) -> partial-aware path:
        #   walked = (1-1)*512 + 256 = 256.
        es = (128, 1792)
        ax_p = Axis(count=128, step=1, dim=0, label=AxisLabel.PARTITION)
        ax_b = Axis(count=1, step=512, dim=1, label=AxisLabel.BLOCK)
        ax_t = Axis(count=1, step=512, dim=1, label=AxisLabel.TILE)
        ax_e = Axis(count=256, step=1, dim=1, label=AxisLabel.ELEM)
        g = Grid(
            element_shape=es,
            axes=(ax_p, ax_b, ax_t, ax_e),
            cursor=0,
            n_batch_dims=0,
            remainder_dims=(1,),
        )
        assert g.remaining == (128, 256)

    def test_sharded_view_takes_normal_path(self):
        """Sharded / interleaved views never enter ``remainder_dims``;
        their ``count * step`` walks past sharded peers but the source
        extent itself is uniform -- the partial-aware path must not
        engage."""
        # 4 owned interleaved tiles of 128 each, peer step=256:
        # outer count=4 step=256, leaf count=128.
        # Without remainder_dims, walked = 4*256 = 1024 (full strided walk).
        es = (128, 1024)
        ax_p = Axis(count=128, step=1, dim=0, label=AxisLabel.PARTITION)
        ax_owned = Axis(count=4, step=256, dim=1, label=AxisLabel.TILE)
        ax_e = Axis(count=128, step=1, dim=1, label=AxisLabel.ELEM)
        g = Grid(
            element_shape=es,
            axes=(ax_p, ax_owned, ax_e),
            cursor=0,
            n_batch_dims=0,
        )
        # remainder_dims must be empty for sharded views.
        assert g.remainder_dims == ()
        assert g.remaining[1] == 1024
