# Neurotile API Reference

**Status:** Draft | **Authors:** Pranav Ladkat (@ladkat) | **Last updated:** April 2026

---

## Overview

Neurotile is a tile iterator library for NKI that replaces manual tiling loops, DMA orchestration, and SBUF tracking with composable, high-level primitives. Production NKI kernels are dominated by boilerplate (~70% of code): manual index arithmetic, SBUF management, DMA patterns, and sharding logic. Neurotile eliminates this boilerplate while preserving explicit control over NKI ISA instructions for the algorithmic core (~30%).

### Motivation

- **Zero-cost abstractions for NKI kernel development** -- neurotile lets kernel authors express algorithms cleanly without hand-managing tiling, indexing, HBM-to-SBUF data movement, or kernel chaining. It achieves performance parity with hand-written NKI kernels and, in practice, exceeds them due to faster iteration cycles.

- **Reducing kernel complexity and development time** -- NKI provides unfiltered access to NeuronCore for maximum performance, but the resulting code is very low-level and verbose: thousands of lines of index math, access pattern computation, and DMA orchestration. This leads to month-long development cycles even for experienced Neuron engineers, limits the talent pool that can contribute, and makes it impractical for researchers or the open-source community to go from algorithm idea to working kernel.

- **Composable streaming kernels for combinatorial model variants** -- LLMs share a Transformer backbone but differ in precision, attention/MoE/MLP variants, normalization, bias, sharding strategy, and fusion boundaries. Today these combinatorial features are handled with branching inside monolithic "mega" kernels, making the codebase hard to navigate, debug, and maintain. Neurotile introduces streaming producers -- each stage takes an input stream and produces an output stream -- that can be composed, swapped, or fused at the call site for each model variant, with zero overhead.

- **Centralizing high-impact optimizations** -- since NKI exposes low-level ISA instructions, every kernel re-implements boilerplate for coalesced DMAs, software pipelining (double/multi-stage buffering), LNC sharding, and memory allocation. Not all kernel authors are aware of every applicable optimization, and retrofitting them requires structural rewrites. Neurotile's APIs provide these optimizations by default -- coalesced multi-tile DMAs, streaming-based pipelining, sharding patterns -- while remaining transparent and overridable.

## Architecture

Neurotile is built on a single composite type:

- **`NDSlice` = `Grid` + `Layout`**
  - `Grid` (`core/grid.py`) -- per-dim level stacks (block step / tile step / element step), remaining region, cursor.
  - `Layout` -- `HBMLayout` or `SBUFLayout` -- physical memory descriptor (source tensor, offset, strides).

Every factory (`nt.tiles`, `nt.blocks`, `nt.tensor_view`, `nt.alloc_tiles`, `nt.alloc_blocks`) returns an `NDSlice`. Every slice (`x[...]`), iteration step, load result, and stream child is also an `NDSlice`. `BlockStream` is the one composite type -- it owns the rotating SBUF buffer pool and hands out `NDSlice` children whose `.load()` routes into the rotating slot.

## Design Philosophy

- **Maximize SBUF data reuse** -- tiles are sized to fit on-chip; hoisting and fusion minimize HBM transfers
- **Composable view primitives** -- `nt.tiles`, `nt.blocks`, `nt.tensor_view`
- **Explicit allocation and pipelining** -- `nt.alloc_tiles`, `nt.alloc_blocks`, `nt.psum_pool`, `.stream(dim=d, buffer_count=N)`
- **Expose NKI loop semantics** -- iteration uses `nl.affine_range` / `nl.sequential_range` directly

---

## Usage

```python
import nki
import nki.language as nl
import nki.isa as nisa
import neurotile as nt
```

---

## 1. Tiles -- `nt.tiles()`

Decomposes a tensor into a logical grid of fixed-size tiles. No data is moved.

```python
src_tiles = nt.tiles(src, tile_size=(128, 256))   # (512, 1024) -> 4x4 tile grid
src_tiles.shape                               # (4, 4) -- tile grid dimensions
```

```
          256       256       256       256
       +---------+---------+---------+---------+
  128  | (0, 0)  | (0, 1)  | (0, 2)  | (0, 3)  |
       +---------+---------+---------+---------+
  128  | (1, 0)  | (1, 1)  | (1, 2)  | (1, 3)  |
       +---------+---------+---------+---------+
  128  | (2, 0)  | (2, 1)  | (2, 2)  | (2, 3)  |
       +---------+---------+---------+---------+
  128  | (3, 0)  | (3, 1)  | (3, 2)  | (3, 3)  |
       +---------+---------+---------+---------+

       src.shape = (512, 1024)          -- element dimensions
       src_tiles.shape = (4, 4)         -- tile grid dimensions
```

**Full signature:**

```python
nt.tiles(source, tile_size=None, access_pattern=None,
         root=None, buffer_type=None, remainder=None)
```

#### Parameters

| Param | Accepts | Default | Role |
|---|---|---|---|
| `source` | HBM tensor / SBUF ndarray / `NDSlice` | required | What to tile |
| `tile_size` | `tuple[int, ...]`, rank >= 2 | None (required for raw sources) | Tile shape; lower-rank than source -> leading dims auto-pad with `1` (batch dims) |
| `access_pattern` | `[[stride, count], ...]` | None (contiguous) | Source layout descriptor; raw sources only |
| `root` | top-level `nl.ndarray` | None | Parent tensor when `source` is a sliced HBM view |
| `buffer_type` | `nl.sbuf` / `nl.shared_hbm` / `nl.private_hbm` / None | None | Source memory region; `nl.psum` rejected |
| `remainder` | `"skip"` / None | None | `"skip"` floor-divides the grid; `None` keeps boundary tiles |

To shard the tile view across cores, slice it: `nt.tiles(...)[block_range(...), :]`.
See Sec. 10 Multi-Core (LNC) Sharding.

Tensors of any dimensionality are supported. Higher-dimensional tensors produce higher-dimensional tile grids.

#### Source modes

The operation performed depends on what `source` is and which other args are set.

| Source | Other args | What tiles() does |
|---|---|---|
| Raw HBM tensor | `tile_size=` required | Construct `NDSlice(Grid, HBMLayout)` |
| Raw HBM tensor | `tile_size=` + `access_pattern=` | Construct with AP-derived strides |
| Raw HBM tensor | `tile_size=` + `remainder="skip"` | Construct with floor-divided grid |
| Raw SBUF ndarray | `tile_size=` + `buffer_type=nl.sbuf` | Construct `NDSlice(Grid, SBUFLayout)` |
| Sliced HBM ndarray | `root=parent` required | Construct; strides from parent, offset from slice |
| NDSlice (tile or block) | no `tile_size=` | **Descend**: strip block axis, reset cursor, return tile-level view |
| NDSlice (tile or block) | `tile_size=` | **Re-tile**: rebuild Grid with new tile size, preserve outer shard axes |

#### Composition matrix

Which args are accepted on which source kind:

| Source kind \ arg | `tile_size=` | `access_pattern=` | `root=` | `buffer_type=` | `remainder=` |
|---|---|---|---|---|---|
| Raw HBM tensor | required | optional | sliced view: required | optional (None / `nl.shared_hbm` / `nl.private_hbm`) | optional |
| Raw SBUF ndarray | required | optional | rejected | required (`nl.sbuf`) | optional |
| NDSlice (re-tile / descend) | optional | rejected | rejected | rejected | rejected |

`nl.psum` is rejected for all sources -- PSUM buffers are produced via `nt.psum_pool()`.

#### Validation rules

`tiles()` rejects misuse with named asserts. Each row maps a rule to the assert text it raises.

| # | Rule | Assert excerpt |
|---|---|---|
| 1 | `tile_size` must be tuple/list of positive ints, rank >= 2 | `tile_size= must have at least 2 dims (P, F)` |
| 2 | `tile_size` rank <= source rank | `tile_size must not exceed source rank` |
| 3 | `tile_size[d]` <= `source.shape[n_batch + d]` | `tile_size[d]=... exceeds source extent ... on dim ...` |
| 4 | `access_pattern` rank == source rank, with `stride > 0` and `count > 0` per dim | `access_pattern has ... levels but source has ... dims` / `(stride/count) must be > 0` |
| 5 | AP `count[d]` <= `source.shape[d]` | `access_pattern[d][1] (count=...) exceeds source.shape[d]=...` |
| 6 | `access_pattern` rejected on NDSlice source | `nt.tiles(source=NDSlice): access_pattern= is not supported` |
| 7 | `buffer_type` is None / `nl.sbuf` / `nl.shared_hbm` / `nl.private_hbm` | `buffer_type=nl.psum is not supported` (or unknown enum value) |
| 8 | NDSlice source rejects `buffer_type=` / `root=` / `remainder=` | `... is only for raw sources` / `remainder= applies at construct time only` |
| 9 | Sliced HBM source needs `root=` (top-level parent) | `pass root=<parent tensor>` |
| 10 | `remainder` is `None` or `"skip"` | `remainder=... is not valid. Must be one of (None, 'skip')` |

##### Construct (raw tensor)

```python
# HBM tensor, default contiguous layout
src_tiles = nt.tiles(src, tile_size=(128, 256))

# SBUF tensor (raw nl.ndarray in SBUF)
sb = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
sb_tiles = nt.tiles(sb, tile_size=(128, 128), buffer_type=nl.sbuf)

# Sliced source: strides from parent, not from the slice's logical shape
src_tiles = nt.tiles(src[:, a:b], tile_size=(128, 128), root=src)

# Custom source layout: strided / transposed / etc
src_tiles = nt.tiles(src, access_pattern=[[2 * N, M // 2], [1, N]], tile_size=(64, N))
```

##### Re-tile an existing NDSlice

```python
# Block-level view -> tile-level view with smaller tiles
block_view = nt.blocks(src, tile_size=(128, 512), block_size=(2, 2))
tile_view = nt.tiles(block_view, tile_size=(128, 128))   # rebuilds Grid for 128x128
```

##### Descend through block level

```python
# No tile_size= given: strip block level, iterate tiles at the view's existing tile_size
block_view = nt.blocks(src, tile_size=(128, 512), block_size=(2, 2))
tile_view = nt.tiles(block_view)           # shape = (M//128, N//512); block gone
```

Both descend (no `tile_size=`) and re-tile (with `tile_size=`) produce a view with `block_size is None` -- the block grouping is dropped. Outer shard / broadcast axes are preserved across both. HBM and SBUF behave symmetrically.

##### Apply sharding to an un-sharded view

```python
# Slice to narrow the view to the rank's owned tiles.
base = nt.tiles(src, tile_size=(128, 256))
sharded = base[block_range(rank, num_shards, total), :]
```

#### Lower-rank `tile_size=` on higher-rank tensors (batch dims)

When `tile_size=` has fewer dims than `source`, the leading source dims are **batch dimensions**: neurotile auto-pads `tile_size` with `1`s on the left. Each batch dim contributes one iteration level and is consumed by a single index. For a 3D tensor with a 2D `tile_size=`, `src_tiles.shape` is `(B, M // tile_size[0], N // tile_size[1])` -- the first index selects a batch slab, subsequent indices address the tile grid.

Batch dims behave like slab selectors -- index them down (via iteration or integer indexing) before calling `.load()` / `.store()` / `.stream()`. Neurotile asserts this explicitly if batch dims are unconsumed.

`tile_size=` **must be at least 2-D** (P and F axes explicit). 1-D `tile_size=(N,)` is rejected -- users who want a P=N partition tile write `tile_size=(N, 1)`, and users who want an F=N column tile write `tile_size=(1, N)`. No shorthand, no auto-inference of P/F orientation.

#### Tile size <-> source shape <-> SBUF shape mapping

After validation and auto-padding, `tile_size` maps position-by-position to source: `tile_size[d]` operates on `source.shape[n_batch + d]`. The padded tile shape's **dim 0** becomes SBUF's **P axis** (hardware partition); the remaining dims are **F axis subdivisions**, preserved as logical structure in the SBUF buffer (not flattened).

| `tile_size=` | Source shape | Padded tile shape | `n_batch` | Tile after full indexing | SBUF shape after `.load()` |
|---|---|---|---:|---|---|
| `(P, F)` | `(M, N)` | `(P, F)` | 0 | `view[i, j]` -> `(P, F)` | `(P, F)` |
| `(P, F)` | `(B, M, N)` | `(1, P, F)` | 1 | `view[b, i, j]` -> `(P, F)` | `(P, F)` |
| `(P, F)` | `(B1, B2, M, N)` | `(1, 1, P, F)` | 2 | `view[b1, b2, i, j]` -> `(P, F)` | `(P, F)` |
| `(P, F1, F2)` | `(M, N1, N2)` | `(P, F1, F2)` | 0 | `view[i, j, k]` -> `(P, F1, F2)` | `(P, F1, F2)` |
| `(P, 1)` | `(M, N)`, P<=M | `(P, 1)` | 0 | `view[i, j]` -> `(P, 1)` | `(P, 1)` |
| `(P, 1)` | `(B, D, S)`, P<=D | `(1, P, 1)` | 1 | `view[b, i, j]` -> `(P, 1)` | `(P, 1)` |
| `(1, F)` | `(M, N)`, F<=N | `(1, F)` | 0 | `view[i, j]` -> `(1, F)` | `(1, F)` |

Key invariant: the **SBUF shape equals `element_shape`** at the time of `.load()`. There is no dimension flattening -- higher-rank tiles stay higher-rank on SBUF, iterating F subdivisions with their logical shape intact.

#### Sub-tile indexing narrows element_shape, not tile_size

After reaching a tile via full indexing, further slicing narrows the tile's `element_shape` but preserves `tile_size` (the declared grain). `.load()` allocates SBUF sized to `element_shape`, not `tile_size`.

| Starting point | Sub-tile operation | New `element_shape` | SBUF shape after `.load()` |
|---|---|---|---|
| `view[i, j]` with tile `(P, F)` | `[p, :]` (scalar on P) | `(1, F)` | `(1, F)` |
| `view[i, j]` with tile `(P, F)` | `[:, k]` (scalar on F) | `(P, 1)` | `(P, 1)` |
| `view[i, j]` with tile `(P, F)` | `[p0:p1, :]` (range on P) | `(p1-p0, F)` | `(p1-p0, F)` |
| `view[i, j]` with tile `(P, F)` | `[:, k0:k1]` (range on F) | `(P, k1-k0)` | `(P, k1-k0)` |
| `view[i, j]` with tile `(P, F)` | `[p0:p1, k0:k1]` | `(p1-p0, k1-k0)` | `(p1-p0, k1-k0)` |
| `view[i, j]` with tile `(P, F1, F2)` | `[:, f1, :]` (scalar on middle F) | `(P, 1, F2)` | `(P, 1, F2)` |
| `view[i, j]` with tile `(P, F1, F2)` | `[:, :, f2]` (scalar on inner F) | `(P, F1, 1)` | `(P, F1, 1)` |

Key invariants:
- The tile's **P axis always maps to dim 0** of the tile (post-batch-drop). Sub-tile slicing can narrow it to a smaller P, but cannot rotate P onto another dim.
- F subdivisions are preserved as logical structure in the SBUF buffer; `.load()` does not flatten them. A tile with `tile_size=(P, F1, F2)` produces an SBUF tile of shape `(P, F1, F2)`.
- The **physical SBUF backing is always 2-D** `(P, F_flat)` regardless of logical rank: a 3-D tile `(P, F1, F2)` is stored as `(P, F1*F2)` and `tile.data.shape` reports the 2-D shape. Sub-tile slices on the logical tile (`tile[:, f1, :]`) flatten through the 2-D physical view internally; the user sees the logical narrowing.
- To map a different source dim onto P, reshape or permute the source first (chain `nt.tensor_view(src).reshape_dim(...)` / `.permute(...)` into `nt.tiles(...)`), or use `.load(transpose=True)` for in-DMA transposition.

---

## 2. Slicing and Indexing

NumPy-style indexing at tile granularity. All results are `NDSlice` view objects (no data movement).

| Pattern | Meaning |
|---|---|
| `tiles[i, j]` | Single tile (element-level after cleanup) |
| `tiles[i]` / `tiles[i, :]` | Row of tiles |
| `tiles[:, j]` | Column of tiles |
| `tiles[a:b]` | Range of tile rows |
| `tiles[a:b, c:d]` | Rectangular sub-region |
| `tiles[a:b:s]` | Stepped slice -- routes through interleaved-shard handling. `s` must be `>= 1`. See Sec. 10 for the round-robin sharding use case. |

Chained indexing supported: `tiles[0][2]` selects the tile at `(0, 2)`.

**Accepted key types:** `int` (compile-time), `slice` (with `step >= 1`), and NKI runtime expressions (`nl.program_id(0)`, `LoopVar`, 1-D SBUF `NDSlice` for vector gather).

**Rejected key types** (named asserts): `bool`, `list`, `dict`, `str`, `float`. Fancy indexing, boolean masking, and ellipsis (`...`) are not supported.

#### Single-int subscript on a child view targets the cursor's dim

A parent index that consumes a dim (e.g. `parent[i, :]`, `parent[bi, :]`, `tiles[:, j]`) advances the view's cursor past the consumed dim. A subsequent **single-int** subscript on the child targets the cursor's dim, not dim 0:

| Parent | Child cursor | Child key | Targets dim |
|---|---|---|---|
| `tiles[i, :]` | 1 | `row[j]` | 1 (TILE on dim 1) |
| `blocks[bi, :]` | 1 | `block_row[bj]` | 1 (BLOCK on dim 1) |
| `tiles[:, j]` | 0 | `col[i]` | 0 (TILE on dim 0) |
| Fresh batched `v` (`n_batch_dims=1`) | 1 | `v[batch_idx]` | 0 (batch slab) |

The deflect rule applies only to single-int keys on a non-batch dim that the cursor has already advanced past. Multi-key subscripts (`view[i, j]`) and batch-dim indexing on a fresh view always target dims 0..len(keys)-1. Don't write `row[:, j]` to "skip" a consumed dim -- the consumed dim is library state, not user-addressable; write `row[j]` directly.

`view[i]` and `view[i, :]` produce identical views; the trailing default `:` is a no-op.

**Batch consume + single trailing tile.** When a tile view's only iteration level is the batch dim and the trailing tile grid is `(1, ..., 1)` (e.g. `tiles=nt.tiles(src, tile_size=(P, F))` on `src.shape=(B, P, F)`), `tiles[b]` consumes the batch dim **and** auto-pops the trivial trailing TILE axes -- the resulting view is already at single-tile element-level (`shape == ()`, ready for `.load()` directly). Do not subscript further (`tiles[b][0, 0]`); call `.load()` on the slab.

#### NDSlice attributes

Every NDSlice (factory output, slice result, transform output, loaded SBUF view) exposes the same public attributes:

| Attribute | Type | Meaning |
|---|---|---|
| `.shape` | `tuple[int, ...]` | Iteration count per dim from the current cursor onward (e.g. tile-grid dims for a tile-level view, block-grid for a block-level view). Decreases as outer dims are indexed away. |
| `.element_shape` | `tuple[int, ...]` | Element-level extent per dim. Narrows on element-level slicing (`.slice(...)` or sub-tile index); unchanged by tile-level indexing. |
| `.tile_size` | `tuple[int, ...]` or None | Per-tile shape (the granularity argument to `nt.tiles`/`nt.blocks`). None on untiled views (`nt.tensor_view`). |
| `.tile_shape` | `tuple[int, ...]` or None | Per-dim tile-grid count (`ceil(element_shape[d] / tile_size[d])`) -- how many tiles span this view's elements. Survives consume (unlike `.shape` which tracks the iteration cursor). None on untiled views. |
| `.block_size` | `tuple[int, ...]` or None | Tiles per block. None when no block level (tile-only views, untiled views). |
| `.block_shape` | `tuple[int, ...]` or None | Per-dim block-grid count (`ceil(element_shape[d] / (block_size[d] * tile_size[d]))`). None when no block level. |
| `.is_tiled` | bool | True when the view was constructed with a `tile_size` (any tiled view; survives consume). |
| `.is_blocked` | bool | True when the view has a block level (any block view; non-block views and untiled views are False). |
| `.ndim` | int | Number of dims = `len(shape)` = `len(element_shape)`. |
| `.is_remainder` | bool | True when the outermost iteration extent is not divisible by tile_size (the trailing tile is a partial). |
| `.source` | `nl.ndarray` | The underlying tensor (HBM root or SBUF buffer). |
| `.offset` | int | Byte/element offset from `.source` to the start of this view. |
| `.strides` | `tuple[int, ...]` | Element strides per dim (HBM physical layout). |
| `.dtype` | NKI dtype | Element dtype. |
| `.buffer_type` | `nl.MemoryRegion` | `nl.shared_hbm`, `nl.private_hbm`, `nl.sbuf`, or `nl.psum`. |
| `.data` | `nl.ndarray` or root | For SBUF-backed views: the underlying `nl.ndarray` to pass to `nisa.*` compute ops. For HBM views: the root source tensor (use `.ap()` for the offset+strides+region; `.data` alone is rarely what you want for HBM). |

Use `.ap()` to build the access-pattern view that DMA / compute ops consume; `.data` only when an op accepts the raw `nl.ndarray` directly.

#### Splitting iteration by remainder

When the iteration extent isn't a multiple of `tile_size`, the trailing tile is a partial (`.is_remainder == True`). Two helpers split the iteration into clean and boundary halves so each can use the right DMA policy:

| Method | Returns |
|---|---|
| `.whole_tiles()` | list of sub-views with `.is_remainder == False` |
| `.remainder_tiles()` | list of sub-views with `.is_remainder == True` |

```python
for tile in tiles.whole_tiles():
    tile.load()                   # full-extent DMA
for rem in tiles.remainder_tiles():
    rem.load(oob_mode=nisa.oob_mode.skip, oob_value=0.0)
```

The alternative -- iterating `.tolist()` and branching on `.is_remainder` per-iteration -- works too; pick the form that reads better at the call site. See Sec.8 for the full remainder model.

---

## 3. Data Movement

#### Manual DMA escape hatches

For cases the high-level methods don't express -- custom DMA scheduling, combining an HBM access pattern with a hand-built SBUF destination -- `NDSlice` exposes two escape hatches you can pair with raw `nisa.dma_copy`:

- **`view.ap()`** -- build the NKI access-pattern view for the region described by this `NDSlice`. For HBM views it reflects tensor + offset + strides + any indirect parameters; for SBUF views it reflects the flat F-offset layout. The returned value is the same object kind that the high-level `.load()` / `.store()` constructs internally. This is what you hand to `nisa.dma_copy` for manual DMA.
- **`view.data`** -- for SBUF-backed views, the underlying `nl.ndarray` that you pass to `nisa.*` compute ops. For HBM views, `view.data` returns the **root source tensor** -- it does not carry the `NDSlice`'s offset, strides, or tile region, so use `view.ap()` (not `view.data`) as the `src=` / `dst=` of an HBM `nisa.dma_copy`.

```python
# Manual HBM -> SBUF copy
src_tiles = nt.tiles(src_hbm, tile_size=(128, 512))
dst = nl.ndarray((128, 512), dtype=src_hbm.dtype, buffer=nl.sbuf)
nisa.dma_copy(dst=dst, src=src_tiles[i, j].ap())

# Manual SBUF -> HBM copy using an allocated SBUF buffer
buf = nt.alloc_tiles(tile_size=(128, 512), buffer_type=nl.sbuf, dtype=src_hbm.dtype)
# ... fill buf.data ...
nisa.dma_copy(dst=dst_tiles[i, j].ap(), src=buf.ap())
```

`.ap()` accepts no arguments -- it captures the region already described by the `NDSlice`. Use `pattern_override=` on `.load()` / `.store()` (section 6) when you need a different AP pattern for the same region without dropping down to raw `nisa.dma_copy`.

#### Single-DMA contract

Every call to `.load()`, `.store()`, or `.stream().load(k)` / `.stream().store(k)` emits **exactly one `nisa.dma_copy` instruction**. If the region cannot be expressed as a single coalesced DMA -- for example, a `tiles[a:b, :]` slice whose HBM layout cannot be captured by NKI's 4-level access pattern -- the call raises an error at trace time. It does not silently fall back to a multi-DMA path.

The single exception is `.fold()` along the partition dimension (`src_dim == 0` or `into_dim == 0`): partition folds emit K separate `nisa.dma_copy` calls because a single AP pattern cannot address disjoint partition ranges (see Sec.11 / Sec.6).

When a multi-tile `.load()` fails the single-DMA check, the workaround is to iterate and load each tile (or block) individually:

```python
# If this raises (e.g. layout does not fit a single coalesced DMA):
#   hoisted = tiles[:, m].load()

# Loop and load each tile with its own single-DMA .load():
col = tiles[:, m]
for k in range(col.shape[0]):
    tile = col[k].load()
    ...
```

### `.ap()` -- access-pattern view

```python
view.ap()
```

No arguments. Returns the NKI access-pattern view describing the NDSlice's region (source + offset + strides + indirect parameters for HBM views; flat F-offset layout for SBUF views).

| Use case | Example |
|---|---|
| Manual DMA between HBM and SBUF | `nisa.dma_copy(dst=sbuf, src=hbm_tiles[i, j].ap())` |
| Feed an SBUF view to an `nisa.*` compute op | `nisa.tensor_tensor(out, x, gamma_bc.ap(), nl.multiply)` |
| Pass a loaded SBUF NDSlice to `.store()` | `dst_tiles[i, j].store(tile.ap())` |

For a custom AP pattern over the same region, use `pattern_override=` on `.load()` / `.store()` instead.

### `.load()` -- HBM to SBUF

```python
view.load(dtype=None, transpose=False, dge_mode=None, oob_mode=None,
          oob_value=None, dst=None, pattern_override=None, out_shape=None)
```

Single coalesced DMA. Returns an `NDSlice(SBUFLayout)`.

#### Parameters

| Param | Accepts | Default | Role |
|---|---|---|---|
| `dtype` | NKI dtype | source dtype | Allocate the SBUF destination at this dtype. The DMA hardware auto-casts on a dtype mismatch with the HBM source -- there is no separate cast op. (`nisa.dma_copy` itself takes no `dtype=`; the cast is purely allocation-driven.) |
| `transpose` | bool | False | DMA-transpose path: F<=128 direct, F%128==0 tiled, other shapes unsupported |
| `dge_mode` | `nisa.dge_mode.*` | None | DMA generation mode hint |
| `oob_mode` | `nisa.oob_mode.skip` / None | None | Suppress out-of-bounds DMA faults at boundaries |
| `oob_value` | float | None | Pre-fill SBUF with this value before DMA; requires `oob_mode` |
| `dst` | `nl.ndarray` (SBUF) | None | Pre-allocated SBUF buffer to load into; library allocates one when omitted (using `dtype=` for its dtype) |
| `pattern_override` | `[[stride, count], ...]` | None | Custom HBM AP; replaces the auto-generated pattern |
| `out_shape` | `(P, F)` | None | SBUF allocation shape; required when `pattern_override` is set (unless `dst` is also given) |

#### Validation rules

| # | Rule | Assert excerpt |
|---|---|---|
| 1 | View must be HBM-backed | `load() is only valid on HBM views` |
| 2 | All batch dims must be consumed first | `load() requires all batch dims to be consumed first` |
| 3 | `oob_value` requires `oob_mode` | `oob_value= requires oob_mode=` |
| 4 | `pattern_override` requires `out_shape` or `dst` | `pattern_override= requires out_shape= or dst=` |
| 5 | Single-DMA contract: layout must fit one coalesced DMA (exception: partition-dim `.fold()` issues K DMAs) | trace-time error from layout |

#### Allocation tips

- **Default**: `.load()` allocates fresh SBUF each call. Fine for one-shot tiles.
- **Re-use a buffer in a loop**: pre-allocate via `nt.alloc_tiles(...)` / `nt.alloc_blocks(...)` and pass as `dst=`. Avoids per-iteration allocation.
- **Hoist a column**: `tiles[:, m].load()` packs all tiles into one coalesced DMA. The returned NDSlice is indexed in SBUF (no further DMA).

#### Examples

```python
# Single tile
tile = tiles[i, j].load()
tile.data                           # underlying nl.ndarray

# Hoist a column (single coalesced DMA over a 4x1 slice)
hoisted = tiles[:, m].load()
tile = hoisted[k]                   # SBUF index, no DMA

# Load into a caller-provided buffer
buf = nt.alloc_tiles(tile_size=(128, 512), buffer_type=nl.sbuf, dtype=src.dtype)
tiles[i, j].load(dst=buf.data)

# Boundary tile with pre-fill
tile = tiles[i, j].load(oob_mode=nisa.oob_mode.skip, oob_value=0.0)
```

```
tiles[:, m].load()    (column m of a 4x8 grid, 128x128 tiles)

  HBM (row-major)                       SBUF
  +---+---+---+----+---+---+---+---+    +----+----+----+----+
  |   |   |   | t0 |   |   |   |   | r0 | t0 | t1 | t2 | t3 |  128 partitions
  +---+---+---+----+---+---+---+---+    +----+----+----+----+
  |   |   |   | t1 |   |   |   |   | r1 |<--- free dim ----->|
  +---+---+---+----+---+---+---+---+      128  128  128  128
  |   |   |   | t2 |   |   |   |   | r2
  +---+---+---+----+---+---+---+---+   Tiles packed contiguously along
  |   |   |   | t3 |   |   |   |   | r3 SBUF free dim. Single coalesced
  +---+---+---+----+---+---+---+---+   DMA gathers non-contiguous column
        free dim --->                  tiles via strided access pattern.
```

### `.store()` -- SBUF to HBM

```python
view.store(data, dtype=None, dge_mode=None, oob_mode=None, pattern_override=None)
```

Single coalesced DMA. Mirror of `.load()`.

#### Parameters

| Param | Accepts | Default | Role |
|---|---|---|---|
| `data` | `nl.ndarray` (SBUF) or `.ap()` view | required | SBUF source |
| `dtype` | NKI dtype | data dtype | Cast `data` to this dtype before DMA. Same allocation-driven semantics as `load(dtype=)`: the DMA hardware infers the cast from the dtype mismatch, no separate cast op. |
| `dge_mode` | `nisa.dge_mode.*` | None | DMA generation mode hint |
| `oob_mode` | `nisa.oob_mode.skip` / None | None | Suppress out-of-bounds DMA faults |
| `pattern_override` | `[[stride, count], ...]` | None | Custom HBM AP |

#### Validation rules

| # | Rule | Assert excerpt |
|---|---|---|
| 1 | View must be HBM-backed | `store() is only valid on HBM views` |
| 2 | All batch dims must be consumed first | `store() requires all batch dims to be consumed first` |
| 3 | `data` cannot be an NDSlice (call `.ap()` first) | `expects ndarray or .ap() view, not NDSlice` |
| 4 | Single-DMA contract (same exception for partition-dim `.fold()`) | trace-time error from layout |

#### Examples

```python
# Store a loaded SBUF NDSlice via .ap()
dst_tiles[i, j].store(tile.ap())

# Store a raw nl.ndarray (e.g., a manual SBUF result)
dst_tiles[i, j].store(sbuf_result)

# Coalesced multi-tile store
out_blocks[0, nb].store(acc.ap())
```

### `.stream()` -- rotating buffer with DMA/compute overlap

```python
view.stream(dim=0, buffer_count=2, dtype=None, pattern_override=None, out_shape=None)
```

Pre-allocates `buffer_count` SBUF buffers sized for **one step along `dim`** and walks the view, rotating buffers so DMAs overlap with compute. Returns a `BlockStream` (an internal helper that owns the rotating pool).

#### What is a "rotating buffer"?

A *rotating buffer* (a "slot" in the rest of this section) is one pre-allocated SBUF region that holds **exactly one streaming step's worth of data** -- the same chunk you'd get from `view[k]`. The chunk size depends on the source view kind:

| Source view | One slot holds |
|---|---|
| Tile-level NDSlice -- single tile column/row (e.g. `tiles[:, m]`) | one **tile** |
| Tile-level NDSlice -- multi-tile slice along the streamed dim | one row/column of **tiles packed together** (single coalesced DMA per slot) |
| Block-level NDSlice (e.g. `blocks[:, b]`) | one **block** = `block_size x tile_size` elements |

`buffer_count` slots are allocated up front; `stream.load(k)` writes into slot `k % buffer_count`. With `buffer_count=2` (double-buffer), iteration `k` overlaps DMA-into-slot-`k+1` with compute-on-slot-`k`.

#### Parameters

| Param | Accepts | Default | Role |
|---|---|---|---|
| `dim` | int in `[0, ndim)` | view's cursor | Iteration axis. Defaults to the view's cursor -- the first dim that still has an iteration-level outer axis. On a fresh view that's dim 0 (or `n_batch_dims` for batched views); on a child of `parent[i, :]` it's dim 1. Pass an explicit int to stream a non-default axis. |
| `buffer_count` | int >= 1 | 2 | Rotating slot count: 2 for double-buffer, 3 for triple-buffer |
| `dtype` | NKI dtype | source dtype | Allocate the rotating SBUF slots at this dtype; same allocation-driven cast semantics as `load(dtype=)`. `stream.load(k, dtype=...)` matching the stream dtype reuses the rotating slot; passing a mismatched dtype on `load(k)` bypasses the slot and allocates a fresh per-call buffer (no DMA/compute overlap that step). |
| `pattern_override` | `[[stride, count], ...]` | None | Custom HBM AP per step (advanced) |
| `out_shape` | `(P, F)` | None | Per-slot SBUF shape; required when `pattern_override` is set |

#### SBUF cost

Each slot holds **one step's worth of elements** along `dim`. Per-slot logical shape:

| Source view | What one slot holds | Per-slot logical shape | Total SBUF (logical elements) |
|---|---|---|---|
| Tile view, `view[:, m]` (one tile column) | one tile | `tile_size` | `N x prod(tile_size)` |
| Tile view, full grid `view`, dim=0 | one row of tiles (packed) | `tile_size[0] x (cols * tile_size[1] * ...)` | `N x tile_size[0] x cols * tile_size[1]` |
| Tile view, full grid `view`, dim=1 | one column of tiles (packed) | `(rows * tile_size[0]) x tile_size[1] * ...` | `N x rows * tile_size[0] x tile_size[1]` |
| Block view `blocks[:, b]` (one block column) | one block (`block_size` tiles) | `block_size * tile_size` | `N x prod(block_size * tile_size)` |

Concrete example -- `tiles=nt.tiles(src, tile_size=(128, 512))` on a `(512, 1024)` source. Tile grid is `(4, 2)`.

| Stream call | Per-slot logical shape | `buffer_count=2` SBUF total |
|---|---|---|
| `tiles[:, 0].stream(buffer_count=2)` | `(128, 512)` (one tile) | 2 x 65,536 = 131,072 elements |
| `tiles[:, 0].stream(buffer_count=3)` | `(128, 512)` | 3 x 65,536 = 196,608 elements |
| `tiles.stream(dim=1, buffer_count=2)` | `(128, 1024)` (one tile column) | 2 x 131,072 = 262,144 elements |

For a block view `nt.blocks(src, tile_size=(128, 256), block_size=(2, 4))`, one slot of `blocks[:, 0].stream()` holds `(2 x 128, 4 x 256) = (256, 1024) = 262,144 logical elements` per slot (SBUF physical layout folds P-tiles into the F axis: `(128, 2048)`).

Heuristic: `total = buffer_count x logical_step_elements`. Use `buffer_count=2` (double-buffer) by default; bump to `3` only when the consumer is small enough that two stages can't hide DMA latency.

#### `BlockStream` operations

The user-facing surface is `stream.load(k)` / `stream[k]` / `stream.store(k)` plus iteration. Slot indexing is implicit: `k % buffer_count` selects the rotating slot.

| Call | Returns | Behavior |
|---|---|---|
| `stream.load(k)` | loaded `NDSlice` (SBUF view of slot `k % buffer_count`) | DMA the next step into the rotating slot; use inside the iteration |
| `stream[k]` | `NDSlice` wrapping slot `k % buffer_count` | No DMA -- retrieves the pre-allocated slot (for output streaming or pre-loaded reads) |
| `stream.store(k)` | -- | DMA from slot `k % buffer_count` back to the HBM view at stream position `k` |
| `for x in stream:` / `stream.tolist()` | iterates pre-bound slot views | Each iteration is a slot whose `.load()` DMAs at the right position |

`stream.load(k)` consumes the streamed dim and places the cursor past it, wrapping if needed -- same semantics as a single-int subscript on the parent. The loaded view exposes the per-step interior structure for further indexing / iteration:

| Stream construction | Loaded slot's view |
|---|---|
| `tiles[:, j].stream().load(k)` | one tile, cursor past iter (single-tile element-level view) |
| `tiles.stream(dim=1).load(j)` | one column packed; cursor wraps to dim 0 (still iterable per-tile via `slot[k]`) |
| `blocks[bi].stream().load(bj)` | one block; cursor wraps to dim 0 to expose the per-block tile grid (`slot[ti, tj]`) |
| `blocks[:, bj].stream().load(bi)` | one block (mirror of the above) |

#### Validation rules

| # | Rule | Assert excerpt |
|---|---|---|
| 1 | `dim` is int in `[0, ndim)` | `dim=... is out of range` |
| 2 | `buffer_count` is int >= 1 | `buffer_count= must be >= 1` |
| 3 | `pattern_override` requires `out_shape` | `pattern_override= requires out_shape=` |

#### What the stream walks (ASCII)

For a 4x4 tile grid with `tile_size=(P, F)`:

```
tiles.shape = (4, 4)              source tile grid

  +------+------+------+------+
  |(0,0) |(0,1) |(0,2) |(0,3) |   row 0
  +------+------+------+------+
  |(1,0) |(1,1) |(1,2) |(1,3) |   row 1
  +------+------+------+------+
  |(2,0) |(2,1) |(2,2) |(2,3) |   row 2
  +------+------+------+------+
  |(3,0) |(3,1) |(3,2) |(3,3) |   row 3
  +------+------+------+------+
```

**Stream a column** (`tiles[:, 0].stream(buffer_count=2)` -- walks dim 0
of the column slice; one tile per slot):

```
slot[0] <- (0, 0)        iteration k=0: stream.load(0) -> slot 0
slot[1] <- (1, 0)        iteration k=1: stream.load(1) -> slot 1
slot[0] <- (2, 0)        iteration k=2: slot 0 is reused (k % 2 == 0)
slot[1] <- (3, 0)        iteration k=3: slot 1 is reused

           +------+
  k=0      |#(0,0)|       slot 0 holds (0, 0); k=1 prefetches (1, 0) into slot 1
           +------+
  k=1      |#(1,0)|       slot 1 holds (1, 0); k=2 prefetches (2, 0) into slot 0
           +------+
  ...
```

**Stream a row** (`tiles[0, :].stream(buffer_count=2)` -- walks dim 0 of
the row slice; one tile per slot):

```
slot[0] <- (0, 0)        k=0
slot[1] <- (0, 1)        k=1
slot[0] <- (0, 2)        k=2 (reuses slot 0)
slot[1] <- (0, 3)        k=3
```

**Stream the full grid along dim=1** (`tiles.stream(dim=1, buffer_count=2)`
-- each step is a full **column** of tiles, one column per slot):

```
slot[0] <- col 0: (0,0), (1,0), (2,0), (3,0)    k=0
slot[1] <- col 1: (0,1), (1,1), (2,1), (3,1)    k=1
slot[0] <- col 2                                 k=2 (reuse)
slot[1] <- col 3                                 k=3

  per-slot logical shape: (rows * P, F)  -- one tile column packed into one buffer
```

**Stream a block column** (`blocks[:, 0].stream(buffer_count=2)` -- each
step is one full **block** = block_size tiles):

```
block_size=(2, 2), so each block is 2x2 tiles.
A block column contains 2 blocks (rows of blocks).

  slot[0] <- block (0, 0):  +------+------+
                            |(0,0) |(0,1) |
                            +------+------+
                            |(1,0) |(1,1) |
                            +------+------+

  slot[1] <- block (1, 0):  +------+------+
                            |(2,0) |(2,1) |
                            +------+------+
                            |(3,0) |(3,1) |
                            +------+------+

  per-slot logical shape: (block_size[0]*P, block_size[1]*F)  -- a whole block per slot
```

Each iteration, the next slot's DMA overlaps with the current slot's
compute; that overlap is what `.stream()` buys you over `.load()`.

#### Examples

```python
# Auto-rotating consume (compiler injects prefetch)
stream = tiles[:, 0].stream(buffer_count=2)
for k in nl.affine_range(K):
    tile = stream.load(k)              # DMA into slot k%2
    nisa.nc_matmul(acc, lhs.data, tile.data)

# Manual prefetch (sequential_range)
stream = tiles[:, 0].stream(buffer_count=2)
stream.load(0)                         # prime
for k in nl.sequential_range(K):
    tile = stream[k]                   # already loaded
    if k + 1 < K:
        stream.load(k + 1)             # prefetch next
    nisa.nc_matmul(acc, lhs.data, tile.data)

# Output streaming (write into rotating slot, then store)
out = out_tiles[:, 0].stream(buffer_count=2)
for k in nl.affine_range(K):
    nisa.tensor_copy(out[k].data, ...) # fill slot k%2
    out.store(k)                       # DMA back to HBM at position k
```

#### `.load()` vs `.stream()` -- which to pick

|  | `.load()` (hoisting) | `.stream()` (rotating) |
|---|---|---|
| **SBUF cost** | All tiles in the slice resident | `buffer_count` slots (1 step each) |
| **DMA overlap with compute** | No | Yes |
| **Data reuse** | Any tile, any time | Each slot valid for one iteration before reuse |
| **Use case** | Stationary operand, small total bytes | Flowing operand, K-dimension iteration |

### `.tolist()` -- Python list of sub-views

```python
view.tolist(dim=None)
```

Materializes sub-views as a Python list. The form NKI's parser iterates natively.

| Param | Accepts | Default | Role |
|---|---|---|---|
| `dim` | int in `[0, ndim)`, or None | None | None follows the grid cursor (auto-drops batch dims); int iterates that dim explicitly, cursor unchanged |

| # | Rule | Assert excerpt |
|---|---|---|
| 1 | `dim` is int or None | `dim= must be int or None` |
| 2 | `dim` in `[0, ndim)` when given | `dim=... is out of range` |

```python
for tile in tiles.tolist():               # cursor-driven
    ...

for col in tiles.tolist(dim=1):           # column-first (non-default axis)
    for tile in col.tolist():
        ...
```

---

## 4. Blocks -- `nt.blocks()`

Groups tiles into fixed-size rectangular blocks for coalesced DMA. The block view is a destination/source descriptor; output accumulation buffers are allocated separately via `nt.alloc_tiles` / `nt.alloc_blocks` (see the "Block output accumulation" subsection below).

**Signature:**

```python
nt.blocks(source, block_size, tile_size=None, access_pattern=None,
          root=None, buffer_type=None)
```

`block_size` is **mandatory**. For a tile-level view (no block structure), use `nt.tiles()`.

```python
blocks = nt.blocks(dst, tile_size=(128, 512), block_size=(2, 2))
blocks.shape          # (2, 4) -- block grid
blocks.block_size     # (2, 2) -- tiles per block
```

#### Parameters

| Param | Accepts | Default | Role |
|---|---|---|---|
| `source` | HBM tensor / SBUF ndarray / `NDSlice` | required | What to tile + group |
| `block_size` | `tuple[int, ...]`, rank >= 2 | required | Tiles per block on each dim; rank <= source rank |
| `tile_size` | `tuple[int, ...]`, rank >= 2 | None | Tile shape; required for raw sources; NDSlice sources inherit |
| `access_pattern` | `[[stride, count], ...]` | None (contiguous) | Source layout descriptor; raw sources only |
| `root` | top-level `nl.ndarray` | None | Parent tensor when `source` is a sliced HBM view |
| `buffer_type` | `nl.sbuf` / `nl.shared_hbm` / `nl.private_hbm` / None | None | Source memory region; `nl.psum` rejected |

To shard a block view across cores, slice it: `nt.blocks(...)[block_range(...), :]`.
See Sec. 10 Multi-Core (LNC) Sharding.

#### Source modes

| Source | Other args | What blocks() does |
|---|---|---|
| Raw HBM tensor | `tile_size=`, `block_size=` required | Construct tile-level view + prepend block axis |
| Raw HBM tensor | + `access_pattern=` / `root=` / `buffer_type=` | Same, with the corresponding tiles()-level effect |
| Raw SBUF ndarray | `tile_size=`, `buffer_type=nl.sbuf`, `block_size=` | Construct tile-level SBUF view + prepend block axis |
| NDSlice (tile or block) | `block_size=` only | **Promote**: prepend block axis on existing view |
| NDSlice | + `tile_size=` | **Re-tile + promote**: rebuild Grid with new tile size, then prepend block axis |

#### Composition matrix

| Source kind \ arg | `tile_size=` | `block_size=` | `access_pattern=` | `root=` | `buffer_type=` |
|---|---|---|---|---|---|
| Raw HBM tensor | required | required | optional | sliced view: required | optional (None / `nl.shared_hbm` / `nl.private_hbm`) |
| Raw SBUF ndarray | required | required | optional | rejected | required (`nl.sbuf`) |
| NDSlice (promote / re-tile) | optional | required | rejected | rejected | rejected |

`nl.psum` is rejected for all sources.

#### Validation rules

`blocks()` inherits all `tiles()` validation (see Sec.1) plus block-level guards:

| # | Rule | Assert excerpt |
|---|---|---|
| 1 | `block_size` is required | `nt.blocks(...): block_size= is required` |
| 2 | `block_size` is tuple/list of positive ints, rank >= 2 | `block_size= must have at least 2 dims (P, F)` |
| 3 | `block_size` rank <= source rank | `block_size has ... dims but source has ... dims` |
| 4 | Raw source requires `tile_size=` | `nt.blocks(<raw source>, ...): tile_size= is required` |
| 5 | NDSlice source rejects `access_pattern=` / `root=` / `buffer_type=` | `... is only for raw sources` |
| 6 | All Sec.1 rules apply (tile_size validation, AP validation, ...) | (forwarded from `_validate_tiles_args`) |

Examples:

```python
# Promote an existing tile view to block level
view = nt.tiles(hbm, tile_size=(128, 512))[block_range(rank, n, total), :]
blocks = nt.blocks(view, block_size=(2, 2))   # slice-narrowed shard preserved

# Re-tile + promote in one call
blocks = nt.blocks(view, tile_size=(64, 512), block_size=(2, 2))

# Slice the block view directly (block-granular sharding)
blocks = nt.blocks(hbm, tile_size=(128, 512), block_size=(2, 2))[
    block_range(rank, n, total), :,
]

# SBUF block view sharded by slicing
sbuf_view = nt.tiles(sbuf_buf, tile_size=(128, 512), buffer_type=nl.sbuf)[:, own]
sbuf_blocks = nt.blocks(sbuf_view, tile_size=(128, 256), block_size=(1, 2))
```

```
  tile grid: 4x8                          block grid: 2x4
  (each cell = one 128x512 tile)          (each cell = one 2x2 block of tiles)

  +-----+-----+-----+-----+-----+-----+-----+-----+    +-----------+-----------+-----------+-----------+
  |(0,0)|(0,1)|(0,2)|(0,3)|(0,4)|(0,5)|(0,6)|(0,7)|    |           |           |           |           |
  +-----+-----+-----+-----+-----+-----+-----+-----+    |  blk(0,0) |  blk(0,1) |  blk(0,2) |  blk(0,3) |
  |(1,0)|(1,1)|(1,2)|(1,3)|(1,4)|(1,5)|(1,6)|(1,7)|    |  2x2 tiles|  2x2 tiles|  2x2 tiles|  2x2 tiles|
  +-----+-----+-----+-----+-----+-----+-----+-----+    +-----------+-----------+-----------+-----------+
  |(2,0)|(2,1)|(2,2)|(2,3)|(2,4)|(2,5)|(2,6)|(2,7)|    |           |           |           |           |
  +-----+-----+-----+-----+-----+-----+-----+-----+    |  blk(1,0) |  blk(1,1) |  blk(1,2) |  blk(1,3) |
  |(3,0)|(3,1)|(3,2)|(3,3)|(3,4)|(3,5)|(3,6)|(3,7)|    |  2x2 tiles|  2x2 tiles|  2x2 tiles|  2x2 tiles|
  +-----+-----+-----+-----+-----+-----+-----+-----+    +-----------+-----------+-----------+-----------+

  blocks.shape = (2, 4)       -- block grid dimensions
  blocks.block_size = (2, 2)  -- tiles per block
  blocks.tile_size = (128, 512)
```

**Block data movement:**

```python
buf = blocks[0, 1].load()       # Single DMA loads 2x2 block
tile = buf[0, 1]                 # Index within block (no DMA)

dst_blocks[0, 1].store(buf.ap()) # Single DMA stores 2x2 block

# Block streaming
stream = blocks[:, 0].stream(buffer_count=2)
block = stream.load(k)           # Load entire block into rotating buffer
```

**Block output accumulation** uses the `nt.alloc_tiles` / `nt.alloc_blocks` factories (see section 13) to allocate a coalesced SBUF buffer, then a single coalesced `.store()` at the end:

```python
# Allocate SBUF for an output block
acc = nt.alloc_tiles(
    tile_size=(TILE_M, TILE_N),
    grid=(M // TILE_M, TILES_IN_BLOCK_N),
    buffer_type=nl.sbuf,
    dtype=lhsT.dtype,
)
nisa.memset(acc.data, 0.0)

# Accumulate into individual tile slots
acc_tile = acc[m, n].data
nisa.tensor_tensor(dst=acc_tile, data1=acc_tile, data2=psum, op=nl.add)

# Coalesced block store
out_blocks[0, nb].store(acc.ap())
```

---

## 5. Iteration Patterns

Two declarative idioms cover every case. The NKI tracer rejects bare
`for x in view:` (NKIObject iteration), so all loops use one of:

### Index-based iteration -- `range(view.shape[d])` + subscript

Use when the body needs the index (pairing with a destination, offset
arithmetic, sharded tile ID, etc.).

```python
rows, cols = tiles.shape
for i in nl.affine_range(rows):          # or sequential_range / range
    for j in nl.affine_range(cols):
        tile = tiles[i, j].load()
        out_tiles[i, j].store(tile.ap())
```

Pick the range function per the desired loop semantic:

| Range | Behavior |
|---|---|
| `nl.affine_range(N)` | Compiler may reorder/parallelize (default tiling choice) |
| `nl.sequential_range(N)` | Strict sequential execution |
| `range(N)` | Compile-time unroll (each iteration emits distinct trace) |

### For-in on `.tolist()` -- no index needed

Use when the body does the same thing to every child and doesn't need its
position. `.tolist()` returns a plain Python list the tracer can iterate.

```python
for tile_view in src_tiles.tolist():
    tile_view.store(process(tile_view.load()).ap())
```

Nested:

```python
for row in blocks.tolist():
    for tile_view in row.tolist():
        tile = tile_view.load()
        ...
```

### `.tolist(dim=N)` -- iterate a non-default axis

`.tolist(dim=d)` materializes sub-views along dim `d` (ignoring the grid
cursor). Use it when you need a specific axis first without changing
view construction.

```python
cols = tiles.tolist(dim=1)               # column-first
for j in range(len(cols)):
    col = cols[j]
    for i in range(col.shape[0]):
        view = col[i]
        ...
```

### Block iteration with tile decomposition

Indexing a block view auto-descends to the tile grid: `blocks.shape` reports the **block grid**, but `blocks[bi, bj].shape` reports the **tile grid inside that block**. No conversion needed.

```python
for bi in range(blocks.shape[0]):
    for bj in range(blocks.shape[1]):
        block_view = blocks[bi, bj]                # tile-grid view inside block
        for ti in range(block_view.shape[0]):
            for tj in range(block_view.shape[1]):
                tile_view = block_view[ti, tj]
                m = bi * 2 + ti                     # global tile coordinates
```

`nt.tiles(block_view)` is also available -- pass no `tile_size=` to strip the block level and iterate at the existing tile size, or pass `tile_size=` to re-tile at a different tile grain in the same call.

### Hoisting + streaming (standard matmul pattern)

```python
for m in range(C_tiles.shape[0]):
    A_loaded = AT_tiles[:, m].load()                          # Hoist LHS (stationary)
    out_row = C_tiles[m]
    for n in range(out_row.shape[0]):
        B_stream = B_tiles[:, n].stream(buffer_count=2)        # Stream RHS (flowing)
        acc = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(acc, 0.0)
        for k in nl.affine_range(K_TILES):
            a_tile = A_loaded[k]                              # SBUF reuse (free)
            b_tile = B_stream.load(k)                         # Pipelined DMA
            nisa.nc_matmul(acc, a_tile.data, b_tile.data)
        out_row[n].store(acc.ap())
```

---

## 6. Access Patterns

### `access_pattern`

`access_pattern=[[stride_d, count_d], ...]` is the **source layout descriptor**.
Each level becomes one dim of the resulting logical view: `stride_d` is the
element stride and `count_d` is the element extent for view dim `d`. When
omitted, neurotile derives the contiguous AP from the source's physical layout.

**AP rank is decoupled from source rank.** The AP defines the logical view's
shape and strides; the source is only required to have enough bytes to hold the
AP's largest addressed offset. So a 3-level AP on a 2-D source is valid (and
produces a 3-D logical view), and a 1-level AP on an N-D source flattens it.

| AP rank vs source rank | Resulting view |
|---|---|
| `len(ap) == source.ndim` | Same rank as source -- common case (strided / windowed reads) |
| `len(ap) > source.ndim` | Higher-rank view; AP exposes structure not present in `source.shape` (e.g. parity dim from `(M, N)` -> `(M, 2, N//2)`). **TODO:** currently requires `tile_size` to match the view shape exactly (single-tile coverage); multi-tile iteration over higher-rank-AP views is not yet supported. |
| `len(ap) < source.ndim` | Lower-rank view; AP collapses source dims into a flat walk |

**Validation:** counts and strides must be positive. The largest addressed
offset (`sum((count_i - 1) * stride_i)`) must be `< product(source.shape)` --
the AP cannot claim elements past the source's end.

AP is orthogonal to tile size -- pass `tile_size=` explicitly to control tile
granularity (`tile_size` rank must match AP rank). It also composes with
`remainder=` and `buffer_type=`. Slice the resulting view to shard across cores.

```python
# Contiguous source layout (equivalent to default derivation)
nt.tiles(src, access_pattern=[[N, M], [1, N]], tile_size=(128, 512))

# Strided source layout: every 2nd row. Dim 0 now has M/2 visible rows.
nt.tiles(src, access_pattern=[[2 * N, M // 2], [1, N]], tile_size=(64, N))

# AP + multi-core sharding via slice
nt.tiles(src, access_pattern=[[N, M], [1, N]], tile_size=(128, 512))[
    block_range(rank, num_shards, total), :,
]
```

Works with `nt.blocks()` as well -- `block_size=` groups tiles defined on top
of the AP-described source layout.

### `pattern_override` and `out_shape`

Some kernels need per-tile AP patterns that differ from natural tensor strides. `pattern_override` on `.load()` and `.store()` replaces the auto-generated `.ap()` pattern while preserving all other metadata (offset, indirect parameters).

```python
tile = src_tiles[0, 0].load(
    pattern_override=[[1024, 32], [2, 512]],   # hand-crafted 2-level pattern
    out_shape=(32, 512),                       # P=32 partitions, F=512 free dim
)
# Result: tile.data[p, f] = src_hbm[p, 2*f]
```

`out_shape` is required with `pattern_override` on `.load()` to specify the SBUF allocation shape as `(P, F)`.

**Constraints:**
- Product of all counts must equal `out_shape[0] * out_shape[1]`
- `out_shape[0]` must not exceed 128 (SBUF partition limit, `MAX_SBUF_PARTITION_ROWS`)
- NKI AP supports at most 4 levels (`MAX_AP_LEVELS`)

**Store with `pattern_override`:**

```python
dst_tiles[0, 0].store(
    tile.ap(),
    pattern_override=[[1024, 32], [2, 512]],
)
# Result: dst_hbm[p, 2*f] = tile.data[p, f]
```

---

## 7. Indirect Indexing

Sections 1-6 cover tile access at compile-time-known grid positions. Production kernels also require **runtime-dynamic** data access -- gathering rows by index, selecting an expert's weights by ID, or shifting a column window by a computed offset. Neurotile exposes these through the same `[]` indexing operator.

### Gather with `vector_offset`

A loaded `NDSlice` with element shape `[P>1, 1]` at position 0 in `[]` triggers `vector_offset` -- each partition entry independently selects which HBM row to fetch:

```python
idx_iter  = nt.tiles(token_position_to_id, tile_size=(128,))
data_iter = nt.tiles(hidden_states, tile_size=(128, H))

for i in range(idx_iter.shape[0]):
    idx  = idx_iter[i].load()                  # load index tile -> SBUF
    tile = data_iter[idx, :].load()            # indirect DMA (gather)
```

```
    index tile (SBUF)         data tensor (HBM)             result (SBUF)
    idx.data = [2,5,0,7]
                              row 0 | a0  a1  ... |
                              row 2 | c0  c1  ... |         ,-----------------,
                              row 5 | f0  f1  ... |         | row 2 (c0 c1..) | <- idx[0]=2
                              row 7 | h0  h1  ... |         | row 5 (f0 f1..) | <- idx[1]=5
                                                            | row 0 (a0 a1..) | <- idx[2]=0
                                                            | row 7 (h0 h1..) | <- idx[3]=7
                                                            +------------------+
```

**Scatter** works symmetrically:

```python
out_iter[idx, :].store(result)              # scatter via vector_offset
```

### Dynamic selection with `scalar_offset`

A scalar loaded view (element shape `[1, 1]`) triggers `scalar_offset` -- a single value shifts the access window uniformly for all rows. Position in `[]` determines `indirect_dim`:

```python
expert_off = offset_view.load()             # .data shape [1, 1]
tile = aff_iter[:, expert_off].load()       # all rows, cols shifted by expert_off
```

### Index type detection

| `[]` expression | Index type | NKI mechanism | `indirect_dim` |
|---|---|---|:---:|
| `iter[vector_tile, :]` | SBUF view `[P>1, 1]` | `vector_offset` | 0 |
| `iter[scalar_tile, :, :]` | SBUF view `[1, 1]` | `scalar_offset` | 0 |
| `iter[:, scalar_tile, :]` | SBUF view `[1, 1]` | `scalar_offset` | 1 |
| `iter[int, scalar_tile, :]` | SBUF view `[1, 1]` | `scalar_offset` | 1 |
| `iter[:, :, scalar_tile]` | SBUF view `[1, 1]` | `scalar_offset` | 2 |

At most one indirect index per `[]` operation. `vector_offset` restricted to position 0; `scalar_offset` works at any position. `indirect_dim` is **source-tensor-relative** (it tracks the original tensor dimension, not the surviving-view dimension), so it does not shift when dimensions are consumed during indexing.

### Mixed static and indirect indexing

Static indices (int, loop variable) compute the base offset; the indirect index becomes the dynamic offset:

```python
kv_iter = nt.tiles(k_cache, tile_size=(num_p, d_head))
tile = kv_iter[batch_id, seq_offset, :].load()   # batch_id=static, seq_offset=indirect
```

### Raw `nl.ndarray` as index

The `[]` operator also accepts raw `nl.ndarray` directly -- no wrapping required. Detection uses the same shape rule: `(1, 1)` -> `scalar_offset`; `shape[0] > 1` -> `vector_offset`.

```python
ind_offset = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
nisa.tensor_scalar(ind_offset, load_offset, nl.add, seqlen_offset)
tile = kv_iter[batch_id, ind_offset, :].load()
```

### DMA control: `oob_mode` and `dge_mode`

Optional parameters on `.load()` and `.store()` for indirect DMA:

```python
tile = data_iter[idx, :].load(
    oob_mode=nisa.oob_mode.skip,    # skip OOB lanes (auto-inserts memset if oob_value set)
    dge_mode=nisa.dge_mode.swdge,   # software DGE mode
)
```

When `oob_value` is set, `.load()` auto-inserts `nisa.memset(dst, oob_value)` before the DMA. Stores with `oob_mode.skip` simply skip OOB writes without pre-fill.

### Transpose with indirect indexing

NKI's `dma_transpose` does not support dynamic access patterns. When `.load(transpose=True)` is called on a view with indirect addressing, neurotile uses a two-step approach: indirect load via `nisa.dma_copy`, then transpose in SBUF via `nisa.nc_transpose`.

### Composition with blocks

Indirect indexing composes with `nt.blocks()`:

- **Scalar offset:** Single coalesced DMA, base shifted uniformly for the entire block
- **Vector offset:** One indirect DMA per tile within the block

```python
data_blocks = nt.blocks(hidden_states, tile_size=(128, H), block_size=(4, 1))

# Scalar: single coalesced DMA
block_buf = data_blocks[scalar_tile, :].load()

# Vector: one DMA per tile in the block
block_buf = data_blocks[idx_block, :].load()
```

### Constraints

1. Data tensor must be in HBM (kernel input/output parameter).
2. Index tensor must be in SBUF (a loaded `NDSlice` or raw `nl.ndarray`).
3. `vector_offset` restricted to `indirect_dim=0`.
4. One indirect dimension per DMA.

### `vector_offset` walks the raw source tensor's dim 0

`indirect_dim=0` indexes the **source NKI tensor's actual dim 0** -- not a logical
view. Metadata-only NeuroTile transforms (`nt.tensor_view(t).flatten_dims(...)`,
`.permute(...)`, `.reshape_dim(...)`, etc.) keep the underlying `t` unchanged, so
`vector_offset` still walks `t`'s original dim 0 and OOBs on indices computed for
the logical view.

To shift which axis the vector indices walk, **reshape the raw tensor** (a real
`.reshape()` on the NKI tensor produces a new handle with new dim 0) before
constructing the view:

```python
# Wrong: vector_offset walks original dim 0 (B), not the flattened axis.
flat = nt.tensor_view(kv_cache).flatten_dims(0, 1)            # metadata view
flat_iter = nt.tiles(flat, tile_size=(1, D))
gathered = flat_iter[flat_idx, 0].load()                       # OOB on B*S indices

# Right: reshape produces a new tensor handle whose dim 0 is the desired axis.
kv_flat = kv_cache.reshape((B * S, D))                         # raw NKI reshape
flat_iter = nt.tiles(kv_flat, tile_size=(1, D))
gathered = flat_iter[flat_idx, 0].load()                       # walks B*S
```

This is the canonical pattern for KV-cache gathers and any indirect load that
needs to address a flattened or otherwise-rearranged axis.

---

## 8. Remainder Handling

When a tensor's shape is not evenly divisible by the tile size, the last tile in each dimension is a **remainder tile** -- it contains fewer elements than the full tile size.

### The remainder problem

Tensor `(300, 500)` with `tile_size=(128, 128)` produces a 3x4 grid:

```
P per row-tile:  [128, 128, 44]        <- row 2 has P-remainder (300 - 2*128 = 44)
F per col-tile:  [128, 128, 128, 116]  <- col 3 has F-remainder (500 - 3*128 = 116)
```

For **individual tiles**, the system computes correct sizes automatically. For **coalesced DMA** and **indirect indexing**, `oob_mode` and `oob_value` are needed.

### API: `oob_mode`, `oob_value`, and `is_remainder`

| Parameter | Effect |
|-----------|--------|
| `oob_mode=nisa.oob_mode.skip` | Suppresses OOB DMA faults. Skipped positions retain SBUF contents. |
| `oob_value=<float>` | Pre-fills remainder SBUF slots with the given value before DMA (requires `oob_mode=skip`). |

Both parameters are **opt-in**. Without `oob_mode`, OOB access raises a DMA fault.

The `.is_remainder` attribute is available on every `NDSlice`:

```python
tiles = nt.tiles(tensor, tile_size=(128, 128))

row = tiles[i, :]
if row.is_remainder:
    data = row.load(oob_mode=nisa.oob_mode.skip, oob_value=0.0)
else:
    data = row.load()  # no oob overhead
```

### Bulk remainder handling

Three options for pre-partitioning boundary tiles from clean tiles:

- **`remainder="skip"`** on `nt.tiles()` / `nt.blocks()` -- truncate the grid to evenly divisible tiles. Boundary tiles are excluded entirely from iteration.
- **`view.whole_tiles()`** -- returns only sub-views that are *not* remainder tiles.
- **`view.remainder_tiles()`** -- returns only the remainder sub-views.

```python
tiles = nt.tiles(src, tile_size=(128, 128))

for clean in tiles[0, :].whole_tiles():
    data = clean.load()                 # fast path, no oob_mode

for rem in tiles[0, :].remainder_tiles():
    data = rem.load(oob_mode=nisa.oob_mode.skip, oob_value=0.0)
```

### Remainder handling by access pattern

| Access Pattern | `is_remainder` | Notes |
|---|---|---|
| **Individual tile** (concrete) | `True` if last tile in any dim | Exact size, no `oob_mode` needed |
| **Individual tile** (indirect) | Always `True` | Full tile size, must use `oob_mode` |
| **Row slice** (`tiles[i, :]`) | `True` if F-rem on last col | 1D row AP, F+P clamped |
| **Column slice** (`tiles[:, j]`) | `True` if P-rem on last row | F-dim SBUF shrinking when all tiles share same F |
| **Range slice** (`tiles[a:b, :]`) | `True` if range includes boundary | Decomposed per tile-row |
| **2D range** (`tiles[a:b, c:d]`) boundary | `True` if any range includes boundary | Decomposed per tile-row |
| **2D range** interior | `False` | No overhead |
| **BlockStream** | Per-block | Per-block remainder detection |

### F-dimension SBUF shrinking

Only F can shrink, and only when all tiles in the slice share the same actual F extent (e.g., column slice with concrete index). P always stays at full `tile_p`.

```python
# Column slice tiles[:, 3]: actual_f = min(128, 500 - 3*128) = 116
# SBUF: (128, 3 * 116) instead of (128, 3 * 128) -- saves 36 elements/partition
```

### Performance: when memset happens

Memset is **targeted** -- only SBUF tile slots with remainder get memset. Full tiles are never touched:

| Scenario | Memset overhead |
|---|---|
| `is_remainder=False` | **Zero** -- no memset, no `oob_skip`, plain DMA |
| 1D row/col, `oob_value` set | **1 memset** -- only last tile slot |
| `oob_mode=skip` without `oob_value` | **No memset** -- OOB positions retain stale SBUF data (fastest remainder path) |

---

## 9. Software Pipelining

### 3-stage pipeline (load/compute/store overlap)

```python
in_stream = in_tiles[:, 0].stream(buffer_count=2)
out_stream = out_tiles[:, 0].stream(buffer_count=2)
in_stream.load(0)                                    # Prologue
for k in nl.sequential_range(N - 1):                 # Steady state
    in_stream.load(k + 1)                            # Load NEXT
    out_tile = out_stream[k]
    nisa.tensor_scalar(out_tile.data, in_stream[k].data, nl.multiply, 2.0)
    out_stream.store(k)                              # Store CURRENT
# Epilogue: process last tile
```

### Triple buffering (`buffer_count=3`)

Deeper overlap -- load, compute, and store on three different buffers simultaneously.

| Pattern | `buffer_count` | Loop primitive | Use case |
|---|---|---|---|
| Implicit streaming | 2 | `nl.affine_range` | Simple streaming |
| Explicit prefetch | 2 | `nl.affine_range` | Matmul with hoist + stream |
| 3-stage pipeline | 2 | `nl.sequential_range` | Element-wise transforms |
| Triple buffer | 3 | `nl.sequential_range` | High-latency DMA |

---

## 10. Multi-Core (LNC) Sharding

Sharding is a slicing concern. To shard a view across cores, slice it.

**Helpers** (in `neurotile`): each returns a Python `slice` object.

```python
block_range(rank, num_shards, total)        # contiguous block per rank
uneven_block_range(rank, num_shards, total) # contiguous, remainder to early ranks
interleaved_range(rank, num_shards, total)  # round-robin (stride = num_shards)
```

The `total` argument counts iteration units on the shard dim:
- `nt.tiles(src, tile_size=...)` -> tiles (`nt.ceiling_div(dim_size, tile_size)`)
- `nt.blocks(src, tile_size, block_size=B)` -> blocks (`nt.ceiling_div(dim_size, B*tile_size)`)

```python
own = nt.block_range(rank=nl.program_id(0),
                     num_shards=nl.num_programs(0),
                     total=nt.ceiling_div(M, 128))

A = nt.tiles(a_input, tile_size=(128, 512))[own, :]   # local to this rank
B = nt.tiles(b_input, tile_size=(128, 512))[own, :]
C = nt.tiles(c_out,   tile_size=(128, 512))[own, :]

for i in range(C.shape[0]):            # i is LOCAL
    for j in range(C.shape[1]):
        a_tile = A[i, j].load()
        ...
```

**Multi-dim sharding** -- compose two slices:

```python
C = nt.tiles(c_out, tile_size=(128, 512))[m_range, n_range]
```

**Runtime vs compile-time `rank`:**
- Compile-time int -- offset folds into `Layout.offset` at trace time.
- Runtime scalar (e.g. `nl.program_id(0)`) -- offset is stored on
  `Layout.indirect` and resolves at runtime. The `block_range`
  helper skips `rank * 1` (Beta 3 parser rejects it) when `owned == 1`.
- `uneven_block_range` with runtime rank requires `total % num_shards == 0`
  (per-rank owned count varies in the uneven case; can't derive from a
  runtime scalar).

| Helper | Distribution | Use case |
|---|---|---|
| `block_range` | Contiguous ranges | Uniform workloads (default) |
| `uneven_block_range` | Contiguous, remainder to earlier ranks | Tile count not divisible by core count |
| `interleaved_range` | Round-robin (stride = `num_shards`) | Non-uniform workloads (causal attention) |

#### `block_range(rank, num_shards, total) -> slice`

| Param | Accepts | Default | Role |
|---|---|---|---|
| `rank` | `int` (compile-time) or runtime scalar (e.g. `nl.program_id(0)`) | required | This core's rank index |
| `num_shards` | positive `int` | required | Total core count; must divide `total` evenly |
| `total` | positive `int` | required | Total unit count on the shard dim (tiles / blocks) |

**Validation rules**

| # | Rule | Assert excerpt |
|---|---|---|
| 1 | `num_shards` is a positive `int` | `block_range: num_shards must be >= 1` / `must be an int` |
| 2 | `total` is a positive `int` | `block_range: total must be >= 1` / `must be an int` |
| 3 | `total % num_shards == 0` | `block_range requires total ... divisible by num_shards ...` |

#### `uneven_block_range(rank, num_shards, total) -> slice`

Same parameters as `block_range`. Distinguishing rule:

| # | Rule | Assert excerpt |
|---|---|---|
| 1-2 | `num_shards`, `total` positive ints | (as above) |
| 3 | Runtime `rank` requires divisibility | `uneven_block_range with runtime rank requires total ... divisible by num_shards ...` |

Compile-time `rank` is required when `total % num_shards != 0` (the per-rank owned count varies and can't be branched at trace time).

#### `interleaved_range(rank, num_shards, total) -> slice`

Same parameters as `block_range`. Returns `slice(rank, total, num_shards)` -- a stepped slice with `step == num_shards`.

| # | Rule | Assert excerpt |
|---|---|---|
| 1-2 | `num_shards`, `total` positive ints | (as above) |
| 3 | `total % num_shards == 0` | `interleaved_range requires total ... divisible by num_shards ...` |

#### `get_shard_info(tensor_shape, tile_size, shard_dim=0, num_shards=None, shard_id=None) -> dict`

Diagnostic helper -- not used by the runtime DMA path. Returns a partition summary for a sharded tile grid.

| Param | Accepts | Default | Role |
|---|---|---|---|
| `tensor_shape` | tuple/list of `int` | required | Source tensor element shape |
| `tile_size` | tuple/list of `int`, same rank as `tensor_shape` | required | Tile shape |
| `shard_dim` | `int` in `[0, len(tensor_shape))` | `0` | Dim along which sharding splits tiles |
| `num_shards` | `int` or `None` | `nl.num_programs(0)` | Total core count |
| `shard_id` | `int` / runtime scalar / `None` | `nl.program_id(0)` | This core's rank |

**Returns** a dict with keys `total_tiles`, `tiles_per_shard`, `shard_id`, `num_shards`, `shard_dim`.

**Validation rules**

| # | Rule | Assert excerpt |
|---|---|---|
| 1 | `tensor_shape` is a tuple/list | `get_shard_info: tensor_shape must be a tuple` |
| 2 | `tile_size` is a tuple/list | `get_shard_info: tile_size must be a tuple` |
| 3 | `tensor_shape` and `tile_size` have the same rank | `get_shard_info: tensor_shape and tile_size must have the same ...` |
| 4 | `shard_dim` is an `int` in range | `get_shard_info: shard_dim ... is out of range ...` |
| 5 | `tensor_shape[shard_dim] % tile_size[shard_dim] == 0` | `get_shard_info: tensor_shape[...] not divisible by tile_size[...]` |

### Constraints

- **Construct slices in the function body**, not at module scope. The Beta 3
  parser does not trace module-level `slice` constants. Helpers always
  construct in body, so calling them inside a kernel is safe.
- **Stepped slices** (`view[a:b:s]`) are legal and route through `Grid.split_peers`
  to produce a peer-walk axis that the AP emitter recognizes as a gap stride.
- **Re-sharding** an already-sharded dim with another slice is allowed -- the
  second slice narrows within the first owned range. Construct a single slice
  from the root tensor when you need replacement (rather than narrowing) semantics.

---

## 11. Tile Operations

### DMA Transpose -- `view.load(transpose=True)`

The only neurotile-provided transpose path. Uses `nisa.dma_transpose` during load.

| F dimension | Mode | Mechanism |
|---|---|---|
| F <= 128 | Direct | Single `nisa.dma_transpose` with 4D AP |
| F > 128, F % 128 == 0 | Tiled | Chunks into F/128 blocks of 128, single `nisa.dma_transpose` with chunked 4D AP |
| F > 128, F % 128 != 0 | Not supported | -- |

```python
tile = view.load(transpose=True)
```

### In-SBUF transpose

Neurotile does not wrap `nisa.nc_transpose`. Call it directly for tiles already in SBUF/PSUM:

```python
psum_tmp = nl.ndarray((tile_k, tile_m), dtype=dtype, buffer=nl.psum)
nisa.nc_transpose(psum_tmp, tile.data)
transposed = nl.ndarray((tile_m, tile_k), dtype=dtype, buffer=nl.sbuf)
nisa.tensor_copy(transposed, psum_tmp)
```

### View transforms (metadata-only, zero runtime cost)

All of these are methods on `NDSlice` and work uniformly on HBM views (before `.load()`) and SBUF views (after `.load()`). They mutate the view's `element_shape` / strides and return a new NDSlice; no DMA, no allocation.

#### Quick reference

| Method | Description | Example |
|---|---|---|
| `.reshape_dim(dim, shape)` | Split a dimension | `(128, 512)` -> `(128, 4, 128)` |
| `.flatten_dims(start_dim, end_dim)` | Merge contiguous dims | `(128, 4, 128)` -> `(128, 512)` |
| `.expand_dim(dim)` | Insert size-1 dim | `(128, 512)` -> `(128, 1, 512)` |
| `.squeeze_dim(dim)` | Remove a size-1 dim | `(128, 1, 512)` -> `(128, 512)` |
| `.permute(dims)` | Reorder dimensions | `(128, 4, 128)` -> `(128, 128, 4)` |
| `.split(dim, n)` | Split dim into n chunks | `(128, 512)` -> `(128, 4, 128)` |
| `.reshape(new_shape)` | Full reshape (contiguous) | `(128, 512)` -> `(128, 2, 256)` |
| `.slice(dim, start, end)` | Sub-select an element range | `(128, 512)` -> `(128, 256)` |
| `.broadcast(dim, size)` | Broadcast a size-1 dim (stride=0) | `(128, 1, 64)` -> `(128, 16, 64)` |
| `.fold(src_dim, into_dim, position="outer")` | Merge non-adjacent dims via DMA recipe | `(P, F, K)` -> `(P, F*K)` (see Fold subsection below) |
| `[key]` | Index/narrow | `nd[:, 2, :]` selects chunk 2 |

#### Parameters and validation (per-method)

| Method | Param | Type | Role | Key validation |
|---|---|---|---|---|
| `.reshape_dim` | `dim` | int in `[0, ndim)` | Dim to split | dim range; product(shape) must equal element_shape[dim] |
| | `shape` | tuple of positive ints | Sub-dim sizes | (downstream) |
| `.flatten_dims` | `start_dim` | int in `[0, ndim)` | First dim to merge | range |
| | `end_dim` | int in `[0, ndim)` | Last dim to merge | range; `start_dim <= end_dim` |
| `.expand_dim` | `dim` | int in `[0, ndim]` | Insert position (end allowed) | (no extra) |
| `.squeeze_dim` | `dim` | int in `[0, ndim)` | Dim to remove | range; element_shape[dim] must be 1 |
| `.permute` | `dims` | permutation of `range(ndim)` | New dim order | SBUF: `dims[0] == 0` (P stays put) |
| `.split` | `dim` | int in `[0, ndim)` | Dim to split | range |
| | `n` | positive int | Chunk count | n divides element_shape[dim] |
| `.reshape` | `new_shape` | tuple of positive ints | Full new shape | total elements unchanged; layout must be contiguous |
| `.slice` | `dim` | int in `[0, ndim)` | Dim to narrow | range |
| | `start`, `end` | ints | Element range `[start, end)` | `0 <= start < end <= element_shape[dim]` |
| `.broadcast` | `dim` | int in `[0, ndim)` | Dim with current size 1 | range; size positive |
| | `size` | positive int | Target broadcast size | (downstream: dim must be size 1) |
| `.fold` | `src_dim`, `into_dim` | ints in `[0, ndim)` | Dim merged away / absorber | both in range; distinct |
| | `position` | `"outer"` / `"inner"` | Merge order in stride math | enum |

#### SBUF constraint (P-axis is hardware-fixed)

For SBUF views, the partition dim (dim 0) cannot be permuted, reshape-dim'd into something else, squeezed away, or have its size changed. Practical rules:

- `.permute(dims)` -- requires `dims[0] == 0`.
- `.reshape_dim(0, ...)` / `.split(0, ...)` -- not supported.
- Element-level `.slice(0, ...)` on SBUF -- routed through SBUF sub-indexing (P-narrow), valid.
- Free-dim transforms (dim >= 1) work as on HBM views.

#### Chained transforms

```python
view = (nt.tensor_view(src_hbm)
    .reshape_dim(dim=1, shape=[8, 128])
    .flatten_dims(start_dim=0, end_dim=1)
    .permute(dims=[1, 0])
)
tile = view.load()   # Single DMA with correct strided access
```

### Fold -- `.fold(src_dim, into_dim, position="outer")`

Merges one dimension into another, reducing `ndim` by 1. Unlike `flatten_dims` (which only works on adjacent dims and is purely metadata), `fold` records a DMA recipe that the next `.load()` / `.store()` applies -- so non-adjacent dims can be merged at DMA time without a real shape rebuild.

| Mode | Trigger | DMA cost |
|---|---|---|
| **Free-dim fold** | `src_dim > 0` and `into_dim > 0` | Single coalesced DMA via stored AP pattern |
| **Partition fold** | `src_dim == 0` or `into_dim == 0` | K separate DMAs (one per slice along the folded P dim) |

The fold recipe is attached to the returned NDSlice as an immutable DMA override; `.load()` / `.store()` route through it transparently. Multiple folds compose -- the AP base threads through the chain.

```python
# Free-dim fold (single DMA)
src_tiles = nt.tiles(src_hbm, tile_size=(P, F, K))
folded = src_tiles[0, 0, 0].fold(2, 1)
tile = folded.load()           # (P, F*K) in SBUF

# Partition fold (K DMAs)
folded = src_tiles[0, 0, 0].fold(2, 0)
tile = folded.load()           # (P*K, F) in SBUF

# Chained: (4, 8, 64, 8) -> fold(3, 2) -> (4, 8, 512) -> fold(0, 1) -> (32, 512)
step1 = src_tiles[0, 0, 0, 0].fold(3, 2)  # free-dim fold (single DMA)
step2 = step1.fold(0, 1)                   # partition fold (multi-DMA, K=4)
tile = step2.load()
```

Fold is symmetric for load and store (round-trip supported).

---

## 12. Allocation and Pools

PSUM buffers are not wrapped by `NDSlice`. Allocate PSUM directly with `nl.ndarray(..., buffer=nl.psum)` for one-off use, or with `nt.psum_pool` for a reusable list of PSUM banks. Only SBUF and HBM are wrapped by the allocation factories below.

### `nt.alloc_tiles` -- allocate a tiled SBUF or HBM buffer

```python
nt.alloc_tiles(tile_size, grid=None, buffer_type=None, dtype=None, element_shape=None)
```

Allocates a buffer and returns an `NDSlice` over it. Pass either `grid=` (tile-aligned: extent = `grid * tile_size`) or `element_shape=` (exact extent; last tile per dim is a remainder when not divisible). Omit both for a single-tile allocation.

#### Parameters

| Param | Accepts | Default | Role |
|---|---|---|---|
| `tile_size` | `tuple[int, ...]`, rank >= 2 | required | Per-tile shape; dim 0 maps to SBUF P axis |
| `grid` | `tuple[int, ...]` | None | Tile-grid shape (one count per `tile_size` dim); mutually exclusive with `element_shape` |
| `buffer_type` | `nl.sbuf` / `nl.shared_hbm` / `nl.private_hbm` | required | Memory region; `nl.psum` rejected (use `nt.psum_pool()`) |
| `dtype` | NKI dtype | required | Element dtype (e.g. `nl.bfloat16`) |
| `element_shape` | `tuple[int, ...]` | None | Logical element extent; mutually exclusive with `grid`. Tile grid is ceil-divided |

#### Validation rules

| # | Rule | Assert excerpt |
|---|---|---|
| 1 | `tile_size` is a tuple/list of positive ints, rank >= 2 | `tile_size= must have at least 2 dims (P, F)` |
| 2 | `buffer_type` is required | `nt.alloc_tiles(...): buffer_type= is required` |
| 3 | `dtype` is required | `nt.alloc_tiles(...): dtype= is required` |
| 4 | `buffer_type` is one of `nl.sbuf` / `nl.shared_hbm` / `nl.private_hbm` | `buffer_type=nl.psum is not supported` |
| 5 | `grid` and `element_shape` are mutually exclusive | `pass either grid= or element_shape=, not both` |
| 6 | `grid` rank == `tile_size` rank, entries are positive ints | `grid has ... dims but tile_size has ... dims` |
| 7 | `element_shape` rank == `tile_size` rank, entries are positive ints | `element_shape has ... dims but tile_size has ... dims` |

#### Examples

```python
# Tile-aligned SBUF accumulator
acc = nt.alloc_tiles(
    tile_size=(128, 512),
    grid=(4, 2),
    buffer_type=nl.sbuf,
    dtype=nl.float32,
)
nisa.memset(acc.data, 0.0)
acc[m, n].data    # tile slot -- an nl.ndarray view
acc.ap()          # contiguous AP over the whole buffer

# Exact-extent SBUF buffer with last F-tile as a remainder
gated = nt.alloc_tiles(
    tile_size=(128, 512),
    element_shape=(128, 1792),    # 1792 = 3*512 + 256 -> last tile is 256-wide
    buffer_type=nl.sbuf,
    dtype=nl.bfloat16,
)
```

### `nt.alloc_blocks` -- allocate a block-structured SBUF or HBM buffer

```python
nt.alloc_blocks(tile_size, block_size, grid=None, buffer_type=None,
                dtype=None, element_shape=None)
```

Like `alloc_tiles` but the returned `NDSlice` carries a block level above the tile grid. Pass either `grid=` (block-aligned: tile grid = `grid * block_size`) or `element_shape=` (exact extent; ceiling-divides into the tile grid). Omit both for a single-block allocation.

#### Parameters

| Param | Accepts | Default | Role |
|---|---|---|---|
| `tile_size` | `tuple[int, ...]`, rank >= 2 | required | Per-tile shape; dim 0 maps to SBUF P axis |
| `block_size` | `tuple[int, ...]`, rank == `tile_size` rank | required | Tiles per block |
| `grid` | `tuple[int, ...]` | None | Block-grid shape; mutually exclusive with `element_shape` |
| `buffer_type` | `nl.sbuf` / `nl.shared_hbm` / `nl.private_hbm` | required | Memory region; `nl.psum` rejected |
| `dtype` | NKI dtype | required | Element dtype |
| `element_shape` | `tuple[int, ...]` | None | Logical element extent; mutually exclusive with `grid` |

#### Validation rules

`alloc_blocks` inherits all `alloc_tiles` validation (see above) plus block-level guards:

| # | Rule | Assert excerpt |
|---|---|---|
| 1 | `block_size` is a tuple/list of positive ints, rank >= 2 | `block_size= must have at least 2 dims (P, F)` |
| 2 | `block_size` rank == `tile_size` rank | `block_size has ... dims but tile_size has ... dims` |
| 3 | All `alloc_tiles` rules apply | (forwarded from `_validate_alloc_tiles_args`) |

#### Examples

```python
# Block-aligned SBUF accumulator: 4x2 block grid, 2x2 tiles per block
out = nt.alloc_blocks(
    tile_size=(128, 512),
    block_size=(2, 2),
    grid=(4, 2),
    buffer_type=nl.sbuf, dtype=nl.bfloat16,
)

# Exact-extent SBUF block buffer with remainder tiles
gated = nt.alloc_blocks(
    tile_size=(128, 512),
    block_size=(1, 4),
    element_shape=(128, 1792),     # 1792 = 3*512 + 256 -> last F-tile is 256-wide
    buffer_type=nl.sbuf, dtype=nl.bfloat16,
)
```

### `nt.psum_pool` -- PSUM accumulator pool

```python
nt.psum_pool(
    tile_size, grid=None, element_shape=None,
    bank_axis=None, bank_ids=None, dtype=None,
)
```

Allocates one `nl.ndarray(buffer=nl.psum)` per grid tile and wraps the tiles in an `NDSlice` over a `PSUMLayout`. Kernels index the pool with the grid shape (`psums[s, i].data`) instead of computing flat indices over a list. PSUM is the per-NeuronCore accumulator memory (8 hardware banks of 2048 elements each).

PSUM is **matmul-output memory**: `nisa.nc_matmul` writes its outputs to PSUM with hardware-accumulate semantics (multiple matmul calls into the same PSUM tile sum), and the matmul engine produces FP32 outputs from bf16/fp8 inputs there. For non-matmul accumulation (element-wise reductions, post-processing), allocate FP32 SBUF directly -- it's the same precision and `store(acc)` performs the FP32 -> source-dtype cast at DMA time, so there's no `tensor_copy` hop. Reserving PSUM banks for non-matmul accumulators wastes scarce PSUM space.

#### Parameters

| Param | Accepts | Default | Role |
|---|---|---|---|
| `tile_size` | `tuple[int, ...]`, rank >= 2 | required | Per-tile (P, F) shape |
| `element_shape` | `tuple[int, ...]`, rank == `tile_size` rank | None | **Recommended.** Actual per-dim extent; ceiling-divided by `tile_size` to derive the grid. Trailing tile on each dim is auto-clamped via `PSUMLayout` when not a multiple of `tile_size` -- handles the partial / remainder case uniformly. Mutually exclusive with `grid` |
| `grid` | `tuple[int, ...]`, rank == `tile_size` rank | None | Tile-aligned tile-grid shape (extent = `grid * tile_size`). Use only when the extent is known to be exactly tile-aligned; otherwise prefer `element_shape=` |
| `bank_axis` | `int` or None | None | Allocation mode selector; see mode table |
| `bank_ids` | tuple/list of bank indices in `[0, 8)`, or None | None | Hardware bank placement; see mode table |
| `dtype` | NKI dtype | `nl.float32` | Element dtype |

#### Allocation modes

`bank_axis` and `bank_ids` together pick one of three allocation modes (the fourth combination is rejected):

| `bank_axis` | `bank_ids` | Mode | `bank_ids` length |
|---|---|---|---|
| `None` | `None` | **Compiler-managed** -- one ndarray per tile, no explicit `address=`; the compiler picks banks. Use when bank placement is not performance-critical. | n/a |
| `None` | provided | **All-fanout** -- every grid tile gets its own physical bank. Use when `tile_f >= 256` (tiles too large to share a bank), or when matmul accumulator independence is required (avoids `NCC_ISCH714`). | `product(grid)`; entries must be unique |
| `int` | provided | **Slot-packed** -- `bank_axis` fans across banks; the remaining grid dims pack as slots within each bank. Use for small `tile_f` to fit multiple matmul outputs per bank. | `grid[bank_axis]` |
| `int` | `None` | **Rejected** -- explicit `bank_axis` is only meaningful with explicit `bank_ids`. | -- |

#### Slot-stride rule (slot-packed mode)

Tiles sharing a physical bank must be at least `slot_stride = max(512, 4 * tile_f)` F-elements apart. This matches the compiler's matmul accumulator-region quantum (verified via raw-NKI probes). Tiles closer together produce `NCC_ISCH714` / `NCC_IBIR110` errors at compile time.

Bank capacity (slot-packed mode): `slots_per_bank * slot_stride <= 2048`. This implies:

| `tile_f` | `slot_stride` | Max `slots_per_bank` |
|---|---|---|
| <= 128 | 512 | 4 |
| 256 | 1024 | 2 |
| 512 | 2048 | 1 (i.e. cannot pack -- use all-fanout) |
| > 512 | 4 * `tile_f` | 1 |

Some F-space inside each bank is intentionally unused so that successive tiles land at distinct accumulator regions.

#### Validation rules

| # | Rule | Assert excerpt |
|---|---|---|
| 1 | `tile_size` is a tuple/list of positive ints, rank >= 2 | `tile_size= must have at least 2 dims (P, F)` |
| 2 | Exactly one of `grid` / `element_shape` is provided | `pass either grid= ... or element_shape=` / `pass either grid= or element_shape=, not both` |
| 3 | `grid` rank == `tile_size` rank, entries are positive ints | `grid has ... dims but tile_size has ... dims` |
| 4 | `element_shape` rank == `tile_size` rank, entries are positive ints | `element_shape has ... dims but tile_size has ... dims` |
| 5 | `bank_axis=int` requires `bank_ids` | `bank_axis= without bank_ids= is not supported` |
| 6 | When given, `bank_ids` entries are `int` in `[0, num_hw_banks)` | `bank_ids[i] must be int` / `bank_ids[i]=N is out of range` |
| 7 | All-fanout mode: `len(bank_ids) == product(grid)`, entries unique | `bank_axis=None expects len(bank_ids)= ...` / `bank_axis=None requires unique bank_ids` |
| 8 | All-fanout mode: `product(grid) <= num_hw_banks` | `grid tiles exceed the 8 available PSUM banks` |
| 9 | Slot-packed mode: `bank_axis` is in `[0, len(tile_size))` | `bank_axis=N is out of range` |
| 10 | Slot-packed mode: `len(bank_ids) == grid[bank_axis]` | `len(bank_ids)=... must match tile_grid_shape[bank_axis]=...` |
| 11 | Slot-packed mode: `grid[bank_axis] <= num_hw_banks` | `tile_grid_shape[bank_axis]=... exceeds the 8 available PSUM banks` |
| 12 | Slot-packed mode: `slots_per_bank * max(512, 4 * tile_f) <= 2048` | `... exceeding the 2048-element PSUM bank capacity` |

#### Examples

Prefer `element_shape=` over `grid=`: it expresses the actual extent and handles tile-aligned and partial-trailing-tile cases uniformly (trailing tile auto-clamps via `PSUMLayout`). Use `grid=` only when you specifically want to drop / over-allocate trailing partial extents.

```python
# Recommended: element_shape= drives the allocation (tile-aligned here
# but the partial case below is the same shape).
psums = nt.psum_pool(tile_size=(128, 512), element_shape=(256, 2048))
for s in range(2):
    for i in range(4):
        nisa.nc_matmul(psums[s, i].data, x, w[s, i])

# Partial trailing tile: actual_h not a multiple of 512. The factory
# ceil-divides; psums[s, last].data auto-clamps to the addressable F.
psums = nt.psum_pool(
    tile_size=(128, 512),
    element_shape=(256, actual_h),
    bank_ids=(0, 1, 2, 3, 4, 5, 6, 7),
)

# All-fanout: every tile on its own bank. Required when tile_f >= 256
# (tiles cannot share a bank).
psums = nt.psum_pool(
    tile_size=(128, 512),
    element_shape=(256, 2048),
    bank_ids=(0, 1, 2, 3, 4, 5, 6, 7),
)

# Slot-packed: tile_f=128 fits 4 tiles per bank (stride 512). Use when
# small matmul outputs need multiple per bank.
psums = nt.psum_pool(
    tile_size=(128, 128),
    element_shape=(256, 512),
    bank_axis=1, bank_ids=(0, 1, 2, 3),
)

# grid= is for tile-aligned allocations where partial trailing tiles are
# undesired (the grid extent rounds up; uses no element_shape clamp).
psums = nt.psum_pool(tile_size=(128, 512), grid=(2, 4))

# One-off PSUM buffer (no pool needed)
psum = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
```


### `nt.tensor_view` -- untiled NDSlice

```python
nt.tensor_view(source, access_pattern=None, buffer_type=None)
```

Returns an element-level `NDSlice` over a raw tensor -- useful as the starting point for reshape / permute / flatten chains before tiling. To re-tile, pass the result to `nt.tiles()`.

#### Parameters

| Param | Accepts | Default | Role |
|---|---|---|---|
| `source` | HBM tensor / SBUF ndarray | required | Raw tensor to view |
| `access_pattern` | `[[stride, count], ...]` | None (contiguous) | Source layout descriptor |
| `buffer_type` | `nl.sbuf` / `nl.shared_hbm` / `nl.private_hbm` / None | None | Source memory region; `nl.psum` rejected |

NDSlice sources are rejected -- chain transforms (`.reshape`, `.permute`, `.flatten_dims`, `.split`, ...) on the existing view directly.

#### Validation rules

| # | Rule | Assert excerpt |
|---|---|---|
| 1 | `source` is a raw tensor, not an NDSlice | `tensor_view is a constructor over a raw tensor` |
| 2 | `buffer_type` is None / `nl.sbuf` / `nl.shared_hbm` / `nl.private_hbm` | `buffer_type=nl.psum is not supported` (or unknown enum value) |
| 3 | `access_pattern` rank == source rank, with `stride > 0` and `count > 0` per dim | `access_pattern has ... levels but source has ... dims` / `(stride/count) must be > 0` |
| 4 | AP `count[d]` <= `source.shape[d]` | `access_pattern[d][1] (count=...) exceeds source.shape[d]=...` |

```python
# HBM source (default)
view = (nt.tensor_view(src_hbm)
    .reshape_dim(1, [8, 128])
    .flatten_dims(0, 1))
tile = view.load()

# SBUF source: pass the underlying nl.ndarray (e.g. from .data)
gamma_bc = nt.tensor_view(gamma_sb.data, buffer_type=nl.sbuf)
gamma_bc = gamma_bc.expand_dim(1).broadcast(1, bxs_tile)
```

### `nt.ceiling_div` -- trace-time integer ceiling division

```python
nt.ceiling_div(a, b)
```

Returns smallest int `q` such that `q * b >= a` -- i.e. `ceil(a / b)`. Pure Python -- safe at trace time. Used for tile-count math when extents are not divisible by tile_size.

| Param | Accepts | Role |
|---|---|---|
| `a` | non-negative int | Dividend |
| `b` | positive int | Divisor |

```python
n_tiles = nt.ceiling_div(M, 128)        # 256 -> 2; 257 -> 3
total_tiles = nt.ceiling_div(M, 128)    # safe with non-divisible M
```

### `nt.largest_divisor` -- trace-time tile group sizing

```python
nt.largest_divisor(n, max_val)
```

Returns the largest int `d` such that `d <= max_val` and `n % d == 0`. Returns 1 when no divisor in range exists (e.g. `n` prime and `max_val < n`). Pure Python -- safe at trace time. Used for block / tile group sizing -- e.g. picking a K-group size that evenly divides K_tiles while fitting the PSUM budget.

| Param | Accepts | Role |
|---|---|---|
| `n` | positive int | Number to find a divisor of |
| `max_val` | positive int | Upper bound on the divisor |

```python
K_GROUP_SIZE = nt.largest_divisor(K_tiles, max_val=8)
```

