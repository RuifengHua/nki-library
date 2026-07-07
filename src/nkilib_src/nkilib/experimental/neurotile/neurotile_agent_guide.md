# NeuroTile -- Agent Usage Guide

How to *write* a kernel. For signatures, parameters, and validation rules see
`doc/neurotile_api_reference.md`.

The library exists so you don't hand-roll index math, buffer rotation, or PSUM
allocation. Reach for the abstractions whenever the alternative is manual
arithmetic on offsets, strides, or bank IDs.

## Picking the right iteration form

- **Need indices for cross-referencing destinations or paired loops:**
  `for i in range(view.shape[0]): view[i, j]`. Don't compute `i*TS:(i+1)*TS` by
  hand -- the factory does it.
- **Just walk every tile:** `for child in view.tolist(): ...`. Nest for inner dims.
- **Iterating a non-default axis:** `view.tolist(dim=N)`.
- **Block view:** index `blocks[bi, bj]` directly; it auto-descends to the tile
  grid. Only call `nt.tiles(view)` when you specifically want the block level
  stripped.
- **Bare `for x in view:` is rejected by the parser.** Always go through
  `tolist()` or `range(...)`.
- **Read tile counts from `view.shape[d]`, not `// TILE_SIZE`.** `K_TILES = K //
  TILE_K` is fragile (floor hides remainders) and duplicates info the API
  already exposes. Use `src_tiles.shape[0]` directly. Assign once
  (`K_TILES = src_tiles.shape[0]`) when reused multiple times in the body.

## Indexing children of consumed views

After `parent[i, :]` / `parent[bi, :]` / `parent[:, j]` consumes a dim, the
child view has its cursor advanced past the consumed dim. A single-int
subscript on the child targets the **cursor's dim**, not dim 0:

```python
row = src_tiles[i, :].load()
for j in range(row.shape[0]):
    tile = row[j]                       # j targets the surviving F-tile dim
    nisa.tensor_scalar(tile.data, tile.data, nl.multiply, 2.0)
```

Don't write `row[:, j]` to "skip past" the consumed dim. The consumed dim is
library state -- not user-addressable. The same applies to block views:
`block_row[bj]` (not `block_row[:, bj]`) drills into one block's interior tile
grid.

`view[i]` and `view[i, :]` produce identical views; drop the trailing default
`:` for readability.

When the parent's only iteration level is the batch dim and the trailing tile
grid is `(1, ..., 1)`, indexing the batch lands directly on a single tile at
element-level -- call `.load()` on the slab; don't subscript further:

```python
src_tiles = nt.tiles(src, tile_size=(P, F))   # src.shape == (B, P, F)
slab = src_tiles[b]                            # already a single tile
tile = slab.load()                             # NOT slab[0, 0].load()
```

## Shaping pools and views to your iteration axes

The single biggest source of clutter in kernels is flat-index arithmetic over
something the API can express directly.

- **PSUM pool: shape it to your natural axes.** If you iterate `(ti, hi)`,
  allocate `nt.psum_pool(grid=(BS, h_count), ...)` and write `psums[ti, hi]`.
  Don't allocate `grid=(BS * h_count, 1)` and index `psums[ti*h_count + hi, 0]`.
- **Use `view.element_shape` / `view.tile_size` / `view.block_size`.** When the
  number you need is already a precomputed attribute, use the attribute. Don't
  rebuild it as `block.block_size[1] * block.tile_size[1]` -- that's
  `block.element_shape[1]`.
- **Prefer `element_shape=` over `grid=` on `psum_pool` / `alloc_tiles` /
  `alloc_blocks`.** `element_shape=` carries the actual extent and auto-clamps
  the trailing partial tile, so the same code path works for tile-aligned and
  remainder cases. Reach for `grid=` only when extent is exactly tile-aligned
  and you specifically want over-allocation.

## Streaming

For *flowing* operands (RHS of matmul, weight K-blocks): use `.stream()`. The
rotating buffers overlap the next DMA with the current compute, which is the
whole point of double-buffering.

For *stationary* operands (LHS hoisted across iterations): one `.load()` up
front, no stream.

Don't manually allocate two buffers and swap them -- `buffer_count=2` is the
declarative form.

**Slot granularity depends on the parent view.** The shape of one rotating
buffer is what one stream step covers; choose the parent shape to match what
you want per slot:

| Parent | One slot holds | Use when |
|---|---|---|
| `tiles[:, j]` | one tile | per-tile compute, one DMA per tile |
| `tiles.stream(dim=1)` | one column packed | per-column work, fewer DMAs, larger SBUF footprint |
| `blocks[bi]` | one block (block_size tiles) | per-block compute, coalesced block DMA |

`.stream()` defaults to the cursor's dim, so `src_blocks[bi].stream(...)`
walks the surviving block dim (no `dim=` needed). Pass `dim=` explicitly only
to override that default.

Annotate the slot shape inline so reviewers don't have to mentally simulate:

```python
# Stream j'th tile column with double buffering -- each buffer is one tile.
col_stream = src_tiles[:, j].stream(buffer_count=2)
```

## PSUM vs SBUF accumulators

PSUM is matmul-output memory. Reach for `nl.psum` when:
- The producer is `nisa.nc_matmul` (matmul writes its outputs to PSUM with
  hardware-accumulate semantics).
- You need FP32 output dtype from bf16/fp8 matmul inputs.

For non-matmul accumulation (element-wise reductions, post-processing), use
**FP32 SBUF directly**. Same precision, fewer ops:

```python
acc = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
nisa.memset(acc, 0.0)
for k in nl.affine_range(K_TILES):
    tile = stream.load(k)
    nisa.tensor_tensor(acc, acc, tile.data, op=nl.add)
dst.store(acc)            # store() casts FP32 -> dst dtype at DMA time
```

The PSUM detour (`nisa.tensor_tensor` into PSUM, then `tensor_copy` PSUM ->
SBUF, then store SBUF) adds one allocation and one copy with no benefit, and
ties up scarce PSUM banks.

## Sharding

Sharding is slicing -- index the view with a slice helper:

```python
r = nt.uneven_block_range(rank=nl.program_id(0), num_shards=nl.num_programs(0), total=N)
view = nt.tiles(src, tile_size=(...))[r, :]
```

**Build the slice in the kernel body, not at module scope** -- module-level
`slice` constants don't trace.

## Anti-patterns

- `for i in range(N): tiles[i*TS:(i+1)*TS]` -- use `view[i]`.
- `K_TILES = K // TILE_K` -- use `view.shape[d]`. The `//` is a floor that
  hides remainder bugs and re-derives info the API already exposes.
- `view[i, :]` when `view[i]` is enough -- the trailing default `:` is a no-op.
- `row[:, j]` / `block_row[:, bj]` referencing a consumed dim -- write `row[j]`
  / `block_row[bj]`. Single-int targets the cursor's dim on a child view.
- Hand-rolled rotating buffers -- use `.stream(buffer_count=2)`.
- `nl.ds(offset, size)` inside a loop -- restructure the view upstream.
- `for row in view:` (bare for-in) -- use `view.tolist()`.
- `psums[i*N + j, 0]` flat-math indexing -- shape the pool to natural axes.
- Recomputing extents the API has as attributes (`block_size * tile_size` instead
  of `element_shape`).
- `nt.tiles(loaded_block)` to "re-tile" a slot from `stream.load(k)` or
  `view[bi, bj].load()` -- the loaded SBUF view already exposes the per-block
  tile grid. Index it directly (`block[ti, tj]`).
- PSUM as a generic accumulator -- PSUM is for `nc_matmul` outputs. For
  element-wise reductions, allocate FP32 SBUF directly; `store()` casts at
  DMA time.
- "Legacy" / "deprecated" / "old form" mentions in code, comments, or
  docstrings. Describe only the current state; past transitions belong in
  CHANGELOG, not inline.

## Shape contracts

Assert at kernel entry so upstream drift fails loud. `element_shape` is the
canonical addressable per-dim extent.

```python
src_v = nt.tiles(src, tile_size=(TS, H0))
assert src_v.element_shape == (M, H), f"got {src_v.element_shape}"
assert src_v.tile_size == (TS, H0), f"got {src_v.tile_size}"
```

For local readability inside the kernel body, **annotate the resulting shape
inline at `.load()` and indexing chain results** -- a reviewer shouldn't have
to mentally simulate the indexing chain to know what `.data` looks like:

```python
tile = block_row[bj].load()              # tile: [128, 256]
slice0 = tile[:, 0, :]                    # [128, 1, 32]
nisa.tensor_scalar(slice0.data, slice0.data, nl.multiply, 2.0)
```

## NKI parser pitfalls

The static parser will reject (or silently mistrace) several Python constructs
inside `@nki.jit` functions. The error message rarely points at the cause -- if
you hit "unsupported expression", `(int, object)`, or "entry function not
found", check this list first.

- **No exceptions.** No `raise`, no `try` / `except`. Use `assert cond, "msg"`.
- **No `is` / `is not`** -- use `==` / `!=`.
- **No `set()`, `sorted()`, `getattr`, `hasattr`, `**kwargs`, `import`** inside
  kernel bodies.
- **No list literals** -- use tuples (`dsts=(dst,)` not `dsts=[dst]`).
- **No string buffer args** -- `nl.sbuf`, `nl.psum`, `nl.shared_hbm`.
- **`int + object`:** when both branches of `isinstance(k, int)` trace and one
  does `offset + k * stride` on a runtime LoopVar, you'll see `(int, object)`.
  Route runtime values through indirect / `nl.ds` paths, not arithmetic.
- **Module-level `slice` constants don't trace** -- build them in the body.
- **LoopVar tiers:** `affine_range` / `sequential_range` are CExpr (full
  arithmetic). `dynamic_range` is scalar -- no arithmetic, no `.ap(offset=)`.
- **Output tensors:** `nl.shared_hbm` (not `nl.hbm`).

## Debugging lessons

- **bf16 rounding looks like a compiler bug.** Before blaming the compiler,
  repro with both small and large magnitudes -- integer test data quantizes to
  bf16 in ways that mask layout / fold mismatches.
- **Use torch end-to-end for bf16** (inputs, reference, matmul). numpy has no
  native bf16; upcast both sides to fp32 at the compare site if needed.
- **`nki.simulate(kernel)(input)`** -- deterministic CPU run; catches logic bugs
  before the device.

## Common kernel skeleton

The structure most production kernels follow:

```python
@nki.jit
def my_kernel(x, w, config):
    M, K = x.shape
    output = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)

    # 1. Shard input/output via slice helpers (built in the body).
    r = nt.uneven_block_range(
        rank=nl.program_id(0), num_shards=nl.num_programs(0),
        total=nt.ceiling_div(M, config.tile_m),
    )
    x_blocks = nt.blocks(x, tile_size=(...), block_size=(...))[r, :]
    out_blocks = nt.blocks(output, tile_size=(...), block_size=(...))[r, :]

    # 2. Stream the flowing operand; stationary operands hoist via .load().
    w_stream = nt.blocks(w, ...).stream(buffer_count=2)
    for bi in range(x_blocks.shape[0]):
        x_sb = x_blocks[bi].load()
        # ... matmul into psum_pool, post-process, store ...
        out_blocks[bi].store(result)

    return output
```
