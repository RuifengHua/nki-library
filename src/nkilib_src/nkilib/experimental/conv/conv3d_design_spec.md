# Conv3D Kernel Design Spec

## Overview

The `conv3d` kernel implements 3D convolution optimized for Trainium2 (trn2) and Trainium3 (trn3). A 3D convolution slides a filter kernel across three spatial dimensions (depth, height, width) of an input tensor, computing a weighted sum at each position to produce an output tensor. The kernel supports configurable stride, asymmetric zero-padding, dilation, bias, optional fused activation, and LNC sharding. It also supports Conv2D (by setting `D=1, K_d=1`) and Conv1D (by setting `D=1, H=1, K_d=1, K_h=1`).

The kernel's strategy is to use the **tensor engine** (`nisa.nc_matmul`) to perform the convolution as a series of matrix multiplications. For each filter position `(k_d, k_h, k_w)`, the convolution reduces along the `C_in` (input channel) dimension with a stationary tensor of the filter weights for a given filter position, with `C_in` on the partition dimension and `C_out` on the free dimension, and a moving tensor corresponding to input values with `C_in` on the partition dimension and and different spatial output positions packed along the free dimension. Each column in the free dimension of the moving tensor corresponds to the input position that the filter at `(k_d, k_h, k_w)` would multiply with for a given location.

By contracting along `C_in` and accumulating across all filter positions into PSUM, we accumulate complete output values using the tensor engine, avoiding the need to use the vector engine for reduction. Once full output values are accumulated in PSUM, we fuse bias addition and activation on the copy out to SBUF then to HBM.

**Output dimensions:**
```
D_out = (D + pad_d_left + pad_d_right - dilation_d * (K_d - 1) - 1) // stride_d + 1
H_out = (H + pad_h_top + pad_h_bottom - dilation_h * (K_h - 1) - 1) // stride_h + 1
W_out = (W + pad_w_left + pad_w_right - dilation_w * (K_w - 1) - 1) // stride_w + 1
```

### Tensor Layout

| Tensor  | Shape                              | Notes |
|---------|------------------------------------|-------|
| Input   | `[B, C_in, D, H, W]`              | Standard channels-first, unchanged from PyTorch |
| Filters | `[K_d, K_h, K_w, C_in, C_out]`    | Reshaped from PyTorch's `[C_out, C_in, K_d, K_h, K_w]` |
| Bias    | `[C_out]`                          | Standard 1D, unchanged from PyTorch |
| Output  | `[B, C_out, D_out, H_out, W_out]` | Standard channels-first, unchanged from PyTorch |

The input, output, and bias tensors use the same layout as standard PyTorch convolution.

The filter tensor is reshaped to `[K_d, K_h, K_w, C_in, C_out]`. This is done because for a given filter position `(k_d, k_h, k_w)`, we need to load a `[C_in, C_out]` slice as the stationary tensor for the matmul. With this layout, the `C_in` and `C_out` dimensions are contiguous in memory for each filter position, making the DMA copies to load filters into SBUF more efficent. As filters are a weight this reshape can be done once on host at a negligible amortized cost.

### Tiling Strategy

The outermost loop is over batch. Within each batch, we keep a set of C_out tiles (filter weights) resident in SBUF, these are the filters needed to fully compute a portion of the output channels across all `C_in` tiles.

For the spatial dimensions, we have three axes: depth (D), height (H), and width (W). Since W values are contiguous in memory, we process multiple output positions along W simultaneously. We also know that for a given `(d, h)` pair, the W output positions are independent and contiguous, so we stack multiple D-H positions along the free dimension of the moving tensor to try to hit the maximum free dimension size (512 elements) for the tensor engine. This is the `num_dh_stacked` parameter.

If `W > 512`, we tile along W as well (`W_tile`).

Within each spatial tile, we iterate over **C_in tiles** for the input. For each C_in tile, we perform a large contiguous DMA copy of the input window from HBM into an SBUF buffer. We then use the vector and scalar engines to scatter the input window into the stacked layout expected by the matmul, where each column in the moving tensor corresponds to the next spatial position that a particular filter position would multiply with. We allocate PSUM banks for the output tiles and accumulate matmul results across all C_in tiles and filter positions. Once we have accumulated complete output values in PSUM, we copy the results out.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW PIPELINE                                        │
│                                                                                  │
│  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐              │
│  │  HBM         │     │  SBUF            │     │  SBUF            │              │
│  │  x_in[C_in,  │ DMA │  Input Window    │ tc  │  Stacked Input   │              │
│  │   D,H,W]     │────►│  [C_in,d_win,    │────►│  [K_REP*C_in,    │              │
│  │              │     │   h_win,w_win]   │     │   num_dh*W_tile] │              │
│  └──────────────┘     └──────────────────┘     └────────┬─────────┘              │
│                        (DMA engine)             (vector/scalar engine)           │
│                                                         │                        │
│                                                         │ nc_matmul              │
│                                                         ▼                        │
│  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐              │
│  │  HBM         │     │  SBUF            │     │  PSUM            │              │
│  │  y_out[C_out,│ DMA │  Result          │ tc  │  Accumulated     │              │
│  │   D_out,     │◄────│  [C_out,         │◄────│  [C_out,         │              │
│  │   H_out,W_out│     │   num_dh*W_tile] │     │   num_dh*W_tile] │              │
│  └──────────────┘     └──────────────────┘     └──────────────────┘              │
│                        (DMA engine)        (vector/scalar engine)                │
│                                            + optional bias/activation            │
│                                                                                  │
│  tc = tensor_copy    DMA = dma_copy    nc_matmul = tensor engine matmul          │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

Pseudocode:

```
for batch in range(B):
    for c_out_group in range(0, C_out, c_out_interleave * P_MAX):

        load filters + bias for this C_out group

        for dh_group in range(dh_start, dh_end, num_dh_stacked):
            for w_tile in range(0, W_out, W_tile):

                allocate PSUM banks for output tiles

                for c_in_tile in range(0, C_in, P_MAX):

                    DMA load input window from HBM → SBUF

                    scatter input window → stacked input layout

                    nc_matmul: accumulate into PSUM

                apply bias + activation, copy PSUM → result SBUF

                DMA copy result SBUF → HBM
```

### Memory Strategy

We use `SbufManager`'s heap allocator to allocate all buffers at kernel start, calculating the total memory needed upfront. Based on the workload characteristics, we determine several interleaving factors:

- **`c_out_interleave`**: How many C_out tiles we process at once. This is prioritized because having more C_out tiles (or ideally all of them) resident in SBUF means we don't have to reload filter data leading to high temporal locality and reuse for the filter weights.

- **`input_window_interleave`**: How many input window buffers we allocate in SBUF. The input window is the contiguous region of `x_in` copied from HBM to SBUF via DMA. Having at least two buffers enables double-buffering (loading the next window while the current one is being scattered). More buffers help when the workload is memory-intensive (large spatial dimensions or large `C_in`).

- **`stacked_input_interleave`**: How many stacked moving tensor buffers we allocate for the actual matmul input. More of these means the vector and scalar engines don't have to wait for a matmul to finish before writing the next scattered input, avoiding memory antidependencies.

- **`w_out_interleave`**: How many result SBUF buffers we allocate for W tiles. More buffers allow overlapping DMA stores of previous results with compute of the current tile.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          SBUF MEMORY LAYOUT                                │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │  Bias buffers       [c_out_interleave] x (P_MAX, 1) fp32         │      │
│  │  (loaded once per C_out  group)                                  │      │
│  ├──────────────────────────────────────────────────────────────────┤      │
│  │  Filter buffers     [c_in_tiles x K_outer_tiles]                 │      │
│  │                     x (stacked_filter_dim, c_out_wide)           │      │
│  │  (loaded once per C_out group)                                   │      │
│  ├──────────────────────────────────────────────────────────────────┤      │
│  │  Result SBUF        [w_out_interleave x c_out_interleave]        │      │
│  │                     x (P_MAX, num_dh_stacked x W_tile)           │      │
│  │  (rotated across W tiles for store/compute overlap)              │      │
│  ├──────────────────────────────────────────────────────────────────┤      │
│  │  Input windows      [input_window_interleave]                    │      │
│  │                     x (P_MAX, d_window, h_window, w_window)      │      │
│  │  (multi-buffered for DMA load / scatter overlap)                 │      │
│  ├──────────────────────────────────────────────────────────────────┤      │
│  │  Stacked inputs     [stacked_input_interleave x K_outer]         │      │
│  │                     x (stacked_filter_dim, effective_free)       │      │
│  │  (multi-buffered for scatter / matmul overlap)                   │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

The memory configuration calculation algorithm starts with maximum interleaving and iteratively reduces until the total SBUF budget is met.

> **Note:** These memory calculations could be improved with a cost model that considers the relative costs of DMA, scatter, and matmul for a given configuration.

### Sharding Strategy

When LNC=2, we either:

- **Shard on D-H** (`shard_on_dh=True`): Each core processes half of the `D_out * H_out` positions and loads all the filters. This is beneficial when the filter data is small relative to the spatial work.

- **Shard on C_out** (`shard_on_dh=False`): Each core processes only half the output channels but must process the entire input window. This is beneficial when the filter data is large.

We determine which strategy to use by comparing the "waste" (load imbalance) of splitting each dimension across two cores: `waste = 2 * ceil(work / 2) - work`. The dimension with less waste is chosen.

> **Note:** This calculation can be improved with a cost model and more analysis. For example, it might be better to shard on C_out in some cases so that both cores avoid reloading filters entirely, even if the D-H split would be more balanced.

### Engine Balancing

The `tensor_copy` operation can run on either the scalar engine or the vector engine. Balancing between these is important for good performance, so we use a configurable modulo factor: every N-th call is sent to one engine vs. the other (default: 1 in 4 calls go to scalar, rest to vector).

Similarly, `memset` can run on the vector engine or GPSIMD engine, and we balance between them with a separate modulo factor (default: 1 in 2 calls go to GPSIMD).

> **Note:** These ratios are currently fixed constants. They could be improved using a cost model that considers whether operations like bias (which requires `tensor_scalar` on the vector engine) or activation (which requires `activation` on the scalar engine) are being used, as those would change the load on each engine.

> **Note:** For `memset`, it may be worth considering whether DMAs use HWDGE (sync engine) vs. SWDGE (GPSIMD) to determine how much work can be offloaded to the GPSIMD.

### K-Replication

When `C_in` is not a multiple of the partition dimension (128), some matmuls will have underutilized tensor engine lanes. To mitigate this when `C_in < 128`, we replicate multiple filter positions along the partition dimension within a single matmul call, packing more useful work into each tensor engine invocation.

The number of positions we can replicate depends on partition access rules:

| C_in range | Partition stride | Max K_REP | Explanation |
|------------|-----------------|-----------|-------------|
| 64 < C_in ≤ 128 | C_in | 1 | Can only access partitions 0–127 from position 0; no room to replicate |
| 32 < C_in ≤ 64 | 64 | 2 | Can access 0–63 from position 0, 64–127 from partition 64 |
| C_in ≤ 32 | 32 | 4 | Can access 0–31 from 0, 32–63 from 32, 64–95 from 64, 96–127 from 96 |

This enables better tensor engine utilization but requires `memset` operations in some cases to zero-fill the empty spaces in the partition dimension not occupied by real data.

### Padding

We only support asymmetric zero-padding currently.

Padding is handled through two mechanisms:

1. **Memset**: Zero-initialize the portions of the stacked input buffer that correspond to padded positions.

2. **Variable free-dimension matmul** (to avoid memsets where possible): Instead of memset-ing padding positions, we adjust the free dimension of the `nc_matmul`. For example, if `W = 128` and we stack 4 D-H groups, the full free dimension would be `4 × 128 = 512`. But if some positions on the edges have padding, we can decrease the free dimension (e.g., `128 → 127`), start accumulating at an offset in PSUM, and use a strided access pattern for the `nc_matmul`. This avoids wasted memset operations and wasted matmul compute on zero-padded positions.

### Fusing Bias and Activation

We support fusing bias addition and activation function on the copy from PSUM to SBUF:

- **Neither bias nor activation**: We do a `tensor_copy` from PSUM to SBUF, which allows us to balance across engines (vector/scalar).
- **Bias only**: We use `tensor_scalar` (vector engine) to add the bias in-flight during the PSUM → SBUF copy.
- **Activation only**: We use `activation` (scalar engine) to apply the activation in-flight during the PSUM → SBUF copy.
- **Both bias and activation**: We first apply bias using `tensor_scalar` (vector engine) from PSUM → SBUF, then apply activation using `activation` (scalar engine) from SBUF → SBUF.

### Stride and Dilation

Stride and dilation are handled by the `tensor_copy` operations that scatter data from the input window buffer into the stacked moving tensor layout. When copying input data for a given filter position and output position, the source indices into the input window account for stride (which output position maps to which input position) and dilation (which filter position maps to which input offset). The scatter logic computes the correct source coordinates using:

```
d_in = d_out * stride_d + k_d * dilation_d - pad_d
h_in = h_out * stride_h + k_h * dilation_h - pad_h
w_in = w_out * stride_w + k_w * dilation_w - pad_w
```

and uses strided `tensor_copy` operations to gather the correct input elements with the appropriate step sizes.
