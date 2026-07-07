# Input Distributions for Attention Block TKG Testing

## Constants

| Symbol | Value | Meaning |
|--------|-------|---------|
| FP8_MAX | 240 (E4M3), 448 (E4M3FN) | Maximum representable value for the FP8 dtype |
| coverage | 4.0 | Gaussian clipping threshold in units of σ (~0.006% clipping rate) |
| d_head | 64 or 128 | Head dimension (config-dependent) |
| H | 3072–8192 | Hidden dimension (config-dependent) |

> **Note on FP8 formats:** This document uses FP8 E4M3 (max=240) in examples,
> but all formulas use `FP8_MAX` symbolically. The code derives `FP8_MAX` from
> the dtype at runtime via `get_max_positive_value_for_dtype()`, so all
> calculations adapt automatically to E4M3FN (max=448) or future FP8 formats.
> Both formats have 3 mantissa bits, so relative precision properties are
> identical.

## Overview

This document explains how test input distributions are chosen for the
`attention_block_tkg` mega-kernel. It covers the mathematical constraints that
inputs must satisfy, how violations manifest as test failures, and the rationale
for each design choice.

There are three critical constraints:

1. **Fan-in scaling** of projections to keep matmul outputs O(1)
2. **Attention scores** must have `std ≈ O(1)` for softmax stability (follows from 1)
3. **FP8 dequant scales** must be calibrated to the actual data at each stage

## The Attention Score Pipeline

```
X_norm = RMSNorm(X, gamma)               # optional normalization
Q, K, V = QKV_projection(X_norm, W_qkv)  # project input to Q, K, V
Q, K = RoPE(Q, K, cos, sin)              # rotary position embedding
K_fp8, V_fp8 = quantize_to_fp8(K, V, kv_scale)  # optional FP8 KV cache
scores = Q @ K_fp8^T × softmax_scale     # attention scores (softmax_scale absorbs k_scale)
probs = softmax(scores)                   # attention weights
attn_out = probs @ V_fp8                  # weighted sum (in FP8 range)
result = attn_out @ W_out_adjusted        # output projection (W_out absorbs v_scale)
```

### FP8 KV cache scale fusion

The kernel operates on raw FP8 KV values without dequantizing. Scale
compensation is handled as follows:

- **k_scale → softmax_scale**: When `softmax_scale=None` (default=(1/√D)), the kernel
  automatically divides by k_scale to dequantize K in the QK matmul. When
  `softmax_scale` is explicitly provided, the caller must incorporate k_scale
  (e.g., `softmax_scale = scaling / k_scale`).
- **v_scale → W_out**: The caller must always absorb v_scale into W_out or its
  dequant scale to compensate V magnitude in the output projection.

In TKG (token generation), there are two K/V streams:

- **K_active, V_active:** Newly projected from the current token's input X.
- **K_cache, V_cache:** Pre-existing context, stored in FP8 in HBM. Generated
  as test inputs matching the distribution of K_active after cache write.

## Fan-In Scaling

Weight magnitudes must be scaled as `O(1/√fan_in)` to keep matmul outputs O(1):

```
Y = X @ W  →  Var(Y) = H × Var(X) × Var(W)
With Var(W) = 1/H:  Var(Y) = Var(X)  (variance-preserving)
```

For NONE: `W ~ N(0, σ=1/√H)`. For STATIC/ROW: `generate_quant_tensor` with
`fan_in=H` calibrates `w_scale = 1/(std(W_fp8) × √H)`.

## Softmax Stability

Softmax amplifies noise catastrophically when scores are large:

| Score std | Regime | Effect of bf16 noise |
|-----------|--------|---------------------|
| ≪ 1 | Flat | Negligible probability shifts |
| ≈ 1 | Transitional | Moderate shifts, acceptable for testing |
| ≫ 1 | Peaky | Ranking flips → catastrophic errors |

With fan-in scaling and k_scale fusion: `std(score) = std(Q) × std(K)` where
K is the dequantized value (O(1)). For all quantization types (NONE, ROW,
STATIC), `std(score) ≈ 0.33–1.0` depending on RMSNorm. The k_scale fusion
ensures scores stay O(1) even with FP8 KV cache.

## Distribution Walkthrough (STATIC FP8 + kv_quant)

The most complex case. All values assume `rmsnorm_X=True` (default).

### Active token path

```
Step 1: Input X ~ Uniform[-1, 1], std = 0.577

Step 2: RMSNorm → std(X_norm) ≈ 1.0

Step 3: Input FP8 quantization (STATIC only)
  in_scale = coverage × std(X_norm) / FP8_MAX × jitter
  jitter ~ Uniform[0.8, 1.2] (tests robustness to imperfect calibration)
  X_fp8 = clip(X_norm / in_scale, -FP8_MAX, FP8_MAX)
  X_fp8 is Gaussian with std ≈ FP8_MAX / coverage (e.g., 60 for E4M3)

Step 4: QKV projection
  K_active = (X_fp8 @ W_fp8) × w_scale × in_scale
  w_scale from fan_in=H: 1/(std(W_fp8) × √H) with 0.8–1.2× jitter
  std(K_active) ≈ std(X_norm) ≈ 1.0 (variance-preserving)

Step 5: RoPE — preserves magnitude, std(K_active) unchanged
  If qk_norm enabled: std(K_active) overridden to ≈ 1.0

Step 6: FP8 KV cache quantization
  kv_scale = FP8_MAX / (coverage × std(K_active))
  std(K_cache_fp8) = std(K_active) × kv_scale = FP8_MAX / coverage (e.g., 60)
```

### Cache path

```
Step 7: K_cache ~ N(0, FP8_MAX / coverage) clipped to [-FP8_MAX, FP8_MAX], cast to FP8
  Matches K_active_fp8 distribution from Step 6.
```

### Attention and output projection

```
Step 8: Attention scores
  When softmax_scale=None: kernel computes (1/√d_head) / kv_scale automatically
  When softmax_scale is explicit: caller must incorporate kv_scale
  score = Q × effective_softmax_scale @ K_cache_fp8
  std(score) = std(Q) × std(K_active) ≈ 1.0 (k_scale cancels out)

Step 9: attn_out = softmax(scores) @ V_cache_fp8
  std(attn_out) ≈ std(V_cache_fp8) ≈ FP8_MAX / coverage (e.g., 60)

Step 10: Output projection input FP8 quantization (STATIC only)
  in_scale_out = coverage × std(attn_out) / FP8_MAX ≈ 1.0 (with jitter)

Step 11: Output projection
  W_out_adjusted = W_out / v_scale (absorbs V magnitude)
  w_scale_out from fan_in = q_heads × d_head
  For ROW/STATIC: v_scale absorbed into weight_dequant_scale_out
  std(result) ≈ O(1)
```

## Other Configurations

Each is a subset of the STATIC walkthrough:

### NONE (bf16, no kv_quant)
Skip Steps 3, 6, 7, 10. `std(K_active) ≈ 1.0` (with RMSNorm) or `0.577`
(without). `std(score) ≈ 1.0` or `0.33`. No scale calibration needed.

### NONE + kv_quant
Skip Step 3, 10. `kv_scale = FP8_MAX/(coverage × std(K_active))`. Cache
std = FP8_MAX / coverage. k_scale fused into softmax_scale, v_scale fused
into W_out.

### ROW (no kv_quant)
Skip Steps 3, 6, 7, 10. Same as NONE but weights are FP8 with per-row
`fan_in`-calibrated scales. `std(K_active)` same as NONE.

### ROW + kv_quant
Skip Step 3, 10. Same kv_scale as NONE+kv_quant. Per-column w_scale variation
handled by using max/mean ratio in kv_scale derivation. k_scale fused into
softmax_scale, v_scale fused into weight_dequant_scale_out.

### STATIC (no kv_quant)
Skip Steps 6, 7. `std(K_active) ≈ 1.0` (with calibrated in_scale). Output
projection `input_dequant_scale_out` calibrated to `std(attn_out) ≈ 0.577`
(bf16 V cache).

### STATIC + kv_quant
Full pipeline as described in the walkthrough. k_scale fused into softmax_scale,
v_scale fused into weight_dequant_scale_out.

## KV Scale Derivation

```
kv_scale = FP8_MAX / (coverage × std(K_active))
```

Coverage = 4σ. The `std(K_active)` depends on:

| Type | RMSNorm | QK norm | std(K_active) |
|------|---------|---------|---------------|
| NONE/ROW | Off | Off | `√(1/3) ≈ 0.577` |
| NONE/ROW | On | Off | `1.0` |
| STATIC | Off | Off | `≈ 0.577` (calibrated in_scale, variance-preserving) |
| STATIC | On | Off | `≈ 1.0` (calibrated in_scale, variance-preserving) |
| Any | — | On | `1.0` (QK norm overrides) |

For ROW: multiply by `max/mean(w_scale)` for K and V columns separately.

Cache std = `std(K_active) × kv_scale = FP8_MAX / coverage`.

### Coverage tradeoff

Coverage controls the tradeoff between FP8 clipping and quantization precision:

- **Higher coverage** → smaller kv_scale → K/V use less of FP8 range → less
  clipping but more quantization noise (fewer FP8 bins utilized)
- **Lower coverage** → larger kv_scale → K/V fill more of FP8 range → better
  precision but more tail clipping

We use coverage = 4σ. At this level, values use ~(4/4.5)² ≈ 79% of the FP8
range, costing ~1 dB of SQNR — negligible impact on cosine similarity. The
Gaussian clipping rate of 0.006% is well under the 1% assert threshold, with
enough margin for CLT tail heaviness, while keeping FP8 utilization high.

### Jitter

All calibrated scales include random jitter to test robustness to imperfect
calibration:

- **QKV w_scale (STATIC/ROW with fan_in):** 0.8–1.2× jitter around the
  fan_in-derived target. Preserves per-projection variation while keeping
  the magnitude correct. For ROW per-column scales, the random per-column
  ratios are preserved and the mean is shifted to the target.
- **STATIC in_scale:** 0.8–1.2× jitter around `coverage × std(X) / FP8_MAX`.
  Tests that the kernel handles slightly miscalibrated input quantization.
- **Output projection input_dequant_scale_out:** 0.8–1.2× jitter around the
  calibrated value. This is the only scale computed as a fixed constant, so
  jitter adds variation that other scales get naturally from
  `generate_quant_tensor`.

## The KV Clipping Assertion

```python
kernel_assert(k_clip <= 0.01, "Too many K values clipped ...")
```

Catches uncalibrated kv_scale early. A warning would allow silent accuracy
degradation.

## Output Projection Input Scale (STATIC)

The attention output has `std ≈ FP8_MAX / coverage` (from V_cache_fp8).
The `input_dequant_scale_out` must be calibrated:

```
input_dequant_scale_out = coverage × std(attn_out) / FP8_MAX ≈ 1.0
```

With jitter (0.8–1.2×) for robustness. When `kv_quant=False`, `std(attn_out) ≈
std(V_cache_bf16) ≈ 0.577` instead.

Note: the v_scale fusion is applied to `weight_dequant_scale_out`, not
`input_dequant_scale_out`. The attention output magnitude is unchanged
(still ≈ FP8_MAX / coverage when kv_quant=True); the v_scale compensation
happens in the weight scale of the output projection.

## Other Input Tensors

| Tensor | Distribution | Rationale |
|--------|-------------|-----------|
| X | Uniform[-1, 1] | Post-layernorm activations are O(1) |
| RMSNorm gammas | Uniform[0.5, 1.5] | ~1.0 in trained models |
| RoPE cos/sin | Uniform[-1, 1] | Bounded by definition |
| Biases | Uniform[-0.1, 0.1] | Small relative to projection output |
| KV cache | N(0, FP8_MAX/coverage) clipped to FP8 | Matches K_active × kv_scale |
| Attention sink | Uniform[0, 1], shape `[q_heads_attn, 1]` | See below |

### Attention sink

The optional `sink` tensor (one scalar per query head, KVDP-expanded to
`q_heads_attn`) is concatenated onto the score (logit) dimension inside
`attention_tkg`, *after* Q has been scaled by `softmax_scale`/`1/√d`, and is
**not** itself rescaled:

```
score = K_prior @ Q          # Q already scaled → std(score) ≈ 0.33–1.0
score = cat([score, sink])   # sink competes as a raw logit
probs = softmax(score)
```

So the sink must live on the same scale as the post-scale scores. With
`std(score) ≈ O(1)` (see Softmax Stability), an O(1) sink participates
meaningfully in the softmax — neither vanishing nor saturating it to argmax —
which is what exercises the sink path. `Uniform[0, 1]` (mean 0.5, std ≈ 0.29)
is O(1) and matches the distribution used by the standalone `attention_tkg`
test, keeping the two test suites consistent.

> **Note:** the sink configs use the default `softmax_scale`. A custom
> `softmax_scale` (e.g. Gemma's 0.05) shrinks the scores while the sink stays
> O(1), so the sink would dominate the softmax. That is still a valid test, but
> to keep the sink *comparable* to the scores under a custom scale the sink
> magnitude would need to scale with it.

## Kernel vs. Torch Reference: Sources of Divergence

| Source | Magnitude | Applies to |
|--------|-----------|------------|
| bf16 SRAM write quantization | ~0.4% per write | All |
| FP8 KV cache (3 mantissa bits) | ~6% per value | kv_quant |
| FP8 output projection input | ~6% per value | STATIC |
| Softmax amplification | Negligible at std(score) ≤ 1 | All |
| Matmul associativity | < 0.1% | All |

## Tolerance Selection

Tolerances are split by output type because K/V cache values are in the FP8
range (std ≈ FP8_MAX / coverage) while X_out is O(1) after v_scale fusion.

**X_out tolerances:**

| Config type | Cosine threshold | rtol | atol |
|------------|-----------------|------|------|
| Non-quantized (bf16) | 0.99 | 0.015 | 1e-5 |
| FP8 kv_quant | 0.995 | 0.05 | 1.0 |

**K/V cache tolerances (kv_quant only):**

| Metric | Value | Derivation |
|--------|-------|------------|
| Cosine | 0.995 | Same as X_out |
| atol | FP8_MAX / 32 | ULP at typical cache magnitude (1σ = FP8_MAX/4): `(FP8_MAX/4) × 2^(-3)` |
| rtol | 0.06 | FP8 rounding at high magnitudes (2-3σ) can exceed 5% relative error |

The atol covers rounding at small values where rtol contributes little. The rtol
covers rounding at large values where the FP8 step size grows. The combined
allclose check (`|diff| ≤ atol + rtol × |golden|`) ensures coverage across the
full FP8 range. K/V rtol is 6% (vs 5% for X_out) because kernel and golden may
round to different adjacent FP8 values at high magnitudes, producing up to one
full step size of error (~12.5% worst case, ~5-6% typical at 2-3σ).

The atol for K/V cache adapts automatically to the FP8 format: 7.5 for E4M3
(max=240), 14 for E4M3FN (max=448).

## Appendix: Adversarial Validation Results

Experiments injecting known errors into the torch ref inputs to verify the test
setup catches real bugs. Tested with 5 configs (NONE+kv_quant, ROW+kv_quant,
STATIC+kv_quant, NONE no-quant, ROW+kv_quant long-context), all with
block_len=32, rmsnorm_X=True, qk_norm_pre_rope=True, test_bias=True.

### Detection matrix

| Injection | NONE_kvq | ROW_kvq | STATIC_kvq | NONE_noquant | long_ROW_kvq |
|-----------|----------|---------|------------|--------------|--------------|
| skip_rmsnorm | ❌ 0.975 | ❌ 0.970 | ❌ 0.988 | ❌ 0.993 | ❌ 0.997 |
| kv_scale_2x | ❌ 0.207 | ❌ 0.312 | ❌ 0.213 | ✅ N/A | ❌ 0.120 |
| skip_rope | ❌ 0.464 | ❌ 0.505 | ❌ 0.793 | ❌ 0.826 | ❌ 0.893 |
| shuffle_block_table | ❌ 0.983 | ❌ 0.980 | ❌ 0.992 | ❌ 0.988 | ⚠️ 1.000 |
| skip_qk_norm | ❌ 0.976 | ❌ 0.949 | ❌ 0.991 | ❌ 0.993 | ❌ 0.993 |
| remove_qkv_bias | ⚠️ 0.999 | ⚠️ 1.000 | ⚠️ 0.999 | ❌ allclose | ⚠️ 1.000 |
| 2x_softmax_scale | ❌ 0.807 | ❌ 0.710 | ❌ 0.837 | ❌ 0.906 | ❌ 0.856 |
| no_softmax_scale | ❌ 0.208 | ❌ 0.312 | ❌ 0.213 | ❌ 0.072 | ❌ 0.120 |
| remove_output_bias | ❌ 0.978 | ❌ 0.906 | ❌ 0.652 | ❌ 0.863 | ❌ 0.361 |
| cache_idx_plus1 | ❌ K/V fail | ❌ K/V fail | ❌ K/V fail | ❌ K/V fail | ❌ K/V fail |

Legend: ❌ cosine = detected (value shown is X_out cosine), ✅ = correctly passes,
⚠️ = undetected. Threshold: 0.995 for quantized, 0.99 for non-quantized.

**Detection rate: 44/50 injections detected (88%).**

### Undetected injections

1. **remove_qkv_bias (4/5 configs):** QKV bias is Uniform[-0.1, 0.1], small
   relative to projection output O(1). Cosine similarity is insensitive to
   this magnitude of additive change. Detected on NONE_noquant via allclose
   (tighter pass_rate=1.0 for non-quantized configs).

2. **kv_scale_2x on NONE_noquant:** Expected — no kv_quant, so kv_scale is
   not used.

3. **shuffle_block_table on long_ROW_kvq:** With 32768 random cache positions,
   shuffled blocks produce statistically similar attention patterns. Short
   context (2048) detects this because fewer positions make the shuffle more
   distinguishable.

### Sensitivity gaps and mitigations

- **Additive bias errors:** Cosine similarity is fundamentally insensitive to
  constant offsets. Mitigation: use allclose with tighter atol for bias-specific
  validation, or use structured (non-random) inputs where bias shifts produce
  directional changes.
- **Block table shuffle on long context:** Random cache values make shuffled
  blocks indistinguishable at large sequence lengths. Mitigation: use
  position-dependent cache values where a shuffle produces detectably different
  attention patterns.
