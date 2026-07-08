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

"""Integration tests for the fused GPT-OSS SWA block kernel (swa_fused_cte).

Covers the production sharding configs:
  - TP4 per-rank shard: (num_q_heads, num_kv_heads) = (16, 2)
  - TP8 per-rank shard: (num_q_heads, num_kv_heads) = (8, 1)

Each config is run with and without a prior segment (prior_tokens in {0, sliding_window}),
always with attention sink tokens, for q_active in {4096, 8192}. head_dim=64, bf16,
sliding_window=128, block_size=128, lnc=2 (head-parallel across the 2 LNC cores).

The kernel reads at most ``sliding_window`` prior tokens from the block KV cache, computes
the SWA layer end-to-end, scatters the freshly-computed (post-RoPE) K and (plain) V for the
active tokens back into the cache, and returns the attention output. The test verifies all
three outputs (``out``, ``k_cache``, ``v_cache``) against the torch reference.
"""

from typing import final

import nki.language as nl
import numpy as np
import pytest
from neuronxcc.starfish.support import dtype as dt

from nkilib_src.nkilib.experimental.attention.swa_fused_cte import swa_fused_cte
from nkilib_src.nkilib.experimental.attention.swa_fused_cte_torch import swa_fused_cte_torch_ref
from test.integration.nkilib.utils.tensor_generators import np_random_sample
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _build_rope_caches(B, S, d_head, prior_len, dtype):
    """Build cos/sin caches for the ACTIVE tokens (absolute positions prior_len..prior_len+S).

    Uses the standard half-split (non-interleaved) RoPE layout: cos/sin are duplicated
    across the two halves so that out = x*cos + rotate_half(x)*sin.
    """
    half = d_head // 2
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, half, dtype=np.float32) / half))
    cos = np.zeros((B, S, d_head), dtype=np.float32)
    sin = np.zeros((B, S, d_head), dtype=np.float32)
    for b in range(B):
        pos = (np.arange(S, dtype=np.float32) + prior_len).reshape(S, 1)  # absolute positions
        freqs = pos * inv_freq.reshape(1, half)  # [S, half]
        c = np.cos(freqs)
        s = np.sin(freqs)
        cos[b] = np.concatenate([c, c], axis=-1)
        sin[b] = np.concatenate([s, s], axis=-1)
    return dt.static_cast(cos, np.float32), dt.static_cast(sin, np.float32)


def generate_inputs(
    *,
    bs,
    num_q_heads,
    num_kv_heads,
    head_dim,
    hidden,
    q_active,
    prior_tokens,
    sliding_window,
    block_size,
    sink_value,
    dtype,
    fp8_packed=False,
    cache_dtype=None,
    k_scale_val=None,
    v_scale_val=None,
):
    """Generate input tensors for the fused SWA kernel test.

    The block KV cache is pre-populated with random (post-RoPE) prior K and plain V for the
    first ``prior_tokens`` positions; the active region's slots are zeroed (the kernel writes
    them). Block tables map logical token order to physical blocks contiguously.
    """
    np.random.seed(42)
    gen = np_random_sample()

    S = q_active
    W = sliding_window
    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    fused_i = q_dim + 2 * kv_dim

    # The prior occupies logical blocks [0, n_prior_blocks); active tokens follow at blocks
    # n_prior_blocks.. (one block per 128-token tile). prior_tokens may exceed W (the SWA window):
    # only the LAST W prior tokens (the last prior block) are read by the kernel.
    n_prior_blocks = (prior_tokens + block_size - 1) // block_size
    num_active_blocks = S // block_size
    total_blocks = n_prior_blocks + num_active_blocks
    max_blocks_per_seq = total_blocks

    hidden_states = gen(shape=(bs, S, hidden), dtype=dtype)
    qkv_weight = gen(shape=(hidden, fused_i), dtype=dtype)
    op_weight = gen(shape=(q_dim, hidden), dtype=dtype)
    qkv_bias = gen(shape=(1, fused_i), dtype=dtype)  # GPT-OSS attention_bias=True: q/k/v projection bias
    op_bias = gen(shape=(1, hidden), dtype=dtype)  # output-projection bias

    # Block KV cache. Prior blocks [0, n_prior_blocks): random in the valid prior positions
    # [0, prior_tokens), zeroed beyond. Active blocks n_prior_blocks.. are zeroed (the kernel's
    # scatter is what the golden compares against).
    k_cache = gen(shape=(total_blocks, num_kv_heads, block_size, head_dim), dtype=dtype)
    v_cache = gen(shape=(total_blocks, num_kv_heads, block_size, head_dim), dtype=dtype)
    kc = dt.static_cast(k_cache, np.float32)
    vc = dt.static_cast(v_cache, np.float32)
    kc[n_prior_blocks:] = 0.0  # zero the active blocks
    vc[n_prior_blocks:] = 0.0
    # Zero invalid prior slots: positions [prior_tokens, n_prior_blocks*block_size) in the prior region.
    for p in range(prior_tokens, n_prior_blocks * block_size):
        kc[p // block_size, :, p % block_size, :] = 0.0
        vc[p // block_size, :, p % block_size, :] = 0.0
    k_cache = dt.static_cast(kc, dtype)
    v_cache = dt.static_cast(vc, dtype)

    # Packed-FP8 KV cache: quantize (x / scale) and pack 2 consecutive tokens into the trailing
    # length-2 axis: (num_blocks, num_kv_heads, block_size, head_dim) -> (num_blocks, num_kv_heads,
    # block_size//2, head_dim, 2) fp8 (token 2i -> [...,0], 2i+1 -> [...,1]). The kernel dequantizes the
    # prior on load and re-quantizes its active write-back.
    k_scale = v_scale = None
    if fp8_packed:

        def _pack(cache_bf16, sval):
            f = dt.static_cast(cache_bf16, np.float32).reshape(total_blocks, num_kv_heads, block_size, head_dim)
            fp8_max = 448.0 if cache_dtype == nl.float8_e4m3fn else 240.0
            q = np.clip(f / sval, -fp8_max, fp8_max)  # (nb, n_kv, bs, d)
            packed = np.stack([q[:, :, 0::2, :], q[:, :, 1::2, :]], axis=-1)  # (nb, n_kv, bs//2, d, 2)
            return dt.static_cast(packed, cache_dtype)

        k_cache = _pack(k_cache, k_scale_val)
        v_cache = _pack(v_cache, v_scale_val)
        k_scale = np.full((1, 1), k_scale_val, dtype=np.float32)
        v_scale = np.full((1, 1), v_scale_val, dtype=np.float32)

    # block_tables: contiguous logical->physical mapping [0, 1, ..., total_blocks-1], sized EXACTLY
    # to total_blocks (NO tail padding). Production block_tables come from the model runtime and we
    # cannot assume any padding, so the kernel must be safe reading a table that is exactly
    # total_blocks wide. It is: every block_tables read goes through _phys_blk, which reads a SINGLE
    # [1,1] element (no row-speculation, unlike seg_cte's multi-element reads), and the logical index
    # is always in [0, total_blocks) (prior-load max n_prior_blocks-1; scatter max total_blocks-1).
    # q_active is a multiple of block_size, so active K/V updates address whole blocks via
    # block_tables directly (no per-token slot_mapping needed).
    bt = np.arange(total_blocks, dtype=np.int32).reshape(1, total_blocks).repeat(bs, axis=0)
    block_tables = dt.static_cast(bt, nl.int32)

    cos_cache, sin_cache = _build_rope_caches(bs, S, head_dim, prior_tokens, dtype)

    sink = dt.static_cast(np.full((bs, num_q_heads), sink_value, dtype=np.float32), np.float32)
    prior_tokens_t = dt.static_cast(np.full((1, 1), prior_tokens, dtype=np.int32), nl.int32)

    return {
        "hidden_states": hidden_states,
        "qkv_weight": qkv_weight,
        "op_weight": op_weight,
        # In-place updated caches: aliased so the kernel writes and we verify the result.
        "k_cache.must_alias_input": k_cache,
        "v_cache.must_alias_input": v_cache,
        "block_tables": block_tables,
        "cos_cache": cos_cache,
        "sin_cache": sin_cache,
        "sink": sink,
        "prior_tokens": prior_tokens_t,
        "qkv_bias": qkv_bias,
        "op_bias": op_bias,
        "scale": 1.0 / np.sqrt(head_dim),
        "sliding_window": sliding_window,
        "block_size": block_size,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "d_head": head_dim,
        "k_scale": k_scale,
        "v_scale": v_scale,
    }


# GPT-OSS SWA layer config. (num_q_heads, num_kv_heads) per TP rank.
#   TP4 -> (16, 2);  TP8 -> (8, 1).
SHARD_CONFIGS = [
    pytest.param(16, 2, id="tp4"),
    pytest.param(8, 1, id="tp8"),
]

_HIDDEN = 2880
_HEAD_DIM = 64
_SWA = 128
_BLOCK = 128
_SINK = 2.0


@pytest_test_metadata(
    name="SWA Fused CTE",
    tags=["model"],
    pytest_marks=["attention", "swa", "fused", "cte"],
)
@final
class TestSwaFusedCTE:
    """Fused GPT-OSS sliding-window-attention block: QKV+RoPE+SWA+OutputProj."""

    def _run(
        self,
        *,
        test_manager,
        platform_target,
        num_q_heads,
        num_kv_heads,
        q_active,
        prior_tokens,
        block_size=_BLOCK,
        fp8_packed=False,
        cache_dtype=None,
        k_scale_val=None,
        v_scale_val=None,
        # Fused kernel chains three bf16 matmuls (QKV -> attention PV -> output proj), so error
        # compounds beyond the ~1% a single attention matmul sees (worst tile ~3.5% rel on
        # large-magnitude outputs). The K/V caches match the golden exactly, which confirms the
        # QKV/RoPE/scatter arithmetic is correct and the `out` gap is bf16 accumulation only.
        # NOTE: 1e-2 was tried with centered [-1,1) inputs but FAILS -- centering introduces
        # cancellation (near-zero outputs whose bf16 abs-error is large RELATIVE to the tiny true
        # value) and shrinks max|ref|, so the global-scalar tol (atol+rtol*max|ref|) gets tighter
        # while worst-element rel-error rises to ~24%. All-positive [0,1) inputs keep max-abs-err /
        # max-abs-ref at ~0.05%, so 4e-2 passes with large margin. Keep 4e-2.
        rtol=4e-2,
        atol=1e-2,
    ):
        bs = 1

        def input_generator(test_config, input_tensor_def=None):
            return generate_inputs(
                bs=bs,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=_HEAD_DIM,
                hidden=_HIDDEN,
                q_active=q_active,
                prior_tokens=prior_tokens,
                sliding_window=_SWA,
                block_size=block_size,
                sink_value=_SINK,
                dtype=nl.bfloat16,
                fp8_packed=fp8_packed,
                cache_dtype=cache_dtype,
                k_scale_val=k_scale_val,
                v_scale_val=v_scale_val,
            )

        def output_tensor_descriptor(kernel_input):
            S = q_active
            n_prior_blocks = (prior_tokens + block_size - 1) // block_size
            total_blocks = n_prior_blocks + S // block_size  # prior blocks + active blocks
            if fp8_packed:
                return {
                    "out": np.zeros((bs, S, _HIDDEN), dtype=nl.bfloat16),
                    "k_cache": np.zeros((total_blocks, num_kv_heads, block_size // 2, _HEAD_DIM, 2), dtype=cache_dtype),
                    "v_cache": np.zeros((total_blocks, num_kv_heads, block_size // 2, _HEAD_DIM, 2), dtype=cache_dtype),
                }
            return {
                "out": np.zeros((bs, S, _HIDDEN), dtype=nl.bfloat16),
                "k_cache": np.zeros((total_blocks, num_kv_heads, block_size, _HEAD_DIM), dtype=nl.bfloat16),
                "v_cache": np.zeros((total_blocks, num_kv_heads, block_size, _HEAD_DIM), dtype=nl.bfloat16),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=swa_fused_cte,
            torch_ref=torch_ref_wrapper(swa_fused_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensor_descriptor,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            rtol=rtol,
            atol=atol,
        )

    # Tiny simulation-only config for fast correctness iteration (small H/S, runs on CPU).
    @pytest.mark.fast
    @pytest.mark.simulation
    @pytest.mark.parametrize("n_q_heads, n_kv_heads", [(8, 2), (4, 1)], ids=["tp4_tiny", "tp8_tiny"])
    @pytest.mark.parametrize("block_size", [128, 64, 32], ids=["bs128", "bs64", "bs32"])
    @pytest.mark.parametrize(
        "prior_tokens",
        [0, 64, 128, 256],
        ids=["no_prior", "partial_prior", "full_prior", "over_window_prior"],  # 256 > W=128
    )
    def test_swa_fused_cte_tiny(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        n_q_heads,
        n_kv_heads,
        block_size,
        prior_tokens,
    ):
        """Tiny sim-only config: H=256, q_active=256, head_dim=64 — fast accuracy loop.

        Sweeps block_size in {128, 64, 32}: a paged KV cache whose page (block_size) is SMALLER
        than the 128-token tile/window, so each tile/window spans 128//block_size physical blocks.
        """

        def input_generator(test_config, input_tensor_def=None):
            return generate_inputs(
                bs=1,
                num_q_heads=n_q_heads,
                num_kv_heads=n_kv_heads,
                head_dim=64,
                hidden=256,
                q_active=256,
                prior_tokens=prior_tokens,
                sliding_window=128,
                block_size=block_size,
                sink_value=2.0,
                dtype=nl.bfloat16,
            )

        def output_tensor_descriptor(kernel_input):
            n_prior_blocks = (prior_tokens + block_size - 1) // block_size
            total_blocks = n_prior_blocks + 256 // block_size  # prior blocks + active blocks
            return {
                "out": np.zeros((1, 256, 256), dtype=nl.bfloat16),
                "k_cache": np.zeros((total_blocks, n_kv_heads, block_size, 64), dtype=nl.bfloat16),
                "v_cache": np.zeros((total_blocks, n_kv_heads, block_size, 64), dtype=nl.bfloat16),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=swa_fused_cte,
            torch_ref=torch_ref_wrapper(swa_fused_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensor_descriptor,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            # Fused kernel chains three bf16 matmuls (QKV -> attention PV -> output proj),
            # so error compounds beyond the ~1% a single attention matmul sees (matches main test).
            rtol=4e-2,
            atol=1e-2,
        )

    # Packed-FP8 KV cache (tiny sim). The K/V caches are stored packed-FP8 and dequantized on the prior
    # load / quantized on the active write-back; sweeps the sub-tile block_size {128,64,32}.
    @pytest.mark.fast
    @pytest.mark.simulation
    @pytest.mark.parametrize("n_q_heads, n_kv_heads", [(8, 2), (4, 1)], ids=["tp4_tiny", "tp8_tiny"])
    @pytest.mark.parametrize("block_size", [128, 64, 32], ids=["bs128", "bs64", "bs32"])
    @pytest.mark.parametrize("prior_tokens", [0, 64, 128], ids=["no_prior", "partial_prior", "full_prior"])
    def test_swa_fused_cte_tiny_fp8(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        n_q_heads,
        n_kv_heads,
        block_size,
        prior_tokens,
    ):
        """Tiny sim-only packed-FP8 KV-cache config: H=256, q_active=256, head_dim=64, block_size sweep."""
        cache_dtype = nl.float8_e4m3fn
        k_scale_val, v_scale_val = 1.5, 0.5

        def input_generator(test_config, input_tensor_def=None):
            return generate_inputs(
                bs=1,
                num_q_heads=n_q_heads,
                num_kv_heads=n_kv_heads,
                head_dim=64,
                hidden=256,
                q_active=256,
                prior_tokens=prior_tokens,
                sliding_window=128,
                block_size=block_size,
                sink_value=2.0,
                dtype=nl.bfloat16,
                fp8_packed=True,
                cache_dtype=cache_dtype,
                k_scale_val=k_scale_val,
                v_scale_val=v_scale_val,
            )

        def output_tensor_descriptor(kernel_input):
            n_prior_blocks = (prior_tokens + block_size - 1) // block_size
            total_blocks = n_prior_blocks + 256 // block_size
            return {
                "out": np.zeros((1, 256, 256), dtype=nl.bfloat16),
                # Packed-FP8 cache shape: (num_blocks, num_kv_heads, block_size//2, d_head, 2).
                "k_cache": np.zeros((total_blocks, n_kv_heads, block_size // 2, 64, 2), dtype=cache_dtype),
                "v_cache": np.zeros((total_blocks, n_kv_heads, block_size // 2, 64, 2), dtype=cache_dtype),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=swa_fused_cte,
            torch_ref=torch_ref_wrapper(swa_fused_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensor_descriptor,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            # FP8 round-trip widens tolerance vs the bf16 path. `out` (the real result) passes well
            # within this; the K cache is the binding term: it stores the post-RoPE K quantized to fp8,
            # and the kernel's bf16 swap-matmul RoPE rounds slightly differently from the torch fp32
            # RoPE, so ~6% of K elements land one fp8 ULP off. Against the global-scalar comparator
            # (tol = rtol*max|ref|, max|ref|~72 for K) one ULP (=8) is ~11% rel -> need rtol >= ~0.12.
            # V (no RoPE) is bit-exact and the prior blocks are byte-identical.
            rtol=0.15,
            atol=0.15,
        )

    @pytest_marks(["model", "optimal"])
    @pytest.mark.parametrize("n_q_heads, n_kv_heads", SHARD_CONFIGS)
    @pytest.mark.parametrize("q_active", [4096, 8192])
    @pytest.mark.parametrize("prior_tokens", [0, _SWA // 2, _SWA], ids=["no_prior", "partial_prior", "with_prior"])
    def test_swa_fused_cte(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        n_q_heads,
        n_kv_heads,
        q_active,
        prior_tokens,
    ):
        """GPT-OSS SWA fused block: TP4/TP8 x prior/no-prior x q_active in {4k, 8k}, sink=2.0."""
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            num_q_heads=n_q_heads,
            num_kv_heads=n_kv_heads,
            q_active=q_active,
            prior_tokens=prior_tokens,
        )

    # Smaller paged-cache block sizes (32/64): the 128-token tile/window spans 128//block_size
    # physical blocks. Production-scale HW accuracy + perf sweep at q_active=8192, with_prior.
    @pytest_marks(["model"])
    @pytest.mark.parametrize("n_q_heads, n_kv_heads", SHARD_CONFIGS)
    @pytest.mark.parametrize("block_size", [64, 32], ids=["bs64", "bs32"])
    def test_swa_fused_cte_block_size(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        n_q_heads,
        n_kv_heads,
        block_size,
    ):
        """GPT-OSS SWA fused block with a sub-tile paged cache block_size (32/64), TP4/TP8, 8k."""
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            num_q_heads=n_q_heads,
            num_kv_heads=n_kv_heads,
            q_active=8192,
            prior_tokens=_SWA,
            block_size=block_size,
        )

    # Production-scale packed-FP8 KV cache, TP4/TP8, 8k with prior, block_size {128,64,32}.
    @pytest_marks(["model"])
    @pytest.mark.parametrize("n_q_heads, n_kv_heads", SHARD_CONFIGS)
    @pytest.mark.parametrize("block_size", [128, 64, 32], ids=["bs128", "bs64", "bs32"])
    def test_swa_fused_cte_fp8(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        n_q_heads,
        n_kv_heads,
        block_size,
    ):
        """GPT-OSS SWA fused block with packed-FP8 KV cache, TP4/TP8, q_active=8192, with prior."""
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            num_q_heads=n_q_heads,
            num_kv_heads=n_kv_heads,
            q_active=8192,
            prior_tokens=_SWA,
            block_size=block_size,
            fp8_packed=True,
            cache_dtype=nl.float8_e4m3fn,
            k_scale_val=1.5,
            v_scale_val=0.5,
            rtol=0.15,
            atol=0.15,
        )
