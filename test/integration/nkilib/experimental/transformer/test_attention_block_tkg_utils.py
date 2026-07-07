# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
from dataclasses import MISSING, dataclass, fields
from typing import Any, List, Optional, Union

import nki.language as nl

from nkilib_src.nkilib.core.utils.common_types import QuantizationType
from nkilib_src.nkilib.experimental.transformer.attention_block_tkg_sharding import KVDPCollectiveMode
from test.utils.common_dataclasses import Platforms


class KVScaleTest(enum.Enum):
    """KV cache quantization scale for FP8 test configs.

    Use DEFAULT for standard test scale (240/2.3), or pass a float
    literal. Only used when kv_quant=True.
    """

    DEFAULT = "default"


# Fields excluded from test_id output.
_TEST_ID_EXCLUDED_FIELDS = {"supported_platforms"}


@dataclass
class AttnBlkTestConfig:
    """Test configuration for attention block TKG kernel."""

    batch: int
    q_heads: int
    d_head: int
    H: int
    H_actual: Optional[int]
    S_ctx: int
    S_max_ctx: int
    S_tkg: int

    block_len: int = 0
    update_cache: bool = True
    K_cache_transposed: bool = False
    rmsnorm_X: bool = True
    skip_rope: bool = False
    rope_contiguous_layout: bool = True
    qk_norm_pre_rope: bool = False
    qk_norm_pre_rope_gamma: bool = False
    qk_norm_post_rope: bool = False
    qk_norm_post_rope_gamma: bool = False
    dtype: Any = nl.bfloat16
    quantization_type: QuantizationType = QuantizationType.NONE
    lnc: int = 2
    skip_output_projection: bool = False
    transposed_in: bool = False
    transposed_out: bool = False
    test_bias: bool = False
    input_in_sb: bool = False
    output_in_sb: bool = False
    softmax_scale: Optional[float] = None
    enable_fa_s_prior_tiling: bool = True
    kv_quant: bool = False
    kv_quant_dtype: str = nl.float8_e4m3
    # When True, block-KV cache includes a kv_heads=1 dim at axis 1.
    # E.g. [blocks, 1, block_len, d_head] instead of [blocks, block_len, d_head].
    cache_has_kv_head_dim: bool = False
    kv_scale: Optional[Union[KVScaleTest, float]] = None
    KVDP: int = 1
    DCP: int = 1
    KVDP_collective_mode: KVDPCollectiveMode = KVDPCollectiveMode.ALL_TO_ALL
    # KVDP collective replica group. None = consecutive [[0..KVDP-1]].
    kvdp_replica_group: Optional[List[List[int]]] = None
    skip_attention: bool = False
    # When True, generate a per-q-head attention sink tensor ([q_heads_attn, 1] @ HBM).
    test_sink: bool = False
    use_pos_id: bool = False
    max_context_len: Optional[int] = None
    sliding_window: int = 0
    cache_lens_mean: Optional[float] = None
    cache_lens_stddev: Optional[float] = None
    fp8_packed: bool = False

    # Platform restrictions for this test config. Set to a set of Platforms values
    # to restrict which hardware this config runs on. None means all platforms.
    # Must match the PlatformAware protocol defined in test.utils.common_dataclasses.
    supported_platforms: set[Platforms] | None = None

    # Maximum NeuronCores available on standard shared-fleet instances (trn2.3xlarge).
    # Configs exceeding this threshold require high-rank (48xl) hosts.
    _HIGH_RANK_THRESHOLD = 4

    def is_high_rank(self) -> bool:
        """Whether this config requires more NeuronCores than a standard shared-fleet instance."""
        kvdp_collective_ranks = sum(len(g) for g in self.kvdp_replica_group) if self.kvdp_replica_group else self.KVDP
        return max(kvdp_collective_ranks, self.DCP) > self._HIGH_RANK_THRESHOLD

    def __post_init__(self):
        if self.kv_quant and self.kv_scale is None:
            self.kv_scale = KVScaleTest.DEFAULT

    def test_id(self, prefix: str = "") -> str:
        """Generate a readable test ID with named parameters.

        Includes all required fields (no default) and any optional field whose
        value differs from its default. Use ``-k "field_name-value"`` with pytest
        to filter by any parameter.
        """
        parts = []
        for f in fields(self):
            if f.name in _TEST_ID_EXCLUDED_FIELDS:
                continue
            val = getattr(self, f.name)
            has_default = f.default is not MISSING or f.default_factory is not MISSING
            if has_default and val == f.default:
                continue
            display = val.value if hasattr(val, "value") else val
            parts.append(f"{f.name}-{display}")
        if prefix:
            prefix = f"{prefix}_"

        return prefix + "-".join(parts)
