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

"""Configuration dataclasses and auto-tuning logic for MXFP8 MLP backward pass matmuls."""

from dataclasses import dataclass
from typing import Dict, Tuple

import nki.language as nl

from ....core.utils.kernel_helpers import div_ceil
from ..common_utils import get_tile_sizes

# Base tile sizes
TILE_M = 128  # M tile size (stationary dimension)
TILE_K = 128  # K tile size after quantization (Q_TILE_K)
TILE_N = 512  # N tile size (moving dimension)
L_TILE_K = 512  # DGT load tile K size (must be 512 for quantize_mx)


@dataclass
class BlockConfig(nl.NKIObject):
    """Blocking factors for matmul tuning.

    TILES_IN_BLOCK_M: Number of M tiles (128 each) to process together
    TILES_IN_BLOCK_N: Number of N tiles (512 each) to process together
    TILES_IN_BLOCK_K: Number of K tiles (512 DGT load -> 128 matmul) to accumulate in PSUM
    """

    TILES_IN_BLOCK_M: int = 8
    TILES_IN_BLOCK_N: int = 1
    TILES_IN_BLOCK_K: int = 8


@dataclass
class MatmulConfig(nl.NKIObject):
    """Per-operation configuration for backward pass matmuls.

    Backward has 4 main operations plus a recompute stage:
    - phase1_down_proj_mm_grad: output_grad[S,H] @ down_weight[H,I] -> [S,I] (S-sharded)
    - phase2_hidden_states_grad: d_gate_up[S,I] @ gate_up_weight[I,H] -> [S,H] (S-sharded)
    - phase3_gate_up_weight_grad: d_gate_up.T[I,S] @ x[S,H] -> [I,H] (I-sharded)
    - phase4_down_weight_grad: output_grad.T[H,S] @ hidden[S,I] -> [H,I] (H-sharded)
    - recompute_gate_up: hidden[S,H] @ gate_up[2I,H].T -> [S,I] (S-sharded, same as fwd gate_up)
    """

    phase1_down_proj_mm_grad: BlockConfig = None
    phase2_hidden_states_grad: BlockConfig = None
    phase3_gate_up_weight_grad: BlockConfig = None
    phase4_down_weight_grad: BlockConfig = None
    recompute_gate_up: BlockConfig = None

    def __post_init__(self):
        """Initialize default BlockConfig for any phase not explicitly provided."""
        if self.phase1_down_proj_mm_grad == None:
            self.phase1_down_proj_mm_grad = BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=8)
        if self.phase2_hidden_states_grad == None:
            self.phase2_hidden_states_grad = BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=8)
        if self.phase3_gate_up_weight_grad == None:
            self.phase3_gate_up_weight_grad = BlockConfig(TILES_IN_BLOCK_M=4, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=8)
        if self.phase4_down_weight_grad == None:
            self.phase4_down_weight_grad = BlockConfig(TILES_IN_BLOCK_M=4, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=8)
        if self.recompute_gate_up == None:
            self.recompute_gate_up = BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=1, TILES_IN_BLOCK_K=8)


DEFAULT_MATMUL_CONFIG = MatmulConfig()


def get_autotuned_config(seq_len: int, hidden_size: int, intermediate_size: int, lnc: int = 2) -> MatmulConfig:
    """Auto-select the best matmul configuration based on input shapes.

    K tiles are 512 elements (L_TILE_K) for DGT.
    """
    # Per-core tile counts for sharded dimensions
    s_per_core = seq_len // lnc if lnc > 1 else seq_len
    i_per_core = intermediate_size // lnc if lnc > 1 else intermediate_size
    h_per_core = hidden_size // lnc if lnc > 1 else hidden_size

    # Per-phase tile sizes (handles non-power-of-2 / sub-tile dimensions)
    # Phase 1: output_grad[S,H] @ down_weight[H,I] -> [S,I]  K=H, M=S, N=I
    down_proj_tiles = get_tile_sizes(hidden_size, s_per_core, intermediate_size)
    # Phase 2: d_gate_up[S,I] @ gate_up_weight[I,H] -> [S,H]  K=I, M=S, N=H
    hidden_grad_tiles = get_tile_sizes(intermediate_size, s_per_core, hidden_size)
    # Phase 3: d_gate_up.T[I,S] @ x[S,H] -> [I,H]  K=S, M=I, N=H
    gate_up_wgrad_tiles = get_tile_sizes(seq_len, i_per_core, hidden_size)
    # Phase 4: output_grad.T[H,S] @ hidden[S,I] -> [H,I]  K=S, M=H, N=I
    down_wgrad_tiles = get_tile_sizes(seq_len, h_per_core, intermediate_size)
    # Recompute: hidden[S,H] @ gate_up[2I,H].T -> [S,I]  K=H, M=S, N=I (same shape as phase 1)
    recompute_tiles = get_tile_sizes(hidden_size, s_per_core, intermediate_size)

    # M dimension tile counts (per core, stationary)
    num_down_proj_m_tiles = div_ceil(s_per_core, down_proj_tiles['tile_m'])
    num_hidden_grad_m_tiles = div_ceil(s_per_core, hidden_grad_tiles['tile_m'])
    num_gate_up_wgrad_m_tiles = div_ceil(i_per_core, gate_up_wgrad_tiles['tile_m'])
    num_down_wgrad_m_tiles = div_ceil(h_per_core, down_wgrad_tiles['tile_m'])

    # K dimension tile counts (using l_tile_k from each phase)
    num_down_proj_k_tiles = div_ceil(hidden_size, down_proj_tiles['l_tile_k'])
    num_hidden_grad_k_tiles = div_ceil(intermediate_size, hidden_grad_tiles['l_tile_k'])
    num_gate_up_wgrad_k_tiles = div_ceil(seq_len, gate_up_wgrad_tiles['l_tile_k'])
    num_down_wgrad_k_tiles = div_ceil(seq_len, down_wgrad_tiles['l_tile_k'])

    # N dimension tile counts
    num_down_proj_n_tiles = div_ceil(intermediate_size, down_proj_tiles['tile_n'])
    num_hidden_grad_n_tiles = div_ceil(hidden_size, hidden_grad_tiles['tile_n'])
    num_gate_up_wgrad_n_tiles = div_ceil(hidden_size, gate_up_wgrad_tiles['tile_n'])
    num_down_wgrad_n_tiles = div_ceil(intermediate_size, down_wgrad_tiles['tile_n'])

    # Recompute tile counts: K=H, M=S (per core), N=I
    num_recompute_m_tiles = div_ceil(s_per_core, recompute_tiles['tile_m'])
    num_recompute_k_tiles = div_ceil(hidden_size, recompute_tiles['l_tile_k'])
    num_recompute_n_tiles = div_ceil(intermediate_size, recompute_tiles['tile_n'])

    def choose_k_blocking(num_k_tiles: int) -> int:
        """
        Default to 1 for mxfp8: DGT+quantize produces data+scale per tile,
        roughly doubling SBUF pressure vs bf16. Higher values can be explored
        but may cause SBUF spills and slowdowns.
        """
        return 1

    def choose_m_blocking(num_m_tiles: int) -> int:
        """Select the largest M blocking factor that evenly divides num_m_tiles."""
        for block_size in [16, 8, 4, 2, 1]:
            if num_m_tiles >= block_size and num_m_tiles % block_size == 0:
                return block_size
        return 1

    def choose_n_blocking(num_n_tiles: int) -> int:
        """Select the largest N blocking factor that evenly divides num_n_tiles."""
        for block_size in [4, 2, 1]:
            if num_n_tiles >= block_size and num_n_tiles % block_size == 0:
                return block_size
        return 1

    # Phase 1: M=S (per core), K=H, N=I
    down_proj_block_m = choose_m_blocking(num_down_proj_m_tiles)
    down_proj_block_k = choose_k_blocking(num_down_proj_k_tiles)
    down_proj_block_n = choose_n_blocking(num_down_proj_n_tiles)

    # Phase 2: M=S (per core), K=I, N=H
    hidden_grad_block_m = choose_m_blocking(num_hidden_grad_m_tiles)
    hidden_grad_block_k = choose_k_blocking(num_hidden_grad_k_tiles)
    hidden_grad_block_n = choose_n_blocking(num_hidden_grad_n_tiles)

    # Phase 3: M=I (per core), K=S, N=H
    gate_up_wgrad_block_m = choose_m_blocking(num_gate_up_wgrad_m_tiles)
    gate_up_wgrad_block_k = choose_k_blocking(num_gate_up_wgrad_k_tiles)
    gate_up_wgrad_block_n = choose_n_blocking(num_gate_up_wgrad_n_tiles)

    # Phase 4: M=H (per core), K=S, N=I
    down_wgrad_block_m = choose_m_blocking(num_down_wgrad_m_tiles)
    down_wgrad_block_k = choose_k_blocking(num_down_wgrad_k_tiles)
    down_wgrad_block_n = choose_n_blocking(num_down_wgrad_n_tiles)

    # Recompute: M=S (per core), K=H, N=I
    recompute_block_m = choose_m_blocking(num_recompute_m_tiles)
    recompute_block_k = choose_k_blocking(num_recompute_k_tiles)
    recompute_block_n = choose_n_blocking(num_recompute_n_tiles)

    return MatmulConfig(
        phase1_down_proj_mm_grad=BlockConfig(
            TILES_IN_BLOCK_M=down_proj_block_m,
            TILES_IN_BLOCK_N=down_proj_block_n,
            TILES_IN_BLOCK_K=down_proj_block_k,
        ),
        phase2_hidden_states_grad=BlockConfig(
            TILES_IN_BLOCK_M=hidden_grad_block_m,
            TILES_IN_BLOCK_N=hidden_grad_block_n,
            TILES_IN_BLOCK_K=hidden_grad_block_k,
        ),
        phase3_gate_up_weight_grad=BlockConfig(
            TILES_IN_BLOCK_M=gate_up_wgrad_block_m,
            TILES_IN_BLOCK_N=gate_up_wgrad_block_n,
            TILES_IN_BLOCK_K=gate_up_wgrad_block_k,
        ),
        phase4_down_weight_grad=BlockConfig(
            TILES_IN_BLOCK_M=down_wgrad_block_m,
            TILES_IN_BLOCK_N=down_wgrad_block_n,
            TILES_IN_BLOCK_K=down_wgrad_block_k,
        ),
        recompute_gate_up=BlockConfig(
            TILES_IN_BLOCK_M=recompute_block_m,
            TILES_IN_BLOCK_N=recompute_block_n,
            TILES_IN_BLOCK_K=recompute_block_k,
        ),
    )


# Shape-specific tuned configs for Qwen3 8B with MXFP8
SHAPE_TUNED_CONFIGS: Dict[Tuple[int, int, int], MatmulConfig] = {
    # TP1: seq=4096, H=4096, I=12288
    (4096, 4096, 12288): MatmulConfig(
        phase1_down_proj_mm_grad=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=8),
        phase2_hidden_states_grad=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=8),
        phase3_gate_up_weight_grad=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=8),
        phase4_down_weight_grad=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=8),
        recompute_gate_up=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=8),
    ),
    # TP2: seq=4096, H=4096, I=6144
    (4096, 4096, 6144): MatmulConfig(
        phase1_down_proj_mm_grad=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=8),
        phase2_hidden_states_grad=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=8),
        phase3_gate_up_weight_grad=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=8),
        phase4_down_weight_grad=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=8),
        recompute_gate_up=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=8),
    ),
    # TP4: seq=4096, H=4096, I=3072
    (4096, 4096, 3072): MatmulConfig(
        phase1_down_proj_mm_grad=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=3, TILES_IN_BLOCK_K=8),
        phase2_hidden_states_grad=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=3),
        phase3_gate_up_weight_grad=BlockConfig(TILES_IN_BLOCK_M=4, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=8),
        phase4_down_weight_grad=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=3, TILES_IN_BLOCK_K=8),
        recompute_gate_up=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=3, TILES_IN_BLOCK_K=8),
    ),
    # TP8: seq=4096, H=4096, I=1536
    (4096, 4096, 1536): MatmulConfig(
        phase1_down_proj_mm_grad=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=3, TILES_IN_BLOCK_K=8),
        phase2_hidden_states_grad=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=8, TILES_IN_BLOCK_K=2),
        phase3_gate_up_weight_grad=BlockConfig(TILES_IN_BLOCK_M=2, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=8),
        phase4_down_weight_grad=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=3, TILES_IN_BLOCK_K=8),
        recompute_gate_up=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=3, TILES_IN_BLOCK_K=8),
    ),
}


def _fix_tiles_in_block_k(tiles_in_block_k: int, num_k_tiles: int) -> int:
    """Reduce TILES_IN_BLOCK_K until it evenly divides num_k_tiles."""
    while tiles_in_block_k > 1 and num_k_tiles % tiles_in_block_k != 0:
        tiles_in_block_k -= 1
    return tiles_in_block_k


def _validate_config(config: MatmulConfig, seq_len: int, hidden_size: int, intermediate_size: int) -> MatmulConfig:
    """Ensure TILES_IN_BLOCK_K divides the K-dimension tile count for each phase."""
    # Phase 1: K=H, Phase 2: K=I, Phase 3: K=S, Phase 4: K=S, Recompute: K=H
    k_tiles = {
        "phase1": div_ceil(hidden_size, L_TILE_K),
        "phase2": div_ceil(intermediate_size, L_TILE_K),
        "phase3": div_ceil(seq_len, L_TILE_K),
        "phase4": div_ceil(seq_len, L_TILE_K),
        "recompute": div_ceil(hidden_size, L_TILE_K),
    }
    phases = [
        ("phase1", config.phase1_down_proj_mm_grad),
        ("phase2", config.phase2_hidden_states_grad),
        ("phase3", config.phase3_gate_up_weight_grad),
        ("phase4", config.phase4_down_weight_grad),
        ("recompute", config.recompute_gate_up),
    ]
    updates = {}
    for name, block_cfg in phases:
        fixed_k = _fix_tiles_in_block_k(block_cfg.TILES_IN_BLOCK_K, k_tiles[name])
        if fixed_k != block_cfg.TILES_IN_BLOCK_K:
            updates[name] = BlockConfig(
                TILES_IN_BLOCK_M=block_cfg.TILES_IN_BLOCK_M,
                TILES_IN_BLOCK_N=block_cfg.TILES_IN_BLOCK_N,
                TILES_IN_BLOCK_K=fixed_k,
            )
    if not updates:
        return config
    return MatmulConfig(
        phase1_down_proj_mm_grad=updates.get("phase1", config.phase1_down_proj_mm_grad),
        phase2_hidden_states_grad=updates.get("phase2", config.phase2_hidden_states_grad),
        phase3_gate_up_weight_grad=updates.get("phase3", config.phase3_gate_up_weight_grad),
        phase4_down_weight_grad=updates.get("phase4", config.phase4_down_weight_grad),
        recompute_gate_up=updates.get("recompute", config.recompute_gate_up),
    )


def get_config_for_shape(seq_len: int, hidden_size: int, intermediate_size: int, lnc: int = 2) -> MatmulConfig:
    """Get the best config for a specific shape."""
    key = (seq_len, hidden_size, intermediate_size)
    if key in SHAPE_TUNED_CONFIGS:
        config = SHAPE_TUNED_CONFIGS[key]
    else:
        config = get_autotuned_config(seq_len, hidden_size, intermediate_size, lnc)
    return _validate_config(config, seq_len, hidden_size, intermediate_size)
