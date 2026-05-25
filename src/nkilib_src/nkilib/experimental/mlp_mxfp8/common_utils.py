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

"""Shared tile constants and helpers for MXFP8 MLP forward and backward kernels."""

import nki.isa as nisa
import nki.language as nl

from ..mxfp_utils.mxfp8_utils.common_dataclasses import BlockDescriptor, TensorDescriptor
from ..mxfp_utils.mxfp8_utils.quantize_mxfp8_utils import (
    INTERLEAVE_FACTOR,
    get_fp8_dtype_x4,
    get_scale_output_shape,
)

# Base tile sizes (shared across fwd and bwd configs)
TILE_M = 128
TILE_N = 512
L_TILE_K = 512
DGT_MIN_K = 128

# Maximum number of M tiles to load at once in a single DMA transfer
# TODO: Replace with auto-tuned value once perf optimization changes are in
MAX_TILES_IN_LOAD_M = 8

# Number of LNC (Logical Neuron Core) cores for LNC2 sharding
NUM_LNC2_CORES = 2

# Number of fused projections in the gate_up weight matrix (gate + up)
NUM_GATE_UP_PROJECTIONS = 2

# Physical tile K after x4 interleaving (L_TILE_K / INTERLEAVE_FACTOR)
MATMUL_TILE_K_PHYSICAL = L_TILE_K // INTERLEAVE_FACTOR

# Tile shape constants for generic_matmul_mxfp8_api
LHS_MATMUL_TILE_PHYSICAL = (MATMUL_TILE_K_PHYSICAL, TILE_M)
RHS_MATMUL_TILE_PHYSICAL = (MATMUL_TILE_K_PHYSICAL, TILE_N)
LHS_LOAD_TILE = (L_TILE_K, TILE_M)
RHS_LOAD_TILE = (L_TILE_K, TILE_N)
LHS_QUANTIZE_TILE = (MATMUL_TILE_K_PHYSICAL, TILE_M * INTERLEAVE_FACTOR)
RHS_QUANTIZE_TILE = (MATMUL_TILE_K_PHYSICAL, TILE_N * INTERLEAVE_FACTOR)


def get_tile_sizes(K, M, N):
    """Return tile sizes adapted to tensor dimensions, matching matmul kernel auto-generation.

    Args:
        K: Contraction dimension (hidden size for gate/up, intermediate size for down).
        M: Stationary dimension (sequence length).
        N: Moving dimension (intermediate size for gate/up, hidden size for down).

    Returns:
        Dict with keys: tile_m, tile_n, l_tile_k, matmul_tile_k_physical,
        lhs_matmul_tile_physical, rhs_matmul_tile_physical,
        lhs_load_tile, rhs_load_tile, lhs_quantize_tile, rhs_quantize_tile.
    """
    tile_m = TILE_M if M >= TILE_M else M
    tile_n = TILE_N if N >= TILE_N else N
    l_tile_k = L_TILE_K if K >= L_TILE_K else K
    matmul_tile_k_physical = l_tile_k // INTERLEAVE_FACTOR

    return {
        'tile_m': tile_m,
        'tile_n': tile_n,
        'l_tile_k': l_tile_k,
        'matmul_tile_k_physical': matmul_tile_k_physical,
        'lhs_matmul_tile_physical': (matmul_tile_k_physical, tile_m),
        'rhs_matmul_tile_physical': (matmul_tile_k_physical, tile_n),
        'lhs_load_tile': (l_tile_k, tile_m),
        'rhs_load_tile': (l_tile_k, tile_n),
        'lhs_quantize_tile': (matmul_tile_k_physical, tile_m * INTERLEAVE_FACTOR),
        'rhs_quantize_tile': (matmul_tile_k_physical, tile_n * INTERLEAVE_FACTOR),
    }


def _build_matmul_params(
    tiles_in_block_m: int,
    tiles_in_block_n: int,
    tiles_in_block_k: int,
    lhs_load_tile_shape: tuple = None,
    rhs_load_tile_shape: tuple = None,
    tiles: dict = None,
) -> BlockDescriptor:
    """Build BlockDescriptor for generic_matmul_mxfp8_api."""
    if tiles == None:
        l_tile_k, tile_m, tile_n = L_TILE_K, TILE_M, TILE_N
        lhs_load_default, rhs_load_default = LHS_LOAD_TILE, RHS_LOAD_TILE
    else:
        l_tile_k = tiles['l_tile_k']
        tile_m = tiles['tile_m']
        tile_n = tiles['tile_n']
        lhs_load_default = tiles['lhs_load_tile']
        rhs_load_default = tiles['rhs_load_tile']
    return BlockDescriptor(
        TILES_IN_BLOCK_M=tiles_in_block_m,
        TILES_IN_BLOCK_N=tiles_in_block_n,
        TILES_IN_BLOCK_K=tiles_in_block_k,
        lhs_matmul_tile_shape_logical=(l_tile_k, tile_m),
        rhs_matmul_tile_shape_logical=(l_tile_k, tile_n),
        lhs_load_tile_shape=lhs_load_tile_shape or lhs_load_default,
        rhs_load_tile_shape=rhs_load_tile_shape or rhs_load_default,
    )


def _compute_load_tile_shape(td: TensorDescriptor, tiles: dict, tile_f: int) -> tuple:
    """Compute load tile shape based on TensorDescriptor state.

    Mirrors the logic in matmul_mxfp8_generic_kernel._validate_and_calculate_shapes:
      - Pre-quantized: load tile = matmul tile physical (already x4)
      - Pre-swizzled BF16: load tile = (K_physical, F * 4)
      - Unswizzled BF16: load tile = (l_tile_k, F) logical, DGT transposes
    """
    if td.is_quantized:
        return (tiles['matmul_tile_k_physical'], tile_f)
    elif td.is_swizzled:
        return (tiles['matmul_tile_k_physical'], tile_f * INTERLEAVE_FACTOR)
    else:
        return (tiles['l_tile_k'], tile_f)


def _allocate_spill_buffer(
    num_k_blocks: int,
    num_f_blocks: int,
    block_f_logical: int,
    tiles_in_block_k: int,
    use_scale_packing: bool,
    data_buffer,
) -> TensorDescriptor:
    """Allocate and zero-initialize an HBM spill/reload buffer for one operand.

    Args:
        num_k_blocks: Number of K-blocks in the full tensor.
        num_f_blocks: Number of blocks in the F dimension (M for LHS, N for RHS).
        block_f_logical: Logical block size in F dimension (BLOCK_M_LOGICAL or BLOCK_N_LOGICAL).
        tiles_in_block_k: Number of K-tiles per block.
        use_scale_packing: Whether to use packed scale layout.
        data_buffer: HBM buffer type (nl.hbm or nl.private_hbm).

    Returns:
        TensorDescriptor wrapping the allocated data and scale buffers.
    """
    fp8_x4_dtype = get_fp8_dtype_x4("float8_e4m3fn")
    k_dim = num_k_blocks * MATMUL_TILE_K_PHYSICAL * tiles_in_block_k
    f_dim = num_f_blocks * block_f_logical
    logical_k = k_dim * INTERLEAVE_FACTOR
    scale_shape = get_scale_output_shape(logical_k, f_dim, MATMUL_TILE_K_PHYSICAL, use_scale_packing)

    return TensorDescriptor(
        data=nl.ndarray((k_dim, f_dim), dtype=fp8_x4_dtype, buffer=data_buffer),
        scales=nl.ndarray(scale_shape, dtype=nl.uint8, buffer=data_buffer),
        is_swizzled=True,
        is_x4=True,
        scales_are_packed=use_scale_packing,
    )


def _store_unswizzled_sbuf_block_to_hbm(
    output_sbuf: nl.ndarray,
    dst_hbm: nl.ndarray,
    row_base: int,
    col_base: int,
    tiles_in_block_m: int,
    block_m: int,
    block_n: int,
    lhs_matmul_tile_m: int,
    block_idx_m: int,
    m_logical: int,
    n_logical: int,
) -> None:
    """DMA copy one unswizzled SBUF accumulator block to HBM with a row offset.

    This helper is only used for storing unswizzled intermediate data (e.g. bf16
    accumulator results) from SBUF to HBM. It should NOT be used for pre-swizzled
    or pre-quantized tensors, which have different physical layouts.

    Args:
        output_sbuf (nl.ndarray): SBUF accumulator with tiled layout.
        dst_hbm (nl.ndarray): Destination HBM tensor.
        row_base (int): Row offset into dst_hbm (e.g. s_base or h_base for LNC sharding).
        col_base (int): Column offset into dst_hbm.
        tiles_in_block_m (int): Number of M-tiles in the block.
        block_m (int): Total M-dimension size of the block (tiles_in_block_m * TILE_M).
        block_n (int): Total N-dimension size of the block (tiles_in_block_n * TILE_N).
        lhs_matmul_tile_m (int): M-dimension size of a single matmul tile.
        block_idx_m (int): Block index along the M dimension.
        m_logical (int): Logical M dimension. Clamps the store height to avoid
            writing past the logical boundary.
        n_logical (int): Logical N dimension. Clamps the store width to avoid
            writing past the logical boundary.

    Returns:
        None.

    Notes:
        TODO: Reuse this helper in the generic matmul API (matmul_mxfp8_generic_api.py).
    """
    actual_n = min(block_n, n_logical - col_base)
    sbuf_step_p = tiles_in_block_m * block_n
    for tile_idx_m in range(tiles_in_block_m):
        output_idx_m = row_base + block_idx_m * block_m + tile_idx_m * lhs_matmul_tile_m
        sbuf_offset = tile_idx_m * block_n
        actual_m = min(lhs_matmul_tile_m, m_logical - (block_idx_m * block_m + tile_idx_m * lhs_matmul_tile_m))
        if actual_m > 0 and actual_n > 0:
            nisa.dma_copy(
                dst=dst_hbm[output_idx_m : output_idx_m + actual_m, col_base : col_base + actual_n],
                src=output_sbuf.ap(
                    pattern=[[sbuf_step_p, actual_m], [1, actual_n]],
                    offset=sbuf_offset,
                ),
            )
