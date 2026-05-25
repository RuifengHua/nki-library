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

"""MLP TKG kernel implementation for token generation scenarios with optional normalization"""

import nki.isa as nisa
import nki.language as nl

from ...quantization.fp8_quantize import row_quantization, static_quantization
from ...subkernels.rmsnorm_tkg import rmsnorm_tkg
from ...utils.allocator import SbufManager
from ...utils.interleave_copy import interleave_copy
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, get_nl_act_fn_from_type
from ...utils.logging import get_logger
from ...utils.tensor_view import TensorView
from ...utils.tiled_range import TiledRange
from ..mlp_parameters import (
    BS_TILE_SIZE,
    MLPParameters,
    mlpp_has_normalization,
    mlpp_has_rms_normalization,
    mlpp_store_fused_add,
)
from .down_projection_mx_shard_H import down_projection_mx_tp_shard_H
from .gate_up_projection_mx_shard_H import gate_up_projection_mx_tp_shard_H
from .mlp_tkg_constants import MLPTKGConstants
from .mlp_tkg_utils import _layout_adapter_hbm, _layout_adapter_sb
from .projection_mx_constants import ProjConfig


def _mlp_tkg_mx_impl(
    params: MLPParameters,
    output_tensor_hbm: nl.ndarray,
    output_stored_add_tensor_hbm: nl.ndarray,
    name_prefix: str = "",
) -> list[nl.ndarray]:
    """
    MLP TKG kernel with MX-backed FP8 quantization support.

    This kernel supports three quantization modes selected via
    ``QuantizationType``:

    - ``MX``: Hardware MXFP quantization using ``nisa.quantize_mx`` with
      real MX block-level scale factors (uint8).
    - ``STATIC_MX``: Software tensor-wise (static) FP8 quantization via
      ``static_quantization()``, combined with dummy MX scales set to 127.
    - ``ROW_MX``: Software row-wise (dynamic) FP8 quantization via
      ``row_quantization()``, combined with dummy MX scales set to 127.

    For STATIC_MX and ROW_MX the ``nisa.quantize_mx`` instruction is *not*
    used.  Instead, separate dummy MX scale tensors filled with 127 are
    allocated and passed to the MX matmul so the hardware MX dequant is
    a no-op.  When ``quant_params.mx_dummy_scale_hbm`` is provided (a
    pre-filled HBM tensor of all 127), the scales are DMA'd from HBM
    instead of memset, avoiding per-layer memset overhead.

    Activation quantization
    -----------------------
    STATIC_MX (tensor-wise / static):
        Single pre-computed scale from the checkpoint::

            quantized_input = clip(hidden / input_dequant_scale, -MAXVAL, MAXVAL)  # fp8[BxS, H]

        ``input_dequant_scale`` shape: ``[_pmax, 1]`` (per-tensor scalar).

    ROW_MX (row-wise / dynamic):
        Per-token scale computed at runtime::

            absmax = max(abs(hidden), dim=-1)          # bf16[BxS]
            dequant_scale = absmax / MAXVAL            # bf16[BxS]
            quant_scale   = 1 / dequant_scale          # bf16[BxS]
            quantized_input = hidden * quant_scale     # fp8[BxS, H]

        ``input_dequant_scale`` shape: ``[_pmax, BxS, 1]`` (rank-3, per-token).

    Weight dequantization scale shapes
    -----------------------------------
    Weights are stored in FP8 (fp8_e4m3fn_x4). Dequant scales are provided
    offline and broadcast across the partition dimension (_pmax = 128):

    Row-wise weight quantization (one scale per row)::

        gate_w_dequant_scale: [1, I] → broadcast → [128, I]
        up_w_dequant_scale:   [1, I] → broadcast → [128, I]
        down_w_dequant_scale: [1, H] → broadcast → [128, H]

    Tensor-wise (static) weight quantization (single scalar per tensor)::

        gate_w_dequant_scale: [1, 1] → broadcast → [128, 1]
        up_w_dequant_scale:   [1, 1] → broadcast → [128, 1]
        down_w_dequant_scale: [1, 1] → broadcast → [128, 1]

    Dimensions:
        H: Hidden dimension size (must be divisible by 512)
        I: Intermediate dimension size
        T: Sequence length (BxS, padded to multiple of 4)
        B: Batch size
        S: Sequence length per batch

    Args:
        params (MLPParameters): MLP configuration with FP8 quantized weights.
            - gate_proj_weights_tensor: [_pmax, n_H512_tile, I] in fp8_x4
            - up_proj_weights_tensor: [_pmax, n_H512_tile, I] in fp8_x4
            - down_proj_weights_tensor: [I_p, ceil(I/512), H] in fp8_x4
              I-contiguous x4 packing: element [p, tile, h] packs
              W[512*tile + 4p + q, h] for q=0..3 (4 consecutive I values
              at the same H column). Shared layout with CTE MX down projection.
            - quant_params.gate_w_scale: dequant scale (shape depends on
              row-wise vs tensor-wise, see above)
            - quant_params.up_w_scale: dequant scale
            - quant_params.down_w_scale: dequant scale
            - quant_params.gate_up_in_scale: activation dequant scale
              (STATIC_MX only)
            - quant_params.down_in_scale: intermediate dequant scale
              (STATIC_MX only)
        output_tensor_hbm (nl.ndarray): [B, S, H], Output tensor in HBM
        output_stored_add_tensor_hbm (nl.ndarray): Optional fused add output in HBM

    Returns:
        list[nl.ndarray]:
            - [output_tensor_hbm] when store_output_in_sbuf=False
            - [down_out_sb] when store_output_in_sbuf=True

    Notes:
        - Supports RMSNorm but not LayerNorm
        - Does not support fused_add or column tiling
        - H must be divisible by 512 for proper quantization alignment
        - T is padded to multiple of 4 for quantization requirements

    Pseudocode (STATIC_MX flow):
        # Optional normalization
        if has_normalization:
            hidden = rmsnorm(hidden)

        # ── Quantize input activations ──
        if quantization_type == STATIC_MX:
            quantized_input, input_dequant_scale = static_quantization(hidden)
        elif quantization_type == ROW_MX:
            quantized_input, input_dequant_scale = row_quantization(hidden)

        # Reinterpret fp8 → fp8_x4 (consecutive-4 H1 packing, zero-cost view)
        # then permute on x4 data (4× fewer elements than fp8 permute)
        quantized_x4 = reinterpret_cast(quantized_input.reshape(..., n_H512, 4), fp8_x4)
        inp_qtz = permute(quantized_x4, [0, 2, 1])  # [H0, n_H512, T_padded]

        # ── Gate projection (uses dummy MX scales) ──
        quantized_gate_out = matmul_mx(inp_qtz, gate_w, dummy_scale, dummy_scale)
        # STATIC_MX: gate_out = quantized_gate_out * combined_dequant_scale
        # ROW_MX:    gate_out = quantized_gate_out * gate_w_dequant_scale * input_dequant_scale
        if gate_bias:
            gate_out += gate_bias
        gate_out = activation(gate_out)

        # ── Up projection (uses dummy MX scales) ──
        quantized_up_out = matmul_mx(quantized_input, up_w, dummy_scale, dummy_scale)
        # STATIC_MX: up_out = quantized_up_out * combined_dequant_scale
        # ROW_MX:    up_out = quantized_up_out * up_w_dequant_scale * input_dequant_scale
        if up_bias:
            up_out += up_bias

        # ── Element-wise multiply ──
        intermediate = gate_out * up_out

        # ── Quantize intermediate ──
        if quantization_type == STATIC_MX:
            quantized_inter, inter_dequant_scale = static_quantization(intermediate)
        elif quantization_type == ROW_MX:
            quantized_inter, inter_dequant_scale = row_quantization(intermediate)

        # ── Down projection (uses dummy MX scales) ──
        quantized_down_out = matmul_mx(quantized_inter, down_w, dummy_scale, dummy_scale)
        # STATIC_MX: down_out = quantized_down_out * combined_dequant_scale
        # ROW_MX:    down_out = quantized_down_out * down_w_dequant_scale * inter_dequant_scale
        if down_bias:
            down_out += down_bias

        # Transpose and store
        output = transpose(down_out)  # [H0, H1, T] → [T, H]
    """
    io_dtype = params.hidden_tensor.dtype

    # Validate inputs
    kernel_assert(
        params.quant_params.is_quant_mx()
        or params.quant_params.is_quant_static_mx()
        or params.quant_params.is_quant_row_mx(),
        "mlp_tkg_mx requires MX, STATIC_MX, or ROW_MX quantization",
    )
    kernel_assert(
        not mlpp_has_normalization(params) or mlpp_has_rms_normalization(params),
        "mlp_tkg_mx only supports RMSNorm or no normalization",
    )

    # Compute kernel dimensions
    dims = MLPTKGConstants.calculate_constants(params)

    # Get MX quantization constants from dims
    _pmax = dims._pmax  # Partition dimension (128)
    _q_width = dims._q_width  # Quantization tile width (4)
    _q_height = dims._q_height  # Quantization tile height (8)

    # Flag for software quantization path (STATIC_MX/ROW_MX use our own quantization, not nisa.quantize_mx)
    is_software_quant = params.quant_params.is_quant_static_mx() or params.quant_params.is_quant_row_mx()
    is_row_mx = params.quant_params.is_quant_row_mx()

    # ============================================================
    # Section 1: Normalization (Optional)
    # ============================================================
    hidden_tensor = params.hidden_tensor
    if mlpp_has_normalization(params):
        if mlpp_has_rms_normalization(params):
            rmsnorm_out = nl.ndarray((dims.H0, dims.T, dims.H1), dtype=io_dtype, buffer=nl.sbuf)
            norm_weights = params.norm_params.normalization_weights_tensor
            eps = params.eps
            rmsnorm_sbm = SbufManager(
                sb_lower_bound=0,
                sb_upper_bound=200 * 1024,
                logger=get_logger("mlp_tkg_mx_rmsnorm"),
                use_auto_alloc=True,
            )
            rmsnorm_sbm.set_name_prefix(name_prefix)
            rmsnorm_out = rmsnorm_tkg(
                input=params.hidden_tensor,
                gamma=norm_weights,
                output=rmsnorm_out,
                eps=eps,
                hidden_dim_tp=True,
                single_core_forced=True,
                sbm=rmsnorm_sbm,
            )
            hidden_tensor = rmsnorm_out
        else:
            kernel_assert(False, "mlp_tkg_mx only supports RMSNorm, LayerNorm is not supported")

    # ============================================================
    # Section 2: Input Quantization and x4 Packing
    # ============================================================
    """
    Quantize input activations to fp8 and pack to fp8_x4 for nc_matmul_mx.
    Calculate tiling dimensions for H and I.

    For STATIC_MX/ROW_MX: software quantization → reinterpret_cast(fp8_x4) → x4 permute.
        Weights are pre-shuffled to match the consecutive-4 H1 packing.
    For MX: _layout_adapter → hardware quantization via nisa.quantize_mx, real MX block scales.
    """
    n_H512_tile_sharded = dims.H_per_shard // (_pmax * _q_width)  # Number of 512-element tiles in H dimension
    n_I512_tile = div_ceil(dims.I, (_pmax * _q_width))  # Number of 512-element tiles in I dimension
    T_padded = div_ceil(dims.T, 4) * 4  # Pad T to multiple of 4 for quantization

    if is_software_quant:
        # ── Software quantization path (STATIC_MX / ROW_MX) ──
        #
        # STATIC_MX flow:
        #   quantized_input, input_dequant_scale = static_quantization(input)
        #   input_dequant_scale is per-tensor [_pmax, 1]
        #
        # ROW_MX flow:
        #   quantized_input, input_dequant_scale = row_quantization(input)
        #   input_dequant_scale is per-token [_pmax, BxS, 1] (rank-3 path)
        #
        # Both paths then: reinterpret_cast(fp8_x4) → x4 permute → gate/up with dummy MX scales → post-matmul dequant
        # Weights are pre-shuffled offline to match the consecutive-4 H1 packing (see _fp8_to_gate_up_x4).

        # ── Dummy MX scales for STATIC_MX/ROW_MX ──
        # All dummy scales are uniform 127 so the hardware MX dequant is a no-op.
        # Optimization: memset as uint32 (value=0x7F7F7F7F = 2139062143) then
        # reinterpret to uint8.  DVE processes one element per partition per cycle,
        # so uint32 gives 4× throughput vs uint8 memset.
        # Scales sharing the same free-dim size share one buffer (tile-dim slicing
        # is contiguous; free-dim slicing is not, so separate buffers per free-dim).
        H_sharded = dims.H // dims.num_shards
        max_tiles = max(n_H512_tile_sharded, n_I512_tile)

        # Buffer for T_padded free-dim (shared by inp_scale and inter_scale_dummy)
        kernel_assert(T_padded % 4 == 0, f"T_padded must be divisible by 4 for uint32 memset, got {T_padded}")
        dummy_scale_T_u32 = nl.ndarray((_pmax, max_tiles, T_padded // 4), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.memset(dst=dummy_scale_T_u32, value=2139062143)
        dummy_scale_T = TensorView(dummy_scale_T_u32).reinterpret_cast(nl.uint8).get_view()
        inp_scale = dummy_scale_T[:, :n_H512_tile_sharded, :]
        inter_scale_dummy = dummy_scale_T[:, :n_I512_tile, :]

        # Buffer for I free-dim (gate/up weight scale)
        kernel_assert(dims.I % 4 == 0, f"I must be divisible by 4 for uint32 memset, got {dims.I}")
        dummy_scale_I_u32 = nl.ndarray((_pmax, n_H512_tile_sharded, dims.I // 4), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.memset(dst=dummy_scale_I_u32, value=2139062143)
        mx_gate_up_w_scale_dummy = TensorView(dummy_scale_I_u32).reinterpret_cast(nl.uint8).get_view()

        # Buffer for H_sharded free-dim (down weight scale)
        kernel_assert(H_sharded % 4 == 0, f"H_sharded must be divisible by 4 for uint32 memset, got {H_sharded}")
        dummy_scale_H_u32 = nl.ndarray((_pmax, n_I512_tile, H_sharded // 4), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.memset(dst=dummy_scale_H_u32, value=2139062143)
        down_w_scale_dummy = TensorView(dummy_scale_H_u32).reinterpret_cast(nl.uint8).get_view()

        if not is_row_mx:
            # ── STATIC_MX: Load scales and pre-compute quant_scale = 1/dequant_scale ──
            gate_up_in_scale = nl.ndarray((_pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=gate_up_in_scale, src=params.quant_params.gate_up_in_scale)
            quant_scale = nl.ndarray((_pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=quant_scale, data=gate_up_in_scale)
            # Pre-load down_in_scale early to overlap DMA with gate/up computation
            down_in_scale = nl.ndarray((_pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=down_in_scale, src=params.quant_params.down_in_scale)

        # inp_qtz is set directly by all paths below.
        inp_qtz = None

        if params.input_in_sbuf or mlpp_has_rms_normalization(params):
            # ── SBUF path: hidden_tensor is [H0, T, H1] in SBUF ──
            # Use natural consecutive-4 H1 packing: reshape [H0, T, n_H512, 4] → reinterpret fp8_x4
            # → permute to [H0, n_H512_sharded, T_padded]. Weights are pre-shuffled to match.
            if is_row_mx:
                quantized_input, input_dequant_scale_raw = row_quantization(
                    hidden_tensor,
                    output_dtype=nl.float8_e4m3fn,
                )
                # Pad dequant_scale from [_pmax, T, 1] to [_pmax, T_padded, 1]
                if T_padded > dims.T:
                    input_dequant_scale = nl.ndarray((_pmax, T_padded, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.memset(dst=input_dequant_scale, value=0.0)
                    nisa.tensor_copy(
                        dst=input_dequant_scale[:, : dims.T, :],
                        src=input_dequant_scale_raw,
                    )
                else:
                    input_dequant_scale = input_dequant_scale_raw
            else:
                quantized_input, input_dequant_scale = static_quantization(
                    hidden_tensor,
                    gate_up_in_scale,
                    quant_scale=quant_scale,
                )

            # quantized_input is fp8 [H0, T, H1] — consecutive H1 layout.
            # Reinterpret fp8 → fp8_x4 on the fresh quantized ndarray (partition offset 0),
            # then slice for shard and permute on x4 data (4× fewer elements to move).
            n_H512_total = n_H512_tile_sharded * dims.num_shards
            qtz_4d = quantized_input.reshape((_pmax, dims.T, n_H512_total, _q_width))
            qtz_x4_4d = TensorView(qtz_4d).reinterpret_cast(nl.float8_e4m3fn_x4)
            qtz_x4 = qtz_x4_4d.reshape((_pmax, dims.T, n_H512_total))

            # Slice for shard, then permute — all on x4 (4× fewer free-dim elements)
            src_perm_x4 = (
                TensorView(qtz_x4)
                .slice(dim=2, start=dims.shard_id * n_H512_tile_sharded, end=(dims.shard_id + 1) * n_H512_tile_sharded)
                .permute(dims=[0, 2, 1])  # [H0, n_H512_sharded, T]
            )

            # Materialize the x4 permute into a contiguous buffer
            inp_qtz_sb = nl.ndarray((_pmax, n_H512_tile_sharded, T_padded), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
            nisa.memset(dst=inp_qtz_sb, value=0)
            nisa.tensor_copy(
                dst=inp_qtz_sb[:, :, : dims.T],
                src=src_perm_x4.get_view(),
            )
            inp_qtz = TensorView(inp_qtz_sb)
        else:
            # ── HBM path: strided DMA load → quantize → consecutive-4 reinterpret_cast ──
            # Load hidden from HBM [T, H] to SBUF [_pmax, T, H1_shard] in natural layout,
            # then quantize and use the same consecutive-4 packing as the SBUF path.
            H1_shard = dims.H_per_shard // _pmax
            hidden_tensor = hidden_tensor.reshape((dims.T, dims.H))
            input_view = (
                TensorView(hidden_tensor)
                .reshape_dim(dim=1, shape=[dims.num_shards, H1_shard, _pmax])
                .permute(dims=[3, 0, 1, 2])
                .select(dim=2, index=dims.shard_id)
            )
            input_sb = nl.ndarray((_pmax, dims.T, H1_shard), dtype=io_dtype, buffer=nl.sbuf)
            nisa.dma_copy(src=input_view.get_view(), dst=input_sb)

            if is_row_mx:
                quantized_input, input_dequant_scale_raw = row_quantization(
                    input_sb,
                    output_dtype=nl.float8_e4m3fn,
                )
                if T_padded > dims.T:
                    input_dequant_scale = nl.ndarray((_pmax, T_padded, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.memset(dst=input_dequant_scale, value=0.0)
                    nisa.tensor_copy(
                        dst=input_dequant_scale[:, : dims.T, :],
                        src=input_dequant_scale_raw,
                    )
                else:
                    input_dequant_scale = input_dequant_scale_raw
            else:
                quantized_input, input_dequant_scale = static_quantization(
                    input_sb,
                    gate_up_in_scale,
                    quant_scale=quant_scale,
                )

            # quantized_input is fp8 [H0, T, H1_shard] — reinterpret to x4 first, then permute (4× fewer elements).
            qtz_4d = quantized_input.reshape((_pmax, dims.T, n_H512_tile_sharded, _q_width))
            qtz_x4_4d = TensorView(qtz_4d).reinterpret_cast(nl.float8_e4m3fn_x4)
            qtz_x4 = qtz_x4_4d.reshape((_pmax, dims.T, n_H512_tile_sharded))

            src_perm_x4 = TensorView(qtz_x4).permute(dims=[0, 2, 1])  # [H0, n_H512_sharded, T]

            inp_qtz_sb = nl.ndarray((_pmax, n_H512_tile_sharded, T_padded), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
            nisa.memset(dst=inp_qtz_sb, value=0)
            nisa.tensor_copy(
                dst=inp_qtz_sb[:, :, : dims.T],
                src=src_perm_x4.get_view(),
            )
            inp_qtz = TensorView(inp_qtz_sb)

    else:
        # ── MX path: existing hardware quantization (unchanged) ──
        input_sb_shfl = None

        if params.input_in_sbuf or mlpp_has_rms_normalization(params):
            input_sb_shfl = _layout_adapter_sb(hidden_tensor, n_prgs=dims.num_shards, prg_id=dims.shard_id)
        else:
            hidden_tensor = hidden_tensor.reshape((dims.T, dims.H))
            input_sb_shfl = _layout_adapter_hbm(hidden_tensor, n_prgs=dims.num_shards, prg_id=dims.shard_id)

        # Allocate quantized tensors for mxfp8 format
        inp_qtz = nl.ndarray(
            (_pmax, n_H512_tile_sharded * T_padded),
            dtype=nl.float8_e4m3fn_x4,
            buffer=nl.sbuf,
            name=f"{name_prefix}input_quantized",
        )
        inp_scale = nl.ndarray(
            inp_qtz.shape,
            dtype=nl.uint8,
            buffer=nl.sbuf,
            name=f"{name_prefix}input_scale",
        )

        # Quantize input from bf16 to mxfp8
        input_flat = input_sb_shfl.reshape((_pmax, n_H512_tile_sharded * T_padded * _q_width))
        nisa.quantize_mx(dst=inp_qtz, src=input_flat, dst_scale=inp_scale)

        # Reshape to tiled format for matmul operations
        inp_qtz = inp_qtz.reshape((_pmax, n_H512_tile_sharded, T_padded))
        inp_scale = inp_scale.reshape(inp_qtz.shape)

    # ---------------- Create ProjConfig ----------------
    # Configuration object for projection operations with H-dimension sharding
    proj_cfg = ProjConfig(
        H=dims.H,
        I=dims.I,
        BxS=T_padded,
        n_prgs=dims.num_shards,
        prg_id=dims.shard_id,
        name_prefix=name_prefix,
    )

    # ============================================================
    # Section 3: Gate Projection with MXFP
    # ============================================================
    # Compute: gate_out = hidden @ gate_weight + gate_bias
    # Allocate and load gate bias in SBUF if present
    gate_bias_sb = None
    if params.bias_params.gate_proj_bias_tensor is not None:
        # Gate bias format in HBM: bf16[I_p, ceil(I/512), 4] where I_p = I//4 if I <= 512 else _pmax
        # We need to load and reshape to [_pmax, n_I512_tile, _q_width] for projection
        gate_bias_sb = nl.ndarray((_pmax, n_I512_tile, _q_width), dtype=nl.bfloat16, buffer=nl.sbuf)

        if dims.I < 512:
            # When I<512, bias HBM is not padded, so pad it in SBUF
            nisa.memset(dst=gate_bias_sb[:, 0, :], value=0.0)
            nisa.dma_copy(
                dst=gate_bias_sb[: dims.I // 4, :, :],
                src=params.bias_params.gate_proj_bias_tensor,
            )
        else:
            # I >= 512, bias is already padded in HBM
            nisa.dma_copy(
                dst=gate_bias_sb,
                src=params.bias_params.gate_proj_bias_tensor,
            )

    # Perform gate projection using MXFP quantized weights
    if is_software_quant:
        gate_w_dequant_sb = nl.ndarray(params.quant_params.gate_w_scale.shape, dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gate_w_dequant_sb, src=params.quant_params.gate_w_scale)

        # ── STATIC_MX and ROW_MX: unified call with w_dequant_scale + input_dequant_scale ──
        gate_out_sb = gate_up_projection_mx_tp_shard_H(
            hidden_qtz_sb=inp_qtz if isinstance(inp_qtz, TensorView) else TensorView(inp_qtz),
            hidden_scale_sb=TensorView(inp_scale),
            weight_qtz=TensorView(params.gate_proj_weights_tensor),
            weight_scale=TensorView(mx_gate_up_w_scale_dummy),
            bias_sb=TensorView(gate_bias_sb) if gate_bias_sb is not None else None,
            cfg=proj_cfg,
            w_dequant_scale=gate_w_dequant_sb,
            input_dequant_scale=input_dequant_scale,
        )

        # Apply activation function to gate output
        nisa.activation(
            dst=gate_out_sb,
            op=get_nl_act_fn_from_type(params.activation_fn),
            data=gate_out_sb,
        )
    else:
        # ── MX path: real MX scales, bias inside projection ──
        gate_out_sb = gate_up_projection_mx_tp_shard_H(
            hidden_qtz_sb=inp_qtz if isinstance(inp_qtz, TensorView) else TensorView(inp_qtz),
            hidden_scale_sb=TensorView(inp_scale),
            weight_qtz=TensorView(params.gate_proj_weights_tensor),
            weight_scale=TensorView(params.quant_params.gate_w_scale),
            bias_sb=TensorView(gate_bias_sb) if gate_bias_sb is not None else None,
            cfg=proj_cfg,
        )

        # MX path: activation only (no dequant needed, real MX scales used in matmul)
        nisa.activation(
            dst=gate_out_sb,
            op=get_nl_act_fn_from_type(params.activation_fn),
            data=gate_out_sb,
        )

    # ============================================================
    # Section 4: Up Projection with MXFP
    # ============================================================
    # Compute: up_out = hidden @ up_weight + up_bias
    # Allocate and load up bias in SBUF if present
    up_bias_sb = None
    if params.bias_params.up_proj_bias_tensor is not None:
        # Up bias format in HBM: bf16[I_p, ceil(I/512), 4] where I_p = I//4 if I <= 512 else _pmax
        # We need to load and reshape to [_pmax, n_I512_tile, _q_width] for projection
        up_bias_sb = nl.ndarray((_pmax, n_I512_tile, _q_width), dtype=nl.bfloat16, buffer=nl.sbuf)

        if dims.I < 512:
            # When I<512, bias HBM is not padded, so pad it in SBUF
            nisa.memset(dst=up_bias_sb[:, 0, :], value=0.0)
            nisa.dma_copy(
                dst=up_bias_sb[: dims.I // 4, :, :],
                src=params.bias_params.up_proj_bias_tensor,
            )
        else:
            # I >= 512, bias is already padded in HBM
            nisa.dma_copy(
                dst=up_bias_sb,
                src=params.bias_params.up_proj_bias_tensor,
            )

    # Perform up projection using MXFP quantized weights
    if is_software_quant:
        up_w_dequant_sb = nl.ndarray(params.quant_params.up_w_scale.shape, dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=up_w_dequant_sb, src=params.quant_params.up_w_scale)

        # ── STATIC_MX and ROW_MX: unified call with w_dequant_scale + input_dequant_scale ──
        up_out_sb = gate_up_projection_mx_tp_shard_H(
            hidden_qtz_sb=inp_qtz if isinstance(inp_qtz, TensorView) else TensorView(inp_qtz),
            hidden_scale_sb=TensorView(inp_scale),
            weight_qtz=TensorView(params.up_proj_weights_tensor),
            weight_scale=TensorView(mx_gate_up_w_scale_dummy),
            bias_sb=TensorView(up_bias_sb) if up_bias_sb is not None else None,
            cfg=proj_cfg,
            w_dequant_scale=up_w_dequant_sb,
            input_dequant_scale=input_dequant_scale,
        )
    else:
        # ── MX path: real MX scales, bias inside projection ──
        up_out_sb = gate_up_projection_mx_tp_shard_H(
            hidden_qtz_sb=inp_qtz if isinstance(inp_qtz, TensorView) else TensorView(inp_qtz),
            hidden_scale_sb=TensorView(inp_scale),
            weight_qtz=TensorView(params.up_proj_weights_tensor),
            weight_scale=TensorView(params.quant_params.up_w_scale),
            bias_sb=TensorView(up_bias_sb) if up_bias_sb is not None else None,
            cfg=proj_cfg,
        )

    # ============================================================
    # Section 5: Element-wise Multiply
    # ============================================================
    # Compute: intermediate = activation(gate_out) * up_out
    # Reuse gate_out_sb buffer for intermediate result
    intermediate_sb = gate_out_sb
    nisa.tensor_tensor(
        dst=intermediate_sb,
        data1=gate_out_sb,
        data2=up_out_sb,
        op=nl.multiply,
    )

    # ============================================================
    # Section 6: Down Projection with MXFP
    # ============================================================
    # Compute: output = intermediate @ down_weight + down_bias
    down_weights = params.down_proj_weights_tensor
    down_scale = params.quant_params.down_w_scale
    down_bias = params.bias_params.down_proj_bias_tensor

    # Allocate and load down projection bias in SBUF if present
    down_bias_sb = None
    if down_bias is not None:
        # Reshape to separate shards: [1, H] -> [dims.num_shards, H1_shard, H0]
        # This works because H = dims.num_shards * H1_shard * H0
        down_bias = down_bias.reshape((dims.num_shards, dims.H1_shard, dims.H0))

        # Select this shard's bias portion using TensorView
        sharded_down_bias_hbm_view = TensorView(down_bias).select(dim=0, index=dims.shard_id)

        # Allocate SBUF buffer for bias with correct layout [H0, H1_shard]
        down_bias_sb = nl.ndarray((dims.H0, dims.H1_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down_bias_sb_view = TensorView(down_bias_sb)

        # dma transpose requirement : 4D AP
        while sharded_down_bias_hbm_view.get_dim() < 4:
            sharded_down_bias_hbm_view = sharded_down_bias_hbm_view.expand_dim(1)
        while down_bias_sb_view.get_dim() < 4:
            down_bias_sb_view = down_bias_sb_view.expand_dim(1)

        # DMA copy the selected shard's bias to SBUF
        # Result is [H0, H1_shard] ready for use in down projection
        nisa.dma_transpose(
            dst=down_bias_sb_view.get_view(),
            src=sharded_down_bias_hbm_view.get_view(),
        )

    # Perform down projection using MXFP quantized weights
    # partial_output=True skips LNC sync when output stays in SBUF (for debugging/inspection)
    if is_software_quant:
        # ── Software quant: quantize intermediate + reinterpret_cast to fp8_x4 ──
        if is_row_mx:
            # ── ROW_MX: row_quantization on intermediate ──
            # intermediate_sb is [_pmax, n_I512_tile, T_padded, _q_width]
            # row_quantization rank-3 expects [P0, BxS, F0] where BxS is the token dim.
            inter_4d = intermediate_sb.reshape((_pmax, n_I512_tile, T_padded, _q_width))

            # Permute [_pmax, n_I512, T_padded, _q_width] → [_pmax, T_padded, n_I512, _q_width]
            inter_permuted = nl.ndarray(
                (_pmax, T_padded, n_I512_tile, _q_width),
                dtype=inter_4d.dtype,
                buffer=nl.sbuf,
            )
            src_perm = TensorView(inter_4d).permute(dims=[0, 2, 1, 3])
            nisa.tensor_copy(dst=inter_permuted, src=src_perm.get_view())

            # Reshape to rank-3 [_pmax, T_padded, n_I512*_q_width] for row_quantization
            inter_3d = inter_permuted.reshape((_pmax, T_padded, n_I512_tile * _q_width))
            quantized_3d, inter_dequant_scale = row_quantization(
                inter_3d,
                output_dtype=nl.float8_e4m3fn,
            )

            # row_quantization with output_dtype returns fp8 directly.
            # Reshape to [_pmax, T_padded, n_I512, _q_width] fp8, reinterpret_cast to fp8_x4
            # BEFORE permuting — this makes the subsequent permute 4× smaller.
            quantized_4d = quantized_3d.reshape((_pmax, T_padded, n_I512_tile, _q_width))
            quantized_x4 = TensorView(quantized_4d).reinterpret_cast(nl.float8_e4m3fn_x4)

            # Permute [_pmax, T_padded, n_I512] → [_pmax, n_I512, T_padded] fp8_x4
            inter_qtz = nl.ndarray(
                (_pmax, n_I512_tile, T_padded),
                dtype=nl.float8_e4m3fn_x4,
                buffer=nl.sbuf,
            )
            src_perm_back = TensorView(quantized_x4.reshape((_pmax, T_padded, n_I512_tile))).permute(dims=[0, 2, 1])
            nisa.tensor_copy(dst=inter_qtz, src=src_perm_back.get_view())
        else:
            # ── STATIC_MX: static_quantization on intermediate ──
            inter_flat = intermediate_sb.reshape((_pmax, n_I512_tile * T_padded * _q_width))
            quantized_inter, inter_dequant_scale = static_quantization(
                inter_flat,
                down_in_scale,
            )
            inter_4d = quantized_inter.reshape((_pmax, n_I512_tile, T_padded, _q_width))

            inter_tv = TensorView(inter_4d)
            inter_qtz = inter_tv.reinterpret_cast(nl.float8_e4m3fn_x4)
            inter_qtz = inter_qtz.reshape((_pmax, n_I512_tile, T_padded))

        # ── STATIC_MX and ROW_MX: unified down projection call ──
        down_w_dequant_sb = nl.ndarray(down_scale.shape, dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=down_w_dequant_sb, src=down_scale)

        down_out_sb = down_projection_mx_tp_shard_H(
            inter_sb=inter_qtz,
            weight=down_weights,
            weight_scale=down_w_scale_dummy,
            bias_sb=down_bias_sb,
            cfg=proj_cfg,
            partial_output=not params.store_output_in_sbuf,
            pre_quantized=True,
            pre_quantized_scale=inter_scale_dummy,
            w_dequant_scale=down_w_dequant_sb,
            input_dequant_scale=inter_dequant_scale,
        )

    else:
        # ── MX path: existing down projection (unchanged) ──
        down_out_sb = down_projection_mx_tp_shard_H(
            inter_sb=intermediate_sb,
            weight=down_weights,
            weight_scale=down_scale,
            bias_sb=down_bias_sb,
            cfg=proj_cfg,
            partial_output=not params.store_output_in_sbuf,
        )

    # ============================================================
    # Section 7: Output Transpose and Storage
    # ============================================================
    if not params.store_output_in_sbuf:
        # Store output to HBM with transpose from [H0, H1, T] to [T, H]

        # Reshape to 2D tensor for easier indexing
        B, S, H = output_tensor_hbm.shape
        output_tensor_hbm = TensorView(output_tensor_hbm).flatten_dims(start_dim=0, end_dim=1)  # [T, H]

        # Create view for this shard's portion of H dimension
        # [T, H_shard]
        output_hbm_view = output_tensor_hbm.slice(
            dim=1, start=dims.shard_id * dims.H_per_shard, end=(dims.shard_id + 1) * dims.H_per_shard
        )

        # Transpose output from [H0, H1, T] to [T, H] layout
        down_out_view = TensorView(down_out_sb).slice(dim=2, start=0, end=dims.T)
        output_sb = nl.ndarray(
            (dims.T, dims.H_per_shard),
            dtype=output_tensor_hbm.dtype,
            buffer=nl.sbuf,
            name=f"{name_prefix}tkg_mlp_output_sb",
        )
        output_sb_view = TensorView(output_sb)

        # Transpose each H1 tile using nc_transpose and interleave_copy
        for h1_tile_idx in range(dims.H1_shard):
            psum_idx = h1_tile_idx % dims._psum_bmax
            tp_psum = nl.ndarray(
                (dims.T, dims.H0),
                dtype=output_tensor_hbm.dtype,
                buffer=nl.psum,
                name=f"{name_prefix}transpose_output_{h1_tile_idx}",
            )
            # Transpose [H0, T] to [T, H0]
            nisa.nc_transpose(dst=tp_psum, data=down_out_view.select(dim=1, index=h1_tile_idx).get_view())
            # Copy transposed tile to output buffer with interleaving
            interleave_copy(
                dst=output_sb_view.slice(
                    dim=1, start=h1_tile_idx * dims.H0, end=(h1_tile_idx + 1) * dims.H0
                ).get_view(),
                src=tp_psum,
                index=h1_tile_idx,
            )

        # DMA copy transposed output to HBM
        nisa.dma_copy(
            dst=output_hbm_view.get_view(),
            src=output_sb_view.get_view(),
        )

        # Reshape back to 3D tensor
        output_tensor_hbm = output_tensor_hbm.base_tensor.reshape((B, S, H))

        return (
            [output_tensor_hbm, output_stored_add_tensor_hbm] if mlpp_store_fused_add(params) else [output_tensor_hbm]
        )

    else:
        # Keep output in SBUF (for debugging or when caller will handle HBM storage)
        return [down_out_sb, output_stored_add_tensor_hbm] if mlpp_store_fused_add(params) else [down_out_sb]


def mlp_tkg_mx(
    params: MLPParameters,
    output_tensor_hbm: nl.ndarray,
    output_stored_add_tensor_hbm: nl.ndarray,
) -> list[nl.ndarray]:
    """Wrapper that tiles along BxS and calls _mlp_tkg_mx_impl per tile."""

    T = params.batch_size * params.sequence_len
    H = params.hidden_size
    tile_size = min(BS_TILE_SIZE, T)

    # Short-circuit: if T fits in one tile, just call impl directly
    if T <= tile_size:
        return _mlp_tkg_mx_impl(params, output_tensor_hbm, output_stored_add_tensor_hbm)

    kernel_assert(
        not params.store_output_in_sbuf,
        "mlp_tkg_mx tiling does not support store_output_in_sbuf with BxS > BS_TILE_SIZE",
    )
    kernel_assert(
        not params.input_in_sbuf,
        "mlp_tkg_mx tiling does not support input_in_sbuf with BxS > BS_TILE_SIZE",
    )

    # Flatten hidden to 2D (T, H) for contiguous slicing
    hidden = params.hidden_tensor.reshape((T, H))

    B, S, H_out = output_tensor_hbm.shape
    output_hbm_2d = output_tensor_hbm.reshape((B * S, H_out))
    output_hbm_view = TensorView(output_hbm_2d)  # [T, H]

    for bxs_tile in TiledRange(T, tile_size):
        params.batch_size = 1
        params.sequence_len = bxs_tile.size

        # Slice hidden input for this tile: (tile_size, H) -> (1, tile_size, H)
        params.hidden_tensor = hidden[bxs_tile.start_offset : bxs_tile.end_offset, :].reshape((1, bxs_tile.size, H))

        # [T, H] -> [T_tile, H]
        output_tile = (
            output_hbm_view.slice(dim=0, start=bxs_tile.start_offset, end=bxs_tile.end_offset)
            .expand_dim(dim=0)
            .get_view()
        )

        _mlp_tkg_mx_impl(
            params,
            output_tile,
            output_stored_add_tensor_hbm,
            name_prefix=f"bxs_{bxs_tile.index}_",
        )

    return [output_tensor_hbm, output_stored_add_tensor_hbm] if mlpp_store_fused_add(params) else [output_tensor_hbm]
