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

"""State Space Duality (SSD) kernel for NKI.

Implements chunk-wise parallel Mamba-2 computation using TensorE matmuls
for intra-chunk structured attention and VectorE scans for decay computation.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil, get_program_sharding_info


@nki.jit
def ssd(
    x: nl.ndarray,
    dt: nl.ndarray,
    A: nl.ndarray,
    B: nl.ndarray,
    C: nl.ndarray,
    chunk_size: int = 128,
    D: nl.ndarray = None,
    initial_state: nl.ndarray = None,
    causal_mask: nl.ndarray = None,
) -> tuple:
    """
    State Space Duality (SSD) scan for Mamba-2 models.

    Performs chunk-wise parallel computation combining TensorE matmuls
    (intra-chunk structured attention) with VectorE cumulative sums
    (decay computation). For each chunk of size Q:

    1. Cumulative decay: cs = cumsum(dt * A)
    2. Intra-chunk: Y_intra = exp(cs) * ((CB * causal) @ (exp(-cs) * dt * x))
       where CB = C @ B^T is the structured attention matrix
    3. State-to-output: Y_off = exp(cs) * (C @ state)
    4. State update: state = exp(cs[-1]) * state + B^T @ (dt * x * exp(cs[-1] - cs))
    5. Output: y = Y_intra + Y_off [+ D * x]

    Dimensions:
        batch: Batch size
        nheads: Number of attention heads
        seqlen: Sequence length (must be divisible by chunk_size)
        headdim: Head dimension (<= 512 for gen2/3 PSUM free dim limit)
        dstate: SSM state dimension (<= 128 for nc_transpose and matmul)
        Q: Chunk size (<= 128, must fit in partition dimension)

    Args:
        x (nl.ndarray): [batch, nheads, seqlen, headdim], Input tensor.
        dt (nl.ndarray): [batch, nheads, seqlen], Timestep tensor. Should be positive.
        A (nl.ndarray): [nheads], State transition scalar per head. Should be negative.
        B (nl.ndarray): [batch, seqlen, dstate], Input projection. Shared across heads.
        C (nl.ndarray): [batch, seqlen, dstate], Output projection. Shared across heads.
        chunk_size (int): Chunk size Q. Must be <= 128 (compile-time constant).
        D (nl.ndarray, optional): [nheads], Skip connection weights. Default: None.
        initial_state (nl.ndarray, optional): [batch, nheads, dstate, headdim], Initial
            hidden state. Default: None (zeros).
        causal_mask (nl.ndarray): [Q, Q], Lower triangular mask. Required.
            Pass np.tril(np.ones((Q, Q), dtype=np.float32)).

    Returns:
        tuple: (y, final_state)
            - y (nl.ndarray): [batch, nheads, seqlen, headdim], Output tensor with same
              dtype as x.
            - final_state (nl.ndarray): [batch, nheads, dstate, headdim], Final hidden
              state in float32.

    Notes:
        - chunk_size <= 128 (must fit in partition dimension)
        - dstate <= 128 (for nc_transpose and matmul stationary free dim)
        - headdim <= 512 (PSUM free dimension limit on gen2/3)
        - seqlen must be divisible by chunk_size
        - ngroups=1 (B/C shared across all heads)
        - Uses float32 accumulation internally for numerical stability
        - A should be negative for stable dynamics (decay < 1)
        - dt should be positive; discretization computes exp(dt * A)
        - Inter-chunk state propagation is sequential; intra-chunk uses matmuls

    Pseudocode:
        for batch_idx in range(batch):
            for head_idx in range(nheads):
                state = initial_state or zeros(dstate, headdim)
                for chunk_idx in range(num_chunks):
                    cs = cumsum(dt_chunk * A[head_idx])
                    CB = C_chunk @ B_chunk^T
                    Y_intra = exp(cs) * ((CB * causal) @ (exp(-cs) * dt * x))
                    Y_off = exp(cs) * (C_chunk @ state)
                    state = exp(cs[-1]) * state + B_chunk^T @ (dt * x * exp(cs[-1] - cs))
                    y_chunk = Y_intra + Y_off [+ D * x]
    """
    # Input validation
    kernel_assert(len(x.shape) == 4, f"x must be 4D (B, H, L, D), got {x.shape}")
    kernel_assert(len(dt.shape) == 3, f"dt must be 3D (B, H, L), got {dt.shape}")
    kernel_assert(len(A.shape) == 1, f"A must be 1D (H,), got {A.shape}")
    kernel_assert(len(B.shape) == 3, f"B must be 3D (B, L, N), got {B.shape}")
    kernel_assert(len(C.shape) == 3, f"C must be 3D (B, L, N), got {C.shape}")
    kernel_assert(chunk_size <= 128, f"chunk_size must be <= 128, got {chunk_size}")

    batch = x.shape[0]
    nheads = x.shape[1]
    seqlen = x.shape[2]
    headdim = x.shape[3]
    dstate = B.shape[2]
    Q = chunk_size
    num_chunks = div_ceil(seqlen, Q)

    kernel_assert(seqlen % Q == 0, f"seqlen ({seqlen}) must be divisible by chunk_size ({Q})")
    kernel_assert(dstate <= 128, f"dstate must be <= 128, got {dstate}")
    kernel_assert(headdim <= 512, f"headdim must be <= 512 for gen2/3, got {headdim}")
    kernel_assert(dt.shape == (batch, nheads, seqlen), f"dt shape mismatch")
    kernel_assert(A.shape[0] == nheads, f"A.shape[0] must equal nheads")
    kernel_assert(B.shape[0] == batch and B.shape[1] == seqlen, f"B shape mismatch")
    kernel_assert(C.shape == B.shape, f"C shape must match B shape")

    A_2d = A.reshape((nheads, 1))
    # Reshape dt for clean DMA (compiler can't handle 3D→2D slice)
    dt_2d = dt.reshape((batch * nheads, seqlen))  # (1, Q) loads for scan
    dt_flat = dt.reshape((batch * nheads * seqlen, 1))  # (Q, 1) loads for element-wise

    # Allocate outputs
    y = nl.ndarray((batch, nheads, seqlen, headdim), dtype=x.dtype, buffer=nl.shared_hbm)
    final_state_out = nl.ndarray((batch, nheads, dstate, headdim), dtype=nl.float32, buffer=nl.shared_hbm)

    # Causal mask: lower triangular — passed as (Q, Q) HBM tensor
    kernel_assert(
        causal_mask != None,
        "causal_mask must be provided as np.tril(np.ones((Q, Q), dtype=np.float32))",
    )

    if D != None:
        kernel_assert(D.shape == (nheads,), f"D must be (nheads,), got {D.shape}")
        D_2d = D.reshape((nheads, 1))

    shuffle_mask = (0,) * 32  # Broadcast partition 0 to all partitions

    # LNC sharding: distribute heads across NeuronCores (round-robin remainder)
    _, n_prgs, prg_id = get_program_sharding_info()
    heads_per_core = nheads // n_prgs + (1 if prg_id < nheads % n_prgs else 0)
    head_offset = (nheads // n_prgs) * prg_id + min(prg_id, nheads % n_prgs)

    for batch_idx in nl.affine_range(batch):
        for h_local in nl.affine_range(heads_per_core):
            head_idx = head_offset + h_local
            # --- Per-head constants (loaded once, reused across chunks) ---

            # A_h: scalar per head (1, 1)
            A_h = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=A_h[0:1, 0:1], src=A_2d[head_idx : head_idx + 1, 0:1])

            # Causal mask (Q, Q)
            causal_sb = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=causal_sb[0:Q, 0:Q], src=causal_mask[0:Q, 0:Q])

            # Ones for cumsum scan (1, Q)
            ones_sb = nl.ndarray((1, Q), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=ones_sb, value=1.0)

            # Zero initial for cumsum (1, 1)
            zero_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=zero_11, value=0.0)

            # Initialize hidden state (dstate, headdim) — persists across chunks
            state_sb = nl.ndarray((dstate, headdim), dtype=nl.float32, buffer=nl.sbuf)
            if initial_state != None:
                nisa.dma_copy(
                    dst=state_sb[0:dstate, 0:headdim],
                    src=initial_state[batch_idx, head_idx, 0:dstate, 0:headdim],
                )
            else:
                nisa.memset(dst=state_sb, value=0.0)

            # D broadcast (Q, 1) — constant across chunks
            if D != None:
                D_val = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=D_val[0:1, 0:1], src=D_2d[head_idx : head_idx + 1, 0:1])
                D_Q = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                for shuf_idx in nl.static_range((Q + 31) // 32):
                    cur_npar = min(32, Q - shuf_idx * 32)
                    nisa.nc_stream_shuffle(
                        src=D_val[0:1, 0:1],
                        dst=D_Q[shuf_idx * 32 : shuf_idx * 32 + cur_npar, 0:1],
                        shuffle_mask=shuffle_mask,
                    )

            # Upper-triangular mask = transpose(causal) — precomputed once per head
            # Used to apply causal mask to CB^T directly (avoiding a per-chunk transpose)
            triu_psum = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=triu_psum[0:Q, 0:Q], data=causal_sb[0:Q, 0:Q])
            triu_sb = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=triu_sb[0:Q, 0:Q], src=triu_psum[0:Q, 0:Q])

            # --- Sequential chunk processing (state carries between chunks) ---
            for chunk_idx in nl.sequential_range(num_chunks):
                chunk_start = chunk_idx * Q

                # ====== Load chunk inputs ======

                # x: (Q, headdim)
                x_sb = nl.ndarray((Q, headdim), dtype=x.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=x_sb[0:Q, 0:headdim],
                    src=x[batch_idx, head_idx, chunk_start : chunk_start + Q, 0:headdim],
                )

                # dt: (1, Q) in free dimension for cumsum scan
                # Load from 2D dt_2d[B*H, L] for clean DMA
                dt_row = batch_idx * nheads + head_idx
                dt_f = nl.ndarray((1, Q), dtype=dt.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=dt_f[0:1, 0:Q],
                    src=dt_2d[dt_row : dt_row + 1, chunk_start : chunk_start + Q],
                )

                # B: (Q, dstate)
                B_sb = nl.ndarray((Q, dstate), dtype=B.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=B_sb[0:Q, 0:dstate],
                    src=B[batch_idx, chunk_start : chunk_start + Q, 0:dstate],
                )

                # C: (Q, dstate)
                C_sb = nl.ndarray((Q, dstate), dtype=C.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=C_sb[0:Q, 0:dstate],
                    src=C[batch_idx, chunk_start : chunk_start + Q, 0:dstate],
                )

                # ====== Step 1: Cumulative decay ======
                # log_decay = dt * A_h, then cs = cumsum(log_decay), all in (1, Q)

                log_decay_f = nl.ndarray((1, Q), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=log_decay_f[0:1, 0:Q],
                    data=dt_f[0:1, 0:Q],
                    op0=nl.multiply,
                    operand0=A_h[0:1, 0:1],
                )

                cs_f = nl.ndarray((1, Q), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor_scan(
                    dst=cs_f[0:1, 0:Q],
                    data0=ones_sb[0:1, 0:Q],
                    data1=log_decay_f[0:1, 0:Q],
                    initial=zero_11[0:1, 0:1],
                    op0=nl.multiply,
                    op1=nl.add,
                )

                # Transpose cs from (1, Q) to (Q, 1) via padded nc_transpose through PSUM
                # (trn1 doesn't support intermediate HBM allocations)
                cs_padded = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=cs_padded[0:1, 0:Q], src=cs_f[0:1, 0:Q])
                cs_tp_psum = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=cs_tp_psum[0:Q, 0:Q], data=cs_padded[0:Q, 0:Q])
                cs_p = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=cs_p[0:Q, 0:1], src=cs_tp_psum[0:Q, 0:1])

                # exp(cs) per partition row — used for output scaling
                exp_cs_p = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(op=nl.exp, data=cs_p[0:Q, 0:1], dst=exp_cs_p[0:Q, 0:1])

                # exp(-cs) per partition row — used for input scaling
                neg_cs_p = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=neg_cs_p[0:Q, 0:1],
                    data=cs_p[0:Q, 0:1],
                    op0=nl.multiply,
                    operand0=-1.0,
                )
                exp_neg_cs_p = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(op=nl.exp, data=neg_cs_p[0:Q, 0:1], dst=exp_neg_cs_p[0:Q, 0:1])

                # Load dt directly as (Q, 1) from flat reshape — avoids Q×Q nc_transpose
                dt_flat_start = dt_row * seqlen + chunk_start
                dt_p = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=dt_p[0:Q, 0:1],
                    src=dt_flat[dt_flat_start : dt_flat_start + Q, 0:1],
                )

                # dt * x: (Q, headdim) — partition-broadcast of dt_p across headdim
                dtx = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=dtx[0:Q, 0:headdim],
                    data=x_sb[0:Q, 0:headdim],
                    op0=nl.multiply,
                    operand0=dt_p[0:Q, 0:1],
                )

                # ====== Step 2: Intra-chunk structured attention ======
                # Y_intra = exp(cs) * ((CB * causal_mask) @ (exp(-cs) * dt * x))
                # where CB = C @ B^T

                # Convert B, C to float32 for matmul consistency with state
                B_f32 = nl.ndarray((Q, dstate), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=B_f32[0:Q, 0:dstate], src=B_sb[0:Q, 0:dstate])

                C_f32 = nl.ndarray((Q, dstate), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=C_f32[0:Q, 0:dstate], src=C_sb[0:Q, 0:dstate])

                # Transpose C and B: (Q, N) → (N, Q) via PSUM
                # nc_matmul: result[M, N] = sum_K stationary[K, M] * moving[K, N]
                C_T_psum = nl.ndarray((dstate, Q), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=C_T_psum[0:dstate, 0:Q], data=C_f32[0:Q, 0:dstate])
                C_T = nl.ndarray((dstate, Q), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=C_T[0:dstate, 0:Q], src=C_T_psum[0:dstate, 0:Q])

                B_T_psum = nl.ndarray((dstate, Q), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=B_T_psum[0:dstate, 0:Q], data=B_f32[0:Q, 0:dstate])
                B_T = nl.ndarray((dstate, Q), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=B_T[0:dstate, 0:Q], src=B_T_psum[0:dstate, 0:Q])

                # CB^T = B @ C^T (compute transposed directly, avoiding per-chunk transpose)
                # stationary=B_T[K=N, M=Q], moving=C_T[K=N, N=Q] → CB^T(Q, Q)
                CB_T_psum = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=CB_T_psum[0:Q, 0:Q],
                    stationary=B_T[0:dstate, 0:Q],
                    moving=C_T[0:dstate, 0:Q],
                )

                CB_T = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=CB_T[0:Q, 0:Q], src=CB_T_psum[0:Q, 0:Q])

                # Apply upper-triangular mask: (CB * tril)^T = CB^T * triu
                nisa.tensor_tensor(
                    dst=CB_T[0:Q, 0:Q],
                    data1=CB_T[0:Q, 0:Q],
                    data2=triu_sb[0:Q, 0:Q],
                    op=nl.multiply,
                )

                # Scale input: X_scaled = dtx * exp(-cs) — (Q, headdim)
                X_scaled = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=X_scaled[0:Q, 0:headdim],
                    data=dtx[0:Q, 0:headdim],
                    op0=nl.multiply,
                    operand0=exp_neg_cs_p[0:Q, 0:1],
                )

                # Y_intra = CB_causal @ X_scaled
                # stationary=CB_T[K=Q, M=Q], moving=X_scaled[K=Q, N=D] → result (Q, D)
                Y_psum = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=Y_psum[0:Q, 0:headdim],
                    stationary=CB_T[0:Q, 0:Q],
                    moving=X_scaled[0:Q, 0:headdim],
                )

                Y_intra = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=Y_intra[0:Q, 0:headdim], src=Y_psum[0:Q, 0:headdim])

                # Scale by exp(cs)
                nisa.tensor_scalar(
                    dst=Y_intra[0:Q, 0:headdim],
                    data=Y_intra[0:Q, 0:headdim],
                    op0=nl.multiply,
                    operand0=exp_cs_p[0:Q, 0:1],
                )

                # ====== Step 3: State-to-output ======
                # Y_off = exp(cs) * (C @ state)
                # Uses state from PREVIOUS chunk (before update)
                # stationary=C_T[K=N, M=Q], moving=state_sb[K=N, N=D] → (Q, D)
                Y_off_psum = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=Y_off_psum[0:Q, 0:headdim],
                    stationary=C_T[0:dstate, 0:Q],
                    moving=state_sb[0:dstate, 0:headdim],
                )

                Y_off = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=Y_off[0:Q, 0:headdim], src=Y_off_psum[0:Q, 0:headdim])

                nisa.tensor_scalar(
                    dst=Y_off[0:Q, 0:headdim],
                    data=Y_off[0:Q, 0:headdim],
                    op0=nl.multiply,
                    operand0=exp_cs_p[0:Q, 0:1],
                )

                # ====== Step 4: Update hidden state ======
                # state_new = exp(cs_last) * state + B^T @ (dtx * exp(cs_last - cs))

                # exp(cs_last) from cs_f at partition 0 (safe for shuffle)
                exp_cs_last = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    op=nl.exp,
                    data=cs_f[0:1, Q - 1 : Q],
                    dst=exp_cs_last[0:1, 0:1],
                )

                # Broadcast exp(cs_last) to (Q, 1) for decay_to_end
                exp_cs_last_Q = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                for shuf_idx in nl.static_range((Q + 31) // 32):
                    cur_npar = min(32, Q - shuf_idx * 32)
                    nisa.nc_stream_shuffle(
                        src=exp_cs_last[0:1, 0:1],
                        dst=exp_cs_last_Q[shuf_idx * 32 : shuf_idx * 32 + cur_npar, 0:1],
                        shuffle_mask=shuffle_mask,
                    )

                # decay_to_end = exp(cs_last - cs) = exp(-cs) * exp(cs_last)
                decay_to_end = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=decay_to_end[0:Q, 0:1],
                    data1=exp_neg_cs_p[0:Q, 0:1],
                    data2=exp_cs_last_Q[0:Q, 0:1],
                    op=nl.multiply,
                )

                # dtx_state = dtx * decay_to_end — (Q, headdim)
                dtx_state = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=dtx_state[0:Q, 0:headdim],
                    data=dtx[0:Q, 0:headdim],
                    op0=nl.multiply,
                    operand0=decay_to_end[0:Q, 0:1],
                )

                # chunk_state = B^T @ dtx_state
                # stationary=B_f32[K=Q, M=N], moving=dtx_state[K=Q, N=D] → (N, D)
                chunk_state_psum = nl.ndarray((dstate, headdim), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=chunk_state_psum[0:dstate, 0:headdim],
                    stationary=B_f32[0:Q, 0:dstate],
                    moving=dtx_state[0:Q, 0:headdim],
                )

                chunk_state_sb = nl.ndarray((dstate, headdim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=chunk_state_sb[0:dstate, 0:headdim],
                    src=chunk_state_psum[0:dstate, 0:headdim],
                )

                # Broadcast exp(cs_last) to (dstate, 1) for state scaling
                exp_cs_last_N = nl.ndarray((dstate, 1), dtype=nl.float32, buffer=nl.sbuf)
                for shuf_idx in nl.static_range((dstate + 31) // 32):
                    cur_npar = min(32, dstate - shuf_idx * 32)
                    nisa.nc_stream_shuffle(
                        src=exp_cs_last[0:1, 0:1],
                        dst=exp_cs_last_N[shuf_idx * 32 : shuf_idx * 32 + cur_npar, 0:1],
                        shuffle_mask=shuffle_mask,
                    )

                # state = exp(cs_last) * state + chunk_state
                nisa.tensor_scalar(
                    dst=state_sb[0:dstate, 0:headdim],
                    data=state_sb[0:dstate, 0:headdim],
                    op0=nl.multiply,
                    operand0=exp_cs_last_N[0:dstate, 0:1],
                )
                nisa.tensor_tensor(
                    dst=state_sb[0:dstate, 0:headdim],
                    data1=state_sb[0:dstate, 0:headdim],
                    data2=chunk_state_sb[0:dstate, 0:headdim],
                    op=nl.add,
                )

                # ====== Step 5: Combine and store output ======

                # y = Y_intra + Y_off
                y_chunk = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=y_chunk[0:Q, 0:headdim],
                    data1=Y_intra[0:Q, 0:headdim],
                    data2=Y_off[0:Q, 0:headdim],
                    op=nl.add,
                )

                # D skip connection: y += D_h * x
                if D != None:
                    Dx = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(
                        dst=Dx[0:Q, 0:headdim],
                        data=x_sb[0:Q, 0:headdim],
                        op0=nl.multiply,
                        operand0=D_Q[0:Q, 0:1],
                    )
                    nisa.tensor_tensor(
                        dst=y_chunk[0:Q, 0:headdim],
                        data1=y_chunk[0:Q, 0:headdim],
                        data2=Dx[0:Q, 0:headdim],
                        op=nl.add,
                    )

                # Store output chunk
                nisa.dma_copy(
                    dst=y[batch_idx, head_idx, chunk_start : chunk_start + Q, 0:headdim],
                    src=y_chunk[0:Q, 0:headdim],
                )

            # Store final hidden state after all chunks
            nisa.dma_copy(
                dst=final_state_out[batch_idx, head_idx, 0:dstate, 0:headdim],
                src=state_sb[0:dstate, 0:headdim],
            )

    return y, final_state_out
