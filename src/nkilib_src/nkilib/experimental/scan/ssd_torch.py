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

"""PyTorch reference implementation for SSD (Mamba-2) chunk-wise parallel scan kernel."""

import torch


def ssd_torch_ref(x, dt, A, B, C, chunk_size=128, D=None, initial_state=None):
    """
    PyTorch reference implementation of SSD (State Space Duality) for Mamba-2.

    Implements chunk-wise computation matching the NKI kernel:
    1. Cumulative decay: cs = cumsum(dt * A)
    2. Intra-chunk: Y_intra = exp(cs) * ((CB * causal) @ (exp(-cs) * dt * x))
    3. State-to-output: Y_off = exp(cs) * (C @ state)
    4. State update: state = exp(cs_last) * state + B^T @ (dt * x * exp(cs_last - cs))
    5. Output: y = Y_intra + Y_off [+ D * x]

    Args:
        x (torch.Tensor): Input tensor (batch, nheads, seqlen, headdim).
        dt (torch.Tensor): Timestep tensor (batch, nheads, seqlen). Should be positive.
        A (torch.Tensor): State transition scalar per head (nheads,). Should be negative.
        B (torch.Tensor): Input projection (batch, seqlen, dstate).
        C (torch.Tensor): Output projection (batch, seqlen, dstate).
        chunk_size (int): Chunk size Q.
        D (torch.Tensor, optional): Skip connection weights (nheads,).
        initial_state (torch.Tensor, optional): Initial state (batch, nheads, dstate, headdim).

    Returns:
        dict: {"y": torch.Tensor, "final_state": torch.Tensor}
    """
    batch, nheads, seqlen, headdim = x.shape
    dstate = B.shape[2]
    Q = chunk_size
    num_chunks = seqlen // Q

    x_f = x.float()
    dt_f = dt.float()
    A_f = A.float()
    B_f = B.float()
    C_f = C.float()

    y = torch.zeros((batch, nheads, seqlen, headdim), dtype=torch.float32)
    final_state = torch.zeros((batch, nheads, dstate, headdim), dtype=torch.float32)

    causal = torch.tril(torch.ones(Q, Q, dtype=torch.float32))

    for b in range(batch):
        for h in range(nheads):
            A_h = A_f[h]

            # Initialize hidden state (dstate, headdim)
            if initial_state is not None:
                state = initial_state[b, h].float().clone()  # (dstate, headdim)
            else:
                state = torch.zeros((dstate, headdim), dtype=torch.float32)

            for ic in range(num_chunks):
                cs_start = ic * Q
                cs_end = cs_start + Q

                # Load chunk inputs
                x_chunk = x_f[b, h, cs_start:cs_end, :]  # (Q, headdim)
                dt_chunk = dt_f[b, h, cs_start:cs_end]  # (Q,)
                B_chunk = B_f[b, cs_start:cs_end, :]  # (Q, dstate)
                C_chunk = C_f[b, cs_start:cs_end, :]  # (Q, dstate)

                # Step 1: Cumulative decay
                log_decay = dt_chunk * A_h  # (Q,)
                cs = torch.cumsum(log_decay, dim=0)  # (Q,)

                exp_cs = torch.exp(cs)  # (Q,)
                exp_neg_cs = torch.exp(-cs)  # (Q,)

                # dt * x: (Q, headdim)
                dtx = dt_chunk.unsqueeze(-1) * x_chunk  # (Q, headdim)

                # Step 2: Intra-chunk structured attention
                # CB = C @ B^T: (Q, Q)
                CB = C_chunk @ B_chunk.T

                # Apply causal mask
                CB_causal = CB * causal

                # Scale input: X_scaled = dtx * exp(-cs)
                X_scaled = dtx * exp_neg_cs.unsqueeze(-1)  # (Q, headdim)

                # Y_intra = exp(cs) * (CB_causal @ X_scaled)
                Y_intra = exp_cs.unsqueeze(-1) * (CB_causal @ X_scaled)  # (Q, headdim)

                # Step 3: State-to-output
                # Y_off = exp(cs) * (C @ state)
                Y_off = exp_cs.unsqueeze(-1) * (C_chunk @ state)  # (Q, headdim)

                # Step 4: Update hidden state
                # state = exp(cs_last) * state + B^T @ (dtx * exp(cs_last - cs))
                cs_last = cs[-1]
                exp_cs_last = torch.exp(cs_last)
                decay_to_end = torch.exp(cs_last - cs)  # (Q,)

                dtx_state = dtx * decay_to_end.unsqueeze(-1)  # (Q, headdim)
                chunk_state = B_chunk.T @ dtx_state  # (dstate, headdim)

                state = exp_cs_last * state + chunk_state

                # Step 5: Combine output
                y_chunk = Y_intra + Y_off
                if D is not None:
                    D_h = D[h].float()
                    y_chunk = y_chunk + D_h * x_chunk

                y[b, h, cs_start:cs_end, :] = y_chunk

            final_state[b, h] = state

    return {"y": y, "final_state": final_state}
