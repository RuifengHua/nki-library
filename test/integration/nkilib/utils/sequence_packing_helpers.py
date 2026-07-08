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

"""Helpers for building bound_min / bound_max tensors from cu_seqlens for striped CP sequence packing."""

import numpy as np


def cu_seqlens_to_striped_bounds(cu_seqlens, total_seqlen, cp_degree):
    """Convert global cu_seqlens into local bound_min / bound_max for striped CP.

    Assumes every entry in cu_seqlens is a multiple of cp_degree — i.e., each
    padded document length is already a multiple of cp_degree. Under that
    invariant, striping (x[:, r::cp_degree, :]) produces an identical local
    document layout on every rank, so the same bound_min / bound_max tensors
    are valid for all ranks.

    Args:
        cu_seqlens (np.ndarray or list): Shape (num_docs + 1,), strictly
            increasing, starts at 0, last entry equals total_seqlen. Every
            entry must be a multiple of cp_degree.
        total_seqlen (int): Total packed sequence length (multiple of cp_degree).
        cp_degree (int): Context-parallel degree.

    Returns:
        bound_min, bound_max: np.ndarray fp32, shape (total_seqlen // cp_degree,),
            identical on every rank.
    """
    cu_seqlens = np.asarray(cu_seqlens, dtype=np.int64)
    assert total_seqlen % cp_degree == 0, f"total_seqlen {total_seqlen} not divisible by cp_degree {cp_degree}"
    assert cu_seqlens[0] == 0 and cu_seqlens[-1] == total_seqlen, "cu_seqlens must start at 0 and end at total_seqlen"
    for seq_end in cu_seqlens:
        assert int(seq_end) % cp_degree == 0, (
            f"cu_seqlens entry {int(seq_end)} not divisible by cp_degree {cp_degree}; "
            "each padded doc length must be a multiple of cp_degree"
        )

    local_total = total_seqlen // cp_degree
    bound_min = np.zeros(local_total, dtype=np.float32)
    bound_max = np.zeros(local_total, dtype=np.float32)
    for doc_idx in range(len(cu_seqlens) - 1):
        local_start = int(cu_seqlens[doc_idx]) // cp_degree
        local_end = int(cu_seqlens[doc_idx + 1]) // cp_degree
        bound_min[local_start:local_end] = local_start
        bound_max[local_start:local_end] = local_end
    return bound_min, bound_max


def stripe_tensor(full_tensor, rank, cp_degree, seq_axis=1):
    """Extract this rank's stripe from a full tensor: full[..., rank::cp_degree, ...]."""
    index = [slice(None)] * full_tensor.ndim
    index[seq_axis] = slice(rank, None, cp_degree)
    return full_tensor[tuple(index)]


def unstripe_tensors(striped_tensors, cp_degree, seq_axis=1):
    """Inverse of stripe_tensor: reassemble the full tensor from per-rank stripes."""
    assert len(striped_tensors) == cp_degree
    full_shape = list(striped_tensors[0].shape)
    full_shape[seq_axis] = full_shape[seq_axis] * cp_degree
    full = np.empty(full_shape, dtype=striped_tensors[0].dtype)
    for rank in range(cp_degree):
        index = [slice(None)] * full.ndim
        index[seq_axis] = slice(rank, None, cp_degree)
        full[tuple(index)] = striped_tensors[rank]
    return full
