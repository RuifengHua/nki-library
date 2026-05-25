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
"""Shared utilities for model config generators."""

import math


def get_sharded_head_counts(tp, n_q_heads, n_kv_heads):
    """Compute per-rank Q and KV head counts after TP sharding, with padding.

    Handles GQA/MQA head count padding when n_kv_heads doesn't evenly divide tp.
    """
    padded_q = math.ceil(n_q_heads / tp) * tp
    if n_q_heads == n_kv_heads:
        padded_kv = padded_q
    elif n_kv_heads < tp or n_kv_heads % tp != 0:
        padded_kv = tp if tp % n_kv_heads == 0 else padded_q
    else:
        padded_kv = n_kv_heads
    return padded_q // tp, padded_kv // tp
