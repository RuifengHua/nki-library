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

"""MLP Backward MXFP8 kernel implementations."""

from .config import (
    DEFAULT_MATMUL_CONFIG,
    L_TILE_K,
    TILE_K,
    TILE_M,
    TILE_N,
    BlockConfig,
    MatmulConfig,
    get_autotuned_config,
    get_config_for_shape,
)
from .mlp_bwd_mxfp8_kernel import mlp_backward_mxfp8_base_nki, mlp_backward_mxfp8_nki
from .recompute import (
    recompute_gate_act,
    recompute_gate_up_projection,
    recompute_hidden,
)

__all__ = [
    'mlp_backward_mxfp8_nki',
    'mlp_backward_mxfp8_base_nki',
    'TILE_M',
    'TILE_K',
    'TILE_N',
    'L_TILE_K',
    'BlockConfig',
    'MatmulConfig',
    'DEFAULT_MATMUL_CONFIG',
    'get_config_for_shape',
    'get_autotuned_config',
    'recompute_gate_up_projection',
    'recompute_gate_act',
    'recompute_hidden',
]
