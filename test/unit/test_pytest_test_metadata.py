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
"""Unit tests for pytest_test_metadata helpers."""

from ..utils.pytest_test_metadata import derive_labeled_kernel_name


def test_core_kernel_returns_bare_name():
    """Core kernels keep the unqualified metadata name (matches pipeline)."""
    path = "/repo/test/integration/nkilib/core/mlp/test_mlp_tkg.py"
    assert derive_labeled_kernel_name(path, "MLP TKG") == "MLP TKG"


def test_experimental_kernel_is_prefixed():
    """Non-core families get the folder capitalized and prepended."""
    path = "/repo/test/integration/nkilib/experimental/moe/moe_tkg/test_x.py"
    assert derive_labeled_kernel_name(path, "MoE TKG") == "Experimental MoE TKG"


def test_missing_metadata_returns_none():
    path = "/repo/test/integration/nkilib/core/mlp/test_mlp_tkg.py"
    assert derive_labeled_kernel_name(path, None) is None


def test_path_outside_test_package_falls_back_to_bare_name():
    """Paths that don't live under integration/nkilib skip the prefix rather than guessing."""
    path = "/some/other/tree/test_mlp.py"
    assert derive_labeled_kernel_name(path, "MLP TKG") == "MLP TKG"


def test_integration_without_nkilib_falls_back_to_bare_name():
    """The marker is specifically integration/nkilib — a lone 'integration' is not enough."""
    path = "/repo/test/integration/other/test_x.py"
    assert derive_labeled_kernel_name(path, "X") == "X"


def test_file_directly_under_nkilib_falls_back_to_bare_name():
    """CDK treats this case (pathParts length 1) as having no family prefix; mirror that."""
    path = "/repo/test/integration/nkilib/test_x.py"
    assert derive_labeled_kernel_name(path, "X") == "X"
