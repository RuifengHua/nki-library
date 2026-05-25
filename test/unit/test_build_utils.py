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
"""Unit tests for build_utils.get_version."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "build-tools" / "bin"))

from build_utils import get_version


class TestGetVersion:
    """Tests for the get_version utility."""

    def test_reads_real_version_file(self):
        from nkilib_src.nkilib.version import __version__

        version = get_version()
        assert version == __version__

    def test_raises_file_not_found_when_version_file_missing(self):
        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Version file not found"):
                get_version()
