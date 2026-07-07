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
"""Unit tests for artifact upload / --force-local-cleanup interaction.

Regression coverage for the "neuron-explorer capture ... open inp-*.bin: no such
file or directory" failure: under fleet contention the host-assignment loop can
re-enter ``prepare_host`` (a host rotation) and upload the *same* once-dumped
artifact directory a second time. If ``--force-local-cleanup`` deleted the input
``*.bin`` files right after the first upload's tarball was built, the second
upload ships an archive that is missing the inputs, and ``neuron-explorer`` on the
second host fails to open them.

These tests exercise the mechanism deterministically (no hardware, no fleet
contention) by uploading the same directory twice with cleanup enabled and
asserting the inputs are still present in what gets shipped on the retry.
"""

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from test.utils.resources import RemoteDirectory


@pytest.fixture
def source_dir():
    """A dumped artifact directory: two large input .bin tensors + a scalar arg."""
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "inp-q-000.bin"), "wb") as f:
            f.write(b"\0" * 4096)
        with open(os.path.join(tmp, "inp-k_cache-000.bin"), "wb") as f:
            f.write(b"\0" * 4096)
        # Scalar args are pickled without a .bin extension; they are NOT cleaned up.
        with open(os.path.join(tmp, "inp-scale-000"), "wb") as f:
            f.write(b"x")
        # The compiled neff also lives in the dir and is not a .bin.
        with open(os.path.join(tmp, "file.neff"), "wb") as f:
            f.write(b"\0" * 1024)
        yield tmp


def _make_remote_dir():
    """A RemoteDirectory whose network send is stubbed to snapshot what it ships."""
    remote = RemoteDirectory(path="/tmp/neuronx-cc/tests/out-x_pid1", connection=MagicMock())
    shipped_snapshots: list[set[str]] = []

    # _send_compressed does `tar -C local_path .`, i.e. it ships whatever files
    # exist in local_path at call time. Snapshot that set instead of doing I/O.
    def fake_send(local_path, remote_path):
        shipped_snapshots.append(set(os.listdir(local_path)))

    remote._send_compressed = fake_send  # type: ignore[method-assign]
    return remote, shipped_snapshots


def test_retry_reupload_still_ships_input_bins(source_dir):
    """A rotation re-uploads the same dir; the retry's archive must keep the inputs.

    This is the exact failure path: attempt #1 uploads + (with cleanup) may drop
    the local .bin, then the host-assignment loop rotates and attempt #2 uploads
    the SAME dir. The second archive must still contain inp-*.bin, otherwise
    neuron-explorer on the second host cannot open them.
    """
    remote, shipped = _make_remote_dir()
    collector = MagicMock()

    # Attempt #1: upload with cleanup enabled (as the pipeline runs it).
    remote.upload_sftp(source_dir, collector, force_local_cleanup=True)
    # Attempt #2: a host rotation re-uploads the same once-dumped directory.
    remote.upload_sftp(source_dir, collector, force_local_cleanup=True)

    assert len(shipped) == 2, "expected two uploads (initial + rotation retry)"
    retry_archive = shipped[1]
    missing = {"inp-q-000.bin", "inp-k_cache-000.bin"} - retry_archive
    assert not missing, (
        f"retry upload shipped an archive missing input tensors {missing}; "
        f"neuron-explorer would fail with 'open <file>: no such file or directory'. "
        f"Archive contained: {sorted(retry_archive)}"
    )
