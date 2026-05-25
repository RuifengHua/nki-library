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
"""Unit tests for ProcessTreeMemoryMonitor."""

import os
import time

import pytest

from ..utils.memory_monitor import MemoryLimitExceeded, ProcessTreeMemoryMonitor, release_process_memory


def test_monitor_tracks_peak_rss():
    """Monitor should return a non-zero peak RSS for the current process."""
    monitor = ProcessTreeMemoryMonitor(os.getpid(), interval_seconds=0.1)
    monitor.start()
    time.sleep(0.3)
    snapshot = monitor.stop()
    assert snapshot.peak_rss_bytes > 0
    assert snapshot.peak_rss_mb > 0


def test_monitor_captures_baseline():
    """Monitor should record baseline RSS at start and compute delta."""
    monitor = ProcessTreeMemoryMonitor(os.getpid(), interval_seconds=0.1)
    monitor.start()
    time.sleep(0.3)
    snapshot = monitor.stop()
    assert snapshot.baseline_rss_bytes > 0
    assert snapshot.peak_rss_bytes >= snapshot.baseline_rss_bytes
    assert snapshot.delta_rss_bytes == snapshot.peak_rss_bytes - snapshot.baseline_rss_bytes
    assert snapshot.delta_rss_mb >= 0


def test_memory_limit_exceeded_raises():
    """Monitor with a limit below current RSS should raise MemoryLimitExceeded."""
    # Use a 10 MB limit and allocate ~50 MB to guarantee the delta exceeds it
    # even when run after 670+ other tests with a noisy RSS baseline.
    limit = 10 * 1024 * 1024
    monitor = ProcessTreeMemoryMonitor(os.getpid(), interval_seconds=0.1, memory_limit_bytes=limit)
    monitor.start()
    with pytest.raises(MemoryLimitExceeded, match="exceeded --memory-limit"):
        # Allocate ~50 MB so delta RSS clearly exceeds the 10 MB limit.
        # bytearray is backed by a single C malloc (mmap for >128KB),
        # so new pages are faulted in and RSS grows immediately.
        _buf = bytearray(50 * 1024 * 1024)
        time.sleep(2)


def test_memory_limit_checks_delta():
    """Memory limit should be checked against delta RSS, not absolute RSS.

    A limit larger than the process baseline but set to 1 byte should not
    trigger because the delta (new allocations) during sleep is ~0.
    """
    monitor = ProcessTreeMemoryMonitor(os.getpid(), interval_seconds=0.1)
    monitor.start()
    time.sleep(0.2)
    baseline_snapshot = monitor.stop()

    # Set limit well above any realistic delta but well below absolute RSS.
    # If limit were checked against absolute RSS, this would always fire.
    limit = baseline_snapshot.peak_rss_bytes - 1
    if limit <= 0:
        pytest.skip("Process RSS too low to test delta vs absolute distinction")

    monitor = ProcessTreeMemoryMonitor(os.getpid(), interval_seconds=0.1, memory_limit_bytes=limit)
    monitor.start()
    # Sleep without allocating significant memory — delta should stay near 0
    time.sleep(0.5)
    snapshot = monitor.stop()
    # Should not have raised — delta is small even though absolute RSS > limit
    assert snapshot.delta_rss_mb < snapshot.peak_rss_mb


def test_release_process_memory_runs_without_error():
    """release_process_memory should not raise on Linux."""
    release_process_memory()
