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
"""Process tree memory monitoring for identifying memory-hungry tests.

Polls the RSS (Resident Set Size) of the current process and all its descendants
via /proc on Linux, tracking the peak value observed during a test. This captures
memory used by both the Python test process and any child processes such as the
neuronx-cc compiler.

The monitor tracks both absolute peak RSS and delta RSS (peak minus baseline at
start). The memory limit (``--memory-limit``) is enforced against the **delta**
so that memory accumulated by prior tests on the same xdist worker does not
count toward the limit.

Before capturing the baseline, :func:`release_process_memory` runs garbage
collection and ``malloc_trim(0)`` to return freed pages to the OS, reducing
baseline noise for borderline tests.

Usage:
    Run pytest with --monitor-memory to enable per-test peak RSS tracking.
    Results are written to memory_monitor.csv in the output directory.

    Run pytest with --memory-limit N to kill tests that exceed N MB of *new*
    memory (delta RSS).
"""

import ctypes
import ctypes.util
import gc
import os
import signal
import threading
from dataclasses import dataclass


@dataclass
class MemorySnapshot:
    """Peak memory measurement result for a single test."""

    peak_rss_bytes: int = 0
    baseline_rss_bytes: int = 0

    @property
    def peak_rss_mb(self) -> float:
        return self.peak_rss_bytes / (1024 * 1024)

    @property
    def delta_rss_bytes(self) -> int:
        return max(0, self.peak_rss_bytes - self.baseline_rss_bytes)

    @property
    def delta_rss_mb(self) -> float:
        return self.delta_rss_bytes / (1024 * 1024)


class MemoryLimitExceeded(Exception):
    """Raised when a test's delta RSS exceeds --memory-limit."""


def _get_descendant_pids(pid: int) -> list[int]:
    """Get all descendant PIDs recursively via /proc/<pid>/task/<pid>/children."""
    try:
        with open(f"/proc/{pid}/task/{pid}/children") as f:
            children = [int(p) for p in f.read().split()]
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return []

    descendants = list(children)
    for child in children:
        descendants.extend(_get_descendant_pids(child))
    return descendants


def _read_rss_bytes(pid: int) -> int:
    """Read VmRSS of a single process from /proc/<pid>/status, in bytes."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024  # kB -> bytes
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return 0


def get_process_tree_rss_bytes(pid: int) -> int:
    """Get total RSS of a process and all its descendants, in bytes."""
    pids = [pid] + _get_descendant_pids(pid)
    return sum(_read_rss_bytes(p) for p in pids)


def release_process_memory() -> None:
    """Release freed memory back to the OS.

    Runs a full garbage collection cycle, then calls libc ``malloc_trim(0)``
    to return free heap pages to the operating system. This reduces RSS baseline
    noise for borderline tests near the memory limit.

    ``malloc_trim`` is glibc-specific; on musl or non-Linux platforms the call
    is silently skipped.
    """
    gc.collect()
    try:
        libc_name = ctypes.util.find_library("c")
        if libc_name is None:
            return
        libc = ctypes.CDLL(libc_name)
        if not hasattr(libc, "malloc_trim"):
            return
        libc.malloc_trim(0)
    except OSError:
        pass


class ProcessTreeMemoryMonitor:
    """Polls process tree RSS in a background thread, tracking the peak.

    Monitors the RSS of the given PID and all its descendant processes
    (e.g., neuronx-cc compiler subprocesses) by reading /proc at a
    configurable interval.

    The monitor records a baseline RSS at ``start()`` and tracks the peak
    RSS during the test. The memory limit is enforced against the **delta**
    (peak - baseline) so that accumulated RSS from prior tests does not
    cause false positives.

    When ``memory_limit_bytes`` is set, ``start()`` installs a SIGUSR1
    handler that raises :class:`MemoryLimitExceeded` in the main thread
    when the delta exceeds the limit. ``stop()`` restores the previous handler.

    Must be started and stopped from the main thread (signal constraint).

    Usage::

        monitor = ProcessTreeMemoryMonitor(os.getpid(), memory_limit_bytes=3 * 1024**3)
        monitor.start()
        try:
            run_test()
        finally:
            snapshot = monitor.stop()
            print(f"Peak RSS: {snapshot.peak_rss_mb:.1f} MB")
            print(f"Delta RSS: {snapshot.delta_rss_mb:.1f} MB")
    """

    def __init__(self, pid: int, interval_seconds: float = 0.5, memory_limit_bytes: int | None = None):
        self._pid = pid
        self._interval = interval_seconds
        self._baseline_rss_bytes = 0
        self._peak_rss_bytes = 0
        self._memory_limit_bytes = memory_limit_bytes
        self._limit_already_exceeded = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._old_handler: signal.Handlers | None = None

    def start(self) -> None:
        """Start monitoring in a daemon thread. Installs SIGUSR1 handler if limit is set."""
        release_process_memory()
        rss = get_process_tree_rss_bytes(self._pid)
        self._baseline_rss_bytes = rss
        self._peak_rss_bytes = rss

        if self._memory_limit_bytes is not None:
            limit_mb = self._memory_limit_bytes / (1024 * 1024)

            def _on_memory_limit(signum, frame):
                delta_mb = max(0, self._peak_rss_bytes - self._baseline_rss_bytes) / (1024 * 1024)
                raise MemoryLimitExceeded(
                    f"Memory delta ({delta_mb:.0f} MB) exceeded --memory-limit ({limit_mb:.0f} MB)"
                )

            self._old_handler = signal.signal(signal.SIGUSR1, _on_memory_limit)

        self._stop_event.clear()
        self._limit_already_exceeded = False
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="memory-monitor")
        self._thread.start()

    def stop(self) -> MemorySnapshot:
        """Stop monitoring, restore signal handler, and return the peak memory snapshot."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._old_handler is not None:
            signal.signal(signal.SIGUSR1, self._old_handler)
            self._old_handler = None
        return MemorySnapshot(
            peak_rss_bytes=self._peak_rss_bytes,
            baseline_rss_bytes=self._baseline_rss_bytes,
        )

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self._interval):
            rss = get_process_tree_rss_bytes(self._pid)
            if rss > self._peak_rss_bytes:
                self._peak_rss_bytes = rss
            delta = rss - self._baseline_rss_bytes
            if self._memory_limit_bytes and delta > self._memory_limit_bytes and not self._limit_already_exceeded:
                self._limit_already_exceeded = True
                os.kill(self._pid, signal.SIGUSR1)
