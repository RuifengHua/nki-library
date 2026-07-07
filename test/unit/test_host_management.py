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
"""Unit tests for host_management retry logic and error reporting."""

import contextlib
import json
import os
import random
import subprocess
import tempfile
from contextlib import closing
from unittest.mock import MagicMock, patch

import pytest
from paramiko import SSHException

from test.utils.common_dataclasses import Platforms, TargetHost
from test.utils.exceptions import InferenceException, LocalExecutionException
from test.utils.host_management import Host, HostManager, LocalHost, detect_local_neuron_devices, temporary_random_seed


class TestHostManagerRetryErrorReporting:
    """Test that host errors are captured and reported when all hosts fail."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_host_manager(self, temp_dir):
        """Create a HostManager with mocked hosts for testing."""
        target_hosts = [
            TargetHost(ssh_host="host1", host_type=Platforms.TRN2),
            TargetHost(ssh_host="host2", host_type=Platforms.TRN2),
        ]

        with patch("test.utils.host_management.SshHost"):
            manager = HostManager(
                base_host_info_path=temp_dir,
                target_hosts=target_hosts,
                neuron_installation_path="/opt/aws/neuron/bin",
                ssh_config_path="~/.ssh/config",
            )

        # Initialize host stats file
        host_stats_path = os.path.join(temp_dir, "host_stats.json")
        with open(host_stats_path, "w") as f:
            json.dump(
                [
                    {"host_alias": "host1", "work_queue_depth": 0, "run_id": manager.run_id, "host_type": "trn2"},
                    {"host_alias": "host2", "work_queue_depth": 0, "run_id": manager.run_id, "host_type": "trn2"},
                ],
                f,
            )

        # Create mock hosts that return their host_id
        mock_host1 = MagicMock()
        mock_host1.get_host_id.return_value = "host1"
        mock_host2 = MagicMock()
        mock_host2.get_host_id.return_value = "host2"

        manager.target_hosts = {"host1": mock_host1, "host2": mock_host2}
        manager.host_types = {"host1": Platforms.TRN2, "host2": Platforms.TRN2}

        return manager

    def test_error_details_included_when_all_retries_exhausted(self, mock_host_manager):
        """Test that error details from each host are included in final exception."""
        mock_collector = MagicMock()

        with pytest.raises(InferenceException) as exc_info:
            with closing(
                mock_host_manager.get_host_assignment_with_retry(
                    platform_target=Platforms.TRN2,
                    collector=mock_collector,
                    connection_failure_cap=2,
                )
            ) as host_generator:
                for host_attempt in host_generator:
                    with host_attempt as host:
                        # Simulate different errors on each host
                        host_id = host.get_host_id()
                        if host_id == "host1":
                            raise TimeoutError("Connection timed out after 30 seconds")
                        else:
                            raise SSHException("Authentication failed")

        # Verify error message contains details from both hosts
        error_message = str(exc_info.value)
        print("\n=== Test 1: All Retries Exhausted ===")
        print(error_message)
        print("=" * 50)
        assert "Connection error after 2 attempts" in error_message
        assert "host1" in error_message or "host2" in error_message
        assert "Errors from each host:" in error_message

    def test_error_details_included_when_no_hosts_available(self, mock_host_manager):
        """Test that previous errors are included when __get_host_assignment__ raises."""
        mock_collector = MagicMock()

        # First, manually fail host1 with a specific error
        with pytest.raises(InferenceException) as exc_info:
            with closing(
                mock_host_manager.get_host_assignment_with_retry(
                    platform_target=Platforms.TRN2,
                    collector=mock_collector,
                    connection_failure_cap=3,
                )
            ) as host_generator:
                attempt = 0
                for host_attempt in host_generator:
                    with host_attempt as host:
                        attempt += 1
                        host_id = host.get_host_id()
                        # Fail both hosts - after 2 failures, no hosts will be available
                        raise OSError(f"Network unreachable on {host_id}")

        # Verify error message contains the enriched details
        error_message = str(exc_info.value)
        print("\n=== Test 2: No Hosts Available ===")
        print(error_message)
        print("=" * 50)
        assert "Errors from" in error_message
        assert "OSError" in error_message or "Network unreachable" in error_message

    def test_single_host_failure_shows_specific_error(self, temp_dir):
        """single host fails, error details should be shown."""
        target_hosts = [
            TargetHost(ssh_host="single-host", host_type=Platforms.TRN2),
        ]

        with patch("test.utils.host_management.SshHost"):
            manager = HostManager(
                base_host_info_path=temp_dir,
                target_hosts=target_hosts,
                neuron_installation_path="/opt/aws/neuron/bin",
                ssh_config_path="~/.ssh/config",
            )

        # Initialize host stats file with single host
        host_stats_path = os.path.join(temp_dir, "host_stats.json")
        with open(host_stats_path, "w") as f:
            json.dump(
                [{"host_alias": "single-host", "work_queue_depth": 0, "run_id": manager.run_id, "host_type": "trn2"}],
                f,
            )

        mock_host = MagicMock()
        mock_host.get_host_id.return_value = "single-host"
        manager.target_hosts = {"single-host": mock_host}
        manager.host_types = {"single-host": Platforms.TRN2}

        mock_collector = MagicMock()

        with pytest.raises(InferenceException) as exc_info:
            with closing(
                manager.get_host_assignment_with_retry(
                    platform_target=Platforms.TRN2,
                    collector=mock_collector,
                    connection_failure_cap=3,
                )
            ) as host_generator:
                for host_attempt in host_generator:
                    with host_attempt as host:
                        raise TimeoutError("SSH connection timed out during inference")

        # Verify the specific error is included
        error_message = str(exc_info.value)
        print("\n=== Test 3: Single Host Failure ===")
        print(error_message)
        print("=" * 50)
        assert "single-host" in error_message
        assert "TimeoutError" in error_message
        assert "SSH connection timed out during inference" in error_message
        assert "Errors from previous host attempts:" in error_message

    def test_patience_rotation_is_transient_does_not_mark_failed(self, mock_host_manager):
        """A QueuePatienceRotation (busy FIFO queue) is transient: the host is NOT marked
        failed and stays eligible, and the loop keeps rotating until it succeeds elsewhere."""
        from test.utils.exceptions import QueuePatienceRotation

        mock_collector = MagicMock()

        with patch.object(
            mock_host_manager, "mark_host_as_failed", wraps=mock_host_manager.mark_host_as_failed
        ) as mark_failed:
            first_host_id = None
            successful_host_id = None
            with closing(
                mock_host_manager.get_host_assignment_with_retry(
                    platform_target=Platforms.TRN2,
                    collector=mock_collector,
                    backoff_seconds=0,
                )
            ) as host_generator:
                for host_attempt in host_generator:
                    with host_attempt as host:
                        host_id = host.get_host_id()
                        if first_host_id is None:
                            first_host_id = host_id
                            # Busy queue on the first host: transient, not a failure.
                            raise QueuePatienceRotation("queue ETA exceeds patience")
                        if host_id == first_host_id:
                            # Keep rotating until we land on a DIFFERENT host.
                            raise QueuePatienceRotation("queue ETA exceeds patience")
                        successful_host_id = host_id

        # The busy host was NEVER marked failed and stayed eligible; the loop
        # advanced to a different host and succeeded.
        assert mark_failed.call_count == 0
        assert mock_host_manager.failed_hosts == set()
        assert first_host_id is not None
        assert successful_host_id is not None
        assert successful_host_id != first_host_id

    def test_acquisition_runs_until_deadline_not_attempt_count(self, temp_dir):
        """Under sustained busy queues the loop rotates until a WALL-CLOCK deadline
        (not a fixed attempt count), never marks hosts failed, and finally raises a
        DISTINCT deadline diagnostic."""
        from test.utils.exceptions import QueuePatienceRotation

        manager = self._build_manager_with_hosts(temp_dir, num_hosts=2)
        mock_collector = MagicMock()

        clock = {"t": 1_700_000_000.0}

        def fake_time():
            return clock["t"]

        def fake_sleep(dt):
            clock["t"] += dt

        deadline_seconds = 100.0
        backoff = 5.0
        rotations = 0
        with (
            patch("test.utils.host_management.time.time", fake_time),
            patch("test.utils.host_management.time.sleep", fake_sleep),
        ):
            with pytest.raises(InferenceException) as exc_info:
                with closing(
                    manager.get_host_assignment_with_retry(
                        platform_target=Platforms.TRN2,
                        collector=mock_collector,
                        deadline_seconds=deadline_seconds,
                        backoff_seconds=backoff,
                    )
                ) as host_generator:
                    for host_attempt in host_generator:
                        with host_attempt as host:
                            rotations += 1
                            raise QueuePatienceRotation("queue busy")

        msg = str(exc_info.value)
        # Far more rotations than the old fixed cap of 3.
        assert rotations > 3
        assert rotations >= int(deadline_seconds / backoff) - 1
        # No host was ever marked failed for being merely busy.
        assert manager.failed_hosts == set()
        # The diagnostic is DISTINCT from the other two terminal messages.
        assert "deadline" in msg.lower()
        assert "No available hosts" not in msg
        assert "Connection error after" not in msg

    def test_connection_failures_still_capped_and_mark_failed(self, temp_dir):
        """Genuine connection failures still mark hosts failed and stop at the cap."""
        manager = self._build_manager_with_hosts(temp_dir, num_hosts=5)
        mock_collector = MagicMock()

        with patch.object(manager, "mark_host_as_failed", wraps=manager.mark_host_as_failed) as mark_failed:
            with pytest.raises(InferenceException) as exc_info:
                with closing(
                    manager.get_host_assignment_with_retry(
                        platform_target=Platforms.TRN2,
                        collector=mock_collector,
                        connection_failure_cap=3,
                        backoff_seconds=0,
                    )
                ) as host_generator:
                    for host_attempt in host_generator:
                        with host_attempt as host:
                            raise OSError("network unreachable")

        msg = str(exc_info.value)
        assert "Connection error after 3 attempts" in msg
        assert mark_failed.call_count == 3
        assert len(manager.failed_hosts) == 3

    def test_host_rotation_count_recorded_on_failure(self, mock_host_manager):
        """Each host rotation records CoreLockHostRotationCount as a delta of 1.0."""
        from test.utils.metrics_collector import MetricName

        mock_collector = MagicMock()
        successful_host_id = None
        iteration = 0
        with closing(
            mock_host_manager.get_host_assignment_with_retry(
                platform_target=Platforms.TRN2,
                collector=mock_collector,
                connection_failure_cap=2,
            )
        ) as host_generator:
            for host_attempt in host_generator:
                with host_attempt as host:
                    iteration += 1
                    if iteration == 1:
                        # Fail the first host, succeed on the next.
                        raise TimeoutError("SSH timed out")
                    successful_host_id = host.get_host_id()
        assert successful_host_id is not None

        rotation_calls = [
            c
            for c in mock_collector.record_metric.call_args_list
            if c.args[0] == MetricName.CORE_LOCK_HOST_ROTATION_COUNT
        ]
        assert len(rotation_calls) == 1
        assert rotation_calls[0].args[1] == 1.0

    def _build_manager_with_hosts(self, temp_dir, num_hosts):
        """Build a HostManager backed by ``num_hosts`` mocked TRN2 hosts."""
        aliases = [f"host{i}" for i in range(1, num_hosts + 1)]
        target_hosts = [TargetHost(ssh_host=a, host_type=Platforms.TRN2) for a in aliases]

        with patch("test.utils.host_management.SshHost"):
            manager = HostManager(
                base_host_info_path=temp_dir,
                target_hosts=target_hosts,
                neuron_installation_path="/opt/aws/neuron/bin",
                ssh_config_path="~/.ssh/config",
            )

        host_stats_path = os.path.join(temp_dir, "host_stats.json")
        with open(host_stats_path, "w") as f:
            json.dump(
                [
                    {"host_alias": a, "work_queue_depth": 0, "run_id": manager.run_id, "host_type": "trn2"}
                    for a in aliases
                ],
                f,
            )

        mocks = {}
        for a in aliases:
            m = MagicMock()
            m.get_host_id.return_value = a
            mocks[a] = m
        manager.target_hosts = mocks
        manager.host_types = {a: Platforms.TRN2 for a in aliases}
        return manager

    def test_host_rotation_count_datapoints_sum_to_n(self, temp_dir):
        """N rotations must emit N datapoints of 1.0 that sum to N (not N(N+1)/2)."""
        from test.utils.metrics_collector import MetricName

        n_rotations = 3
        # Need enough distinct hosts: each failure marks a host failed and
        # rotates to a fresh one, so num_hosts must exceed n_rotations.
        manager = self._build_manager_with_hosts(temp_dir, num_hosts=n_rotations + 1)

        mock_collector = MagicMock()
        successful_host_id = None
        iteration = 0
        with closing(
            manager.get_host_assignment_with_retry(
                platform_target=Platforms.TRN2,
                collector=mock_collector,
                connection_failure_cap=n_rotations + 1,
            )
        ) as host_generator:
            for host_attempt in host_generator:
                with host_attempt as host:
                    iteration += 1
                    if iteration <= n_rotations:
                        # Fail the first ``n_rotations`` hosts, succeed after.
                        raise TimeoutError("SSH timed out")
                    successful_host_id = host.get_host_id()
        assert successful_host_id is not None

        rotation_values = [
            c.args[1]
            for c in mock_collector.record_metric.call_args_list
            if c.args[0] == MetricName.CORE_LOCK_HOST_ROTATION_COUNT
        ]
        # One datapoint per rotation, each exactly 1.0 ...
        assert len(rotation_values) == n_rotations
        assert all(v == 1.0 for v in rotation_values)
        # ... so EMF sum/avg aggregations reflect the true rotation count.
        assert sum(rotation_values) == n_rotations
        # Guard against a regression to the cumulative-count emission.
        assert sum(rotation_values) != n_rotations * (n_rotations + 1) / 2


class _PatienceRotatingHost(Host):
    """Concrete ``Host`` whose ``get_core_allocation`` always raises
    ``QueuePatienceRotation``. Because the real ``Host.execute_command`` opens
    ``with self.get_core_allocation(...)`` with no try/except, the exception
    propagates out of ``execute_command`` -- exactly the production path that the
    retry loop's ``context_manager_wrapper`` must treat as transient.
    """

    def __init__(self, host_id: str):
        super().__init__()
        self._host_id = host_id

    def get_host_id(self) -> str:
        return self._host_id

    def get_core_allocation(
        self,
        collector,
        collective_ranks: int = 1,
        lnc_config: int = 2,
        timeout_seconds: int = 9000,
        poll_period_seconds: int = 0,
    ):
        from test.utils.exceptions import QueuePatienceRotation

        raise QueuePatienceRotation(f"queue ETA exceeds patience on {self._host_id}")

    # Remaining abstract methods are never reached in these tests.
    def _get_debug_output_dir(self, target_directory: str) -> str:
        return target_directory

    def _run_command(self, command, target_directory, neuron_env, collector) -> str:
        return ""

    def _collect_artifacts(self, target_directory, stdout, get_list_of_files_to_copy, collector) -> str:
        return target_directory

    def prepare_host(self, target_directory, collector, skip_remote_cleanup=False, force_local_cleanup=False):
        return contextlib.nullcontext()

    def _run_post_lock_command(self, command, target_directory, collector) -> str:
        return ""

    def get_neuron_device_info(self):
        return []


class TestHostRotationUntilDeadlineIntegration(TestHostManagerRetryErrorReporting):
    """Seam/regression test exercising the two combined behaviors TOGETHER through the
    real ``HostManager`` host-selection (``__get_host_assignment__`` + ``failed_hosts``)
    machinery -- only the wall clock and the per-attempt outcome are faked.

    1. ``QueuePatienceRotation`` (busy FIFO queue ETA exceeds patience) is transient: the
       host is NOT marked failed, NOT counted toward the connection-failure cap, the
       rotation metric is recorded, and rotation continues across hosts until a wall-clock
       deadline.
    2. Genuine connection failures still mark hosts failed and stop at
       ``connection_failure_cap``.
    """

    def _patched_clock(self):
        """Return (time_fn, sleep_fn) over an in-test fake monotonic clock."""
        clock = {"t": 1_700_000_000.0}

        def fake_time():
            return clock["t"]

        def fake_sleep(dt):
            clock["t"] += dt

        return fake_time, fake_sleep

    def test_a_sustained_busy_reselects_hosts_and_terminates_on_deadline(self, temp_dir):
        """(a) Every attempt is busy: the loop rotates far past ``num_hosts`` (re-selecting
        hosts that stayed eligible), never marks a host failed, never raises
        "No available hosts", and terminates with the DISTINCT deadline diagnostic."""
        from test.utils.exceptions import QueuePatienceRotation

        num_hosts = 3
        manager = self._build_manager_with_hosts(temp_dir, num_hosts=num_hosts)
        mock_collector = MagicMock()

        fake_time, fake_sleep = self._patched_clock()
        deadline_seconds = 100.0
        backoff = 5.0
        expected_min_rotations = int(deadline_seconds / backoff) - 1  # ~19

        attempted_hosts = []
        with (
            patch("test.utils.host_management.time.time", fake_time),
            patch("test.utils.host_management.time.sleep", fake_sleep),
        ):
            with pytest.raises(InferenceException) as exc_info:
                with closing(
                    manager.get_host_assignment_with_retry(
                        platform_target=Platforms.TRN2,
                        collector=mock_collector,
                        deadline_seconds=deadline_seconds,
                        backoff_seconds=backoff,
                    )
                ) as host_generator:
                    for host_attempt in host_generator:
                        with host_attempt as host:
                            attempted_hosts.append(host.get_host_id())
                            raise QueuePatienceRotation("queue ETA exceeds patience")

        rotations = len(attempted_hosts)
        # Rotated far more than the host count -> hosts were RE-selected, proving they
        # stayed eligible the whole time.
        assert rotations > num_hosts
        assert rotations >= expected_min_rotations
        # Some host must have been chosen more than once (re-selection).
        assert max(attempted_hosts.count(h) for h in set(attempted_hosts)) > 1
        # Only real hosts from the pool were ever selected.
        assert set(attempted_hosts).issubset(set(manager.target_hosts.keys()))
        # Merely-busy hosts were NEVER excluded.
        assert manager.failed_hosts == set()
        # Distinct deadline termination, not the other two terminal messages.
        msg = str(exc_info.value)
        assert "deadline" in msg.lower()
        assert "No available hosts" not in msg
        assert "Connection error after" not in msg

    def test_b_connection_failures_exclude_hosts_and_stop_at_cap(self, temp_dir):
        """(b) Genuine connection failures still exclude hosts (add to ``failed_hosts``) and
        stop at ``connection_failure_cap`` with the connection-error message."""
        connection_failure_cap = 3
        num_hosts = 3
        manager = self._build_manager_with_hosts(temp_dir, num_hosts=num_hosts)
        mock_collector = MagicMock()

        with patch.object(manager, "mark_host_as_failed", wraps=manager.mark_host_as_failed) as mark_failed:
            with pytest.raises(InferenceException) as exc_info:
                with closing(
                    manager.get_host_assignment_with_retry(
                        platform_target=Platforms.TRN2,
                        collector=mock_collector,
                        connection_failure_cap=connection_failure_cap,
                        backoff_seconds=0,
                    )
                ) as host_generator:
                    for host_attempt in host_generator:
                        with host_attempt as host:
                            raise OSError("network unreachable")

        msg = str(exc_info.value)
        assert f"Connection error after {connection_failure_cap} attempts" in msg
        assert "deadline" not in msg.lower()
        assert mark_failed.call_count == connection_failure_cap
        assert len(manager.failed_hosts) == connection_failure_cap
        assert manager.failed_hosts.issubset(set(manager.target_hosts.keys()))

    def test_c_rotation_metric_recorded_once_per_patience_rotation(self, temp_dir):
        """(c) In the sustained-busy scenario, ``CoreLockHostRotationCount`` is recorded
        exactly once per patience rotation, each datapoint == 1.0."""
        from test.utils.exceptions import QueuePatienceRotation
        from test.utils.metrics_collector import MetricName

        manager = self._build_manager_with_hosts(temp_dir, num_hosts=3)
        mock_collector = MagicMock()

        fake_time, fake_sleep = self._patched_clock()
        deadline_seconds = 100.0
        backoff = 5.0

        rotations = 0
        with (
            patch("test.utils.host_management.time.time", fake_time),
            patch("test.utils.host_management.time.sleep", fake_sleep),
        ):
            with pytest.raises(InferenceException):
                with closing(
                    manager.get_host_assignment_with_retry(
                        platform_target=Platforms.TRN2,
                        collector=mock_collector,
                        deadline_seconds=deadline_seconds,
                        backoff_seconds=backoff,
                    )
                ) as host_generator:
                    for host_attempt in host_generator:
                        with host_attempt as host:  # noqa: F841
                            rotations += 1
                            raise QueuePatienceRotation("queue ETA exceeds patience")

        rotation_values = [
            c.args[1]
            for c in mock_collector.record_metric.call_args_list
            if c.args[0] == MetricName.CORE_LOCK_HOST_ROTATION_COUNT
        ]
        # One datapoint per rotation, each exactly 1.0, and no failure metrics.
        assert len(rotation_values) == rotations
        assert all(v == 1.0 for v in rotation_values)
        assert manager.failed_hosts == set()

    def test_d_busy_host_is_reselectable_and_loop_can_succeed(self, temp_dir):
        """(d) A host that patience-rotates stays eligible; a later attempt succeeds and the
        context manager exits cleanly without any host being marked failed."""
        from test.utils.exceptions import QueuePatienceRotation

        manager = self._build_manager_with_hosts(temp_dir, num_hosts=3)
        mock_collector = MagicMock()

        attempted_hosts = []
        successful_host_id = None
        with patch.object(manager, "mark_host_as_failed", wraps=manager.mark_host_as_failed) as mark_failed:
            with closing(
                manager.get_host_assignment_with_retry(
                    platform_target=Platforms.TRN2,
                    collector=mock_collector,
                    backoff_seconds=0,
                )
            ) as host_generator:
                for host_attempt in host_generator:
                    with host_attempt as host:
                        host_id = host.get_host_id()
                        attempted_hosts.append(host_id)
                        if len(attempted_hosts) <= 2:
                            # Busy queue on the first couple of attempts: transient.
                            raise QueuePatienceRotation("queue ETA exceeds patience")
                        # Notify success simply by not raising.
                        successful_host_id = host_id

        assert successful_host_id is not None
        assert len(attempted_hosts) >= 3
        # No host was excluded for being merely busy; the loop ended on success.
        assert mark_failed.call_count == 0
        assert manager.failed_hosts == set()

    def test_e_patience_rotation_through_execute_command_is_transient(self, temp_dir):
        """Gap #1: drive ``QueuePatienceRotation`` THROUGH the real
        ``Host.execute_command``/``get_core_allocation`` (not raised directly in the
        consumer body). The first two attempts call ``execute_command`` -- whose
        ``get_core_allocation`` raises -- and the propagated exception must be treated
        as transient (no host marked failed); a later attempt then succeeds."""
        from test.utils.metrics_collector import MetricName

        num_hosts = 3
        manager = self._build_manager_with_hosts(temp_dir, num_hosts=num_hosts)
        # Replace the MagicMock hosts with concrete hosts that exercise the real
        # execute_command -> get_core_allocation propagation path.
        manager.target_hosts = {a: _PatienceRotatingHost(a) for a in manager.target_hosts}
        mock_collector = MagicMock()

        attempted_hosts = []
        successful_host_id = None
        with patch.object(manager, "mark_host_as_failed", wraps=manager.mark_host_as_failed) as mark_failed:
            with closing(
                manager.get_host_assignment_with_retry(
                    platform_target=Platforms.TRN2,
                    collector=mock_collector,
                    backoff_seconds=0,
                )
            ) as host_generator:
                for host_attempt in host_generator:
                    with host_attempt as host:
                        attempted_hosts.append(host.get_host_id())
                        if len(attempted_hosts) <= 2:
                            # Real production path: the exception originates inside
                            # get_core_allocation and propagates out of execute_command.
                            host.execute_command("cmd", "/tmp", mock_collector, 1, 2)
                        else:
                            successful_host_id = host.get_host_id()

        # The loop advanced past the transient rotations and succeeded.
        assert successful_host_id is not None
        assert len(attempted_hosts) >= 3
        # Transient: no host was marked failed and none excluded.
        assert mark_failed.call_count == 0
        assert manager.failed_hosts == set()
        # The two propagated rotations were recorded as rotation datapoints.
        rotation_values = [
            c.args[1]
            for c in mock_collector.record_metric.call_args_list
            if c.args[0] == MetricName.CORE_LOCK_HOST_ROTATION_COUNT
        ]
        assert len(rotation_values) == 2
        assert all(v == 1.0 for v in rotation_values)

    def test_f_mixed_patience_and_connection_failures_only_genuine_count_to_cap(self, temp_dir):
        """Gap #2: interleave patience rotations and genuine connection failures in one
        acquisition. Only genuine failures count toward ``connection_failure_cap`` and
        mark hosts failed, while the rotation metric is recorded for ALL non-success
        attempts; termination is the connection-error message, NOT the deadline."""
        from test.utils.exceptions import QueuePatienceRotation
        from test.utils.metrics_collector import MetricName

        connection_failure_cap = 3
        # Enough hosts so genuine-failure exclusions never exhaust the pool.
        manager = self._build_manager_with_hosts(temp_dir, num_hosts=6)
        mock_collector = MagicMock()

        # 2 patience rotations + 3 genuine failures (interleaved). Terminates on the
        # 3rd genuine failure (the cap), so all 5 outcomes are consumed.
        outcomes = ["patience", "oserror", "patience", "oserror", "oserror"]
        idx = {"i": 0}

        with patch.object(manager, "mark_host_as_failed", wraps=manager.mark_host_as_failed) as mark_failed:
            with pytest.raises(InferenceException) as exc_info:
                with closing(
                    manager.get_host_assignment_with_retry(
                        platform_target=Platforms.TRN2,
                        collector=mock_collector,
                        connection_failure_cap=connection_failure_cap,
                        backoff_seconds=0,
                    )
                ) as host_generator:
                    for host_attempt in host_generator:
                        with host_attempt as host:  # noqa: F841
                            kind = outcomes[idx["i"]]
                            idx["i"] += 1
                            if kind == "patience":
                                raise QueuePatienceRotation("queue ETA exceeds patience")
                            raise OSError("network unreachable")

        msg = str(exc_info.value)
        assert f"Connection error after {connection_failure_cap} attempts" in msg
        assert "deadline" not in msg.lower()
        # Only the 3 genuine failures marked hosts failed / counted toward the cap.
        assert mark_failed.call_count == connection_failure_cap
        assert len(manager.failed_hosts) == connection_failure_cap
        # All 5 non-success attempts (2 patience + 3 genuine) recorded a rotation datapoint.
        rotation_values = [
            c.args[1]
            for c in mock_collector.record_metric.call_args_list
            if c.args[0] == MetricName.CORE_LOCK_HOST_ROTATION_COUNT
        ]
        assert len(rotation_values) == len(outcomes)
        assert all(v == 1.0 for v in rotation_values)

    def test_g_patience_branch_sleeps_with_configured_backoff(self, temp_dir):
        """Gap #3: the patience branch must call ``time.sleep(backoff_seconds)`` with the
        configured backoff (guards against a busy-loop if the backoff is removed). Patch
        ``time.sleep`` with a mock and advance ``time.time`` independently so the loop
        still terminates at the deadline."""
        from test.utils.exceptions import QueuePatienceRotation

        manager = self._build_manager_with_hosts(temp_dir, num_hosts=3)
        mock_collector = MagicMock()

        backoff = 5.0
        deadline_seconds = 10.0
        # time.time advances on its own (independent of the mocked sleep) so the
        # wall-clock deadline is reached regardless of whether sleep blocks.
        base = 1_700_000_000.0
        ticks = iter([base + i * 3.0 for i in range(1000)])
        sleep_mock = MagicMock()

        with (
            patch("test.utils.host_management.time.time", lambda: next(ticks)),
            patch("test.utils.host_management.time.sleep", sleep_mock),
        ):
            with pytest.raises(InferenceException) as exc_info:
                with closing(
                    manager.get_host_assignment_with_retry(
                        platform_target=Platforms.TRN2,
                        collector=mock_collector,
                        deadline_seconds=deadline_seconds,
                        backoff_seconds=backoff,
                    )
                ) as host_generator:
                    for host_attempt in host_generator:
                        with host_attempt as host:  # noqa: F841
                            raise QueuePatienceRotation("queue ETA exceeds patience")

        assert "deadline" in str(exc_info.value).lower()
        # The patience branch backed off at least once, always with the configured value.
        assert sleep_mock.call_count > 0
        assert all(call.args == (backoff,) for call in sleep_mock.call_args_list)

    def test_h_soft_join_over_patience_rotates_before_upload(self, temp_dir):
        """Integration seam: a soft-join over-patience rotation must propagate out of the
        ``with host_attempt as host:`` body BEFORE ``prepare_host`` (the artifact upload)
        runs, rotate to another host WITHOUT marking any host failed, and be counted as a
        rotation. This mirrors the orchestrator body order
        (``soft_join_queue`` -> ``prepare_host``): if soft-join raises
        ``QueuePatienceRotation``, ``prepare_host`` must never be reached."""
        from test.utils.exceptions import QueuePatienceRotation
        from test.utils.metrics_collector import MetricName

        num_hosts = 2
        manager = self._build_manager_with_hosts(temp_dir, num_hosts=num_hosts)
        mock_collector = MagicMock()

        # Every host's soft_join_queue raises over-patience; prepare_host is a shared spy
        # that must NEVER be called on the rotation path (proves no upload happened).
        prepare_spy = MagicMock(return_value=contextlib.nullcontext())
        for host in manager.target_hosts.values():
            host.soft_join_queue.side_effect = QueuePatienceRotation("queue ETA exceeds patience")
            host.prepare_host = prepare_spy

        fake_time, fake_sleep = self._patched_clock()
        deadline_seconds = 50.0
        backoff = 5.0
        # ~9 rotations before the wall-clock deadline trips.
        expected_min_rotations = int(deadline_seconds / backoff) - 1

        attempted_hosts = []
        with (
            patch("test.utils.host_management.time.time", fake_time),
            patch("test.utils.host_management.time.sleep", fake_sleep),
        ):
            with pytest.raises(InferenceException) as exc_info:
                with closing(
                    manager.get_host_assignment_with_retry(
                        platform_target=Platforms.TRN2,
                        collector=mock_collector,
                        deadline_seconds=deadline_seconds,
                        backoff_seconds=backoff,
                    )
                ) as host_generator:
                    for host_attempt in host_generator:
                        with host_attempt as host:
                            attempted_hosts.append(host.get_host_id())
                            # Mirror the orchestrator body order exactly: soft-join the
                            # FIFO queue BEFORE uploading via prepare_host. soft_join
                            # raises over-patience, so prepare_host must be unreachable.
                            host.soft_join_queue(
                                collector=mock_collector,
                                collective_ranks=1,
                                lnc_config=2,
                            )
                            host.prepare_host(
                                target_directory="/tmp",
                                collector=mock_collector,
                            )

        # (1) The upload was NEVER attempted on the rotation path.
        prepare_spy.assert_not_called()
        # (2) An over-patience rotation is transient, not a host failure.
        assert manager.failed_hosts == set()
        # (3) The rotation metric was recorded exactly once per rotation (each value 1.0).
        rotation_values = [
            c.args[1]
            for c in mock_collector.record_metric.call_args_list
            if c.args[0] == MetricName.CORE_LOCK_HOST_ROTATION_COUNT
        ]
        rotations = len(attempted_hosts)
        assert rotations >= expected_min_rotations
        assert len(rotation_values) == rotations
        assert all(v == 1.0 for v in rotation_values)
        # (4) Only real pool hosts were attempted and at least one stayed eligible and was
        # re-selected (rotation kept hosts in the pool, no exclusion).
        assert set(attempted_hosts).issubset(set(manager.target_hosts.keys()))
        assert max(attempted_hosts.count(h) for h in set(attempted_hosts)) > 1
        # Distinct deadline termination (not "No available hosts" / connection error).
        msg = str(exc_info.value)
        assert "deadline" in msg.lower()
        assert "No available hosts" not in msg
        assert "Connection error after" not in msg


class TestTemporaryRandomSeed:
    """Tests for the temporary_random_seed context manager."""

    def setup_method(self) -> None:
        """Set a deterministic seed before each test."""
        random.seed(42)

    def teardown_method(self) -> None:
        """Reseed from system entropy so tests don't leak deterministic state."""
        random.seed()

    def test_restores_original_state(self) -> None:
        """RNG state before and after the context manager should be identical."""
        state_before = random.getstate()
        with temporary_random_seed(999):
            random.random()
        assert random.getstate() == state_before

    def test_choice_deterministic_with_same_seed(self) -> None:
        """Same temporary seed produces the same random.choice result."""
        items = ["host_a", "host_b", "host_c"]

        with temporary_random_seed(123):
            first = random.choice(items)

        with temporary_random_seed(123):
            second = random.choice(items)

        assert first == second

    def test_outer_sequence_unchanged_with_choice(self) -> None:
        """random.choice inside the context manager should not alter the outer RNG sequence."""
        expected_next = random.choice(["a", "b", "c"])

        random.seed(42)
        with temporary_random_seed(999):
            random.choice(["a", "b", "c"])
        assert random.choice(["a", "b", "c"]) == expected_next


SAMPLE_NEURON_LS_OUTPUT = json.dumps(
    [
        {
            "neuron_device": 0,
            "bdf": "00:1e.0",
            "cpu_affinity": "0-3",
            "numa_node": "0",
            "connected_to": None,
            "nc_count": 2,
            "memory_size": 34359738368,
            "neuroncore_ids": [0, 1],
            "neuron_processes": [],
        }
    ]
)


class TestDetectLocalNeuronDevices:
    """Tests for detect_local_neuron_devices()."""

    def setup_method(self):
        # Clear lru_cache between tests so each test gets a fresh call
        from test.utils.host_management import _run_neuron_ls

        _run_neuron_ls.cache_clear()

    @patch("test.utils.host_management.os.path.isfile", return_value=True)
    @patch("test.utils.host_management.subprocess.run")
    def test_returns_true_when_devices_found(self, mock_run, _mock_isfile):
        mock_run.return_value = MagicMock(returncode=0, stdout=SAMPLE_NEURON_LS_OUTPUT)
        assert detect_local_neuron_devices("/opt/aws/neuron/bin") is True

    @patch("test.utils.host_management.os.path.isfile", return_value=True)
    @patch("test.utils.host_management.subprocess.run")
    def test_returns_false_when_no_devices(self, mock_run, _mock_isfile):
        mock_run.return_value = MagicMock(returncode=0, stdout="[]")
        assert detect_local_neuron_devices("/opt/aws/neuron/bin") is False

    @patch("test.utils.host_management.os.path.isfile", return_value=True)
    @patch("test.utils.host_management.subprocess.run")
    def test_returns_false_when_neuron_ls_not_installed(self, mock_run, _mock_isfile):
        mock_run.side_effect = FileNotFoundError("neuron-ls not found")
        assert detect_local_neuron_devices("/opt/aws/neuron/bin") is False

    @patch("test.utils.host_management.os.path.isfile", return_value=True)
    @patch("test.utils.host_management.subprocess.run")
    def test_returns_false_on_timeout(self, mock_run, _mock_isfile):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="neuron-ls", timeout=10)
        assert detect_local_neuron_devices("/opt/aws/neuron/bin") is False

    @patch("test.utils.host_management.os.path.isfile", return_value=True)
    @patch("test.utils.host_management.subprocess.run")
    def test_returns_false_on_nonzero_exit(self, mock_run, _mock_isfile):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert detect_local_neuron_devices("/opt/aws/neuron/bin") is False

    def test_returns_false_when_binary_missing(self):
        """Binary check should short-circuit without spawning subprocess."""
        with patch("test.utils.host_management.os.path.isfile", return_value=False):
            assert detect_local_neuron_devices("/nonexistent/path") is False


def _make_mock_device(core_ids: list[int], lnc_config: int = 2) -> MagicMock:
    """Create a mock NeuronDeviceInfo with the given logical core IDs."""
    device = MagicMock()
    device.neuroncore_ids = core_ids
    device.nc_count = len(core_ids)
    device.logical_neuroncore_config = lnc_config
    return device


class TestLocalHostCoreAllocation:
    """Tests for LocalHost.get_core_allocation() with file-based locking."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def local_host(self, temp_dir):
        return LocalHost("/opt/aws/neuron/bin", "localhost", temp_dir)

    def test_allocates_correct_cores(self, local_host):
        """Core allocation returns the requested number of logical cores."""
        mock_collector = MagicMock()
        mock_collector.timer = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))
        )

        with patch.object(local_host, "get_neuron_device_info", return_value=[_make_mock_device([0, 1, 2, 3])]):
            with local_host.get_core_allocation(collector=mock_collector, collective_ranks=2, lnc_config=2) as alloc:
                assert len(alloc.logical_core_ids) == 2
                assert alloc.lnc_config == 2
                assert alloc.host_id == "localhost"

    def test_releases_cores_after_use(self, local_host):
        """Cores are released back to the pool after the context manager exits."""
        mock_collector = MagicMock()
        mock_collector.timer = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))
        )

        with patch.object(local_host, "get_neuron_device_info", return_value=[_make_mock_device([0, 1, 2, 3])]):
            with local_host.get_core_allocation(collector=mock_collector, collective_ranks=2, lnc_config=2):
                pass

            # After exiting, all physical cores should be free — allocate all 4 logical (8 physical)
            with local_host.get_core_allocation(collector=mock_collector, collective_ranks=4, lnc_config=2) as alloc:
                assert len(alloc.logical_core_ids) == 4

    def test_lnc1_and_lnc2_dont_overlap(self, local_host):
        """LNC1 and LNC2 allocations must not share physical cores."""
        mock_collector = MagicMock()
        mock_collector.timer = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))
        )

        # 4 logical cores at LNC2 = 8 physical cores total
        with patch.object(local_host, "get_neuron_device_info", return_value=[_make_mock_device([0, 1, 2, 3])]):
            # Allocate 1 logical core at LNC2 = 2 physical cores
            with local_host.get_core_allocation(
                collector=mock_collector, collective_ranks=1, lnc_config=2
            ) as alloc_lnc2:
                lnc2_physical = set(range(alloc_lnc2.logical_core_ids[0] * 2, alloc_lnc2.logical_core_ids[0] * 2 + 2))

                # Allocate 1 logical core at LNC1 = 1 physical core, must not overlap
                with local_host.get_core_allocation(
                    collector=mock_collector, collective_ranks=1, lnc_config=1
                ) as alloc_lnc1:
                    lnc1_physical = {alloc_lnc1.logical_core_ids[0]}
                    assert lnc2_physical.isdisjoint(lnc1_physical), (
                        f"Physical cores overlap: LNC2={lnc2_physical}, LNC1={lnc1_physical}"
                    )


class TestLocalHostExecuteCommand:
    """Tests for LocalHost.execute_command()."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def local_host(self, temp_dir):
        return LocalHost("/opt/aws/neuron/bin", "localhost", temp_dir)

    def _mock_collector(self):
        collector = MagicMock()
        collector.timer = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))
        )
        return collector

    @patch("test.utils.host_management.subprocess.run")
    def test_successful_execution_sets_env_vars(self, mock_run, local_host, temp_dir):
        """Verify env vars are set correctly on local execution."""
        mock_run.return_value = MagicMock(returncode=0, stdout="ok\n")
        collector = self._mock_collector()

        with patch.object(local_host, "get_neuron_device_info", return_value=[_make_mock_device([0, 1])]):
            local_host.execute_command(
                command="echo hello",
                target_directory=temp_dir,
                collector=collector,
                collective_ranks=1,
                lnc_config=2,
            )

        # Check subprocess was called with correct env vars
        call_kwargs = mock_run.call_args
        env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env")
        assert "NEURON_RT_VISIBLE_CORES" in env
        assert env["NEURON_LOGICAL_NC_CONFIG"] == "2"
        assert env["NEURON_RT_ENABLE_OCP"] == "1"

    @patch("test.utils.host_management.subprocess.run")
    def test_raises_on_nonzero_exit(self, mock_run, local_host, temp_dir):
        """Non-zero exit code raises LocalExecutionException."""
        mock_run.return_value = MagicMock(returncode=1, stdout="error output", stderr="some error")
        collector = self._mock_collector()

        with patch.object(local_host, "get_neuron_device_info", return_value=[_make_mock_device([0, 1])]):
            with pytest.raises(LocalExecutionException):
                local_host.execute_command(
                    command="false",
                    target_directory=temp_dir,
                    collector=collector,
                    collective_ranks=1,
                    lnc_config=2,
                )

    @patch("test.utils.host_management.subprocess.run")
    def test_artifact_collection(self, mock_run, local_host, temp_dir):
        """do_copy_artifacts moves files into infer_result/ subdirectory."""
        mock_run.return_value = MagicMock(returncode=0, stdout="output.npy\n")
        collector = self._mock_collector()

        # Create a file in target_directory that should be moved
        test_file = os.path.join(temp_dir, "output.npy")
        with open(test_file, "w") as f:
            f.write("data")

        with patch.object(local_host, "get_neuron_device_info", return_value=[_make_mock_device([0, 1])]):
            result_path = local_host.execute_command(
                command="echo output.npy",
                target_directory=temp_dir,
                collector=collector,
                collective_ranks=1,
                lnc_config=2,
                do_copy_artifacts=True,
                get_list_of_files_to_copy=lambda stdout: ["output.npy"],
            )

        assert result_path == os.path.join(temp_dir, "infer_result")
        assert os.path.exists(os.path.join(result_path, "output.npy"))


# =============================================================================
# SshHost.get_core_allocation poll-loop hardening + caller patience rotation.
#
# Covers: POLL_PERIOD as the poll-period default, the
# max_retryable_errors * (POLL_PERIOD + POLL_JITTER_MAX) <= STALE_THRESHOLD
# retry-budget invariant,
# the pure should_rotate predicate, QUEUED+ETA>patience -> dequeue + rotate,
# DRAINING stay-and-probe (no rotate), and dequeue-on-abandon (timeout/give-up).
# =============================================================================

import test.utils.host_management as host_management  # noqa: E402
from test.utils.core_lock_manager import (  # noqa: E402
    AllocationOutcome,
    AllocationStatus,
    CoreAllocation,
    CoreLockManager,
    LockAcquisitionError,
)
from test.utils.exceptions import QueuePatienceRotation, TimeoutException  # noqa: E402
from test.utils.host_management import (  # noqa: E402
    FAST_POLL_PERIOD,
    PATIENCE_SECONDS,
    POLL_JITTER_MAX,
    POLL_PERIOD,
    STALE_THRESHOLD,
    SshHost,
    select_poll_base,
    should_rotate,
)
from test.utils.metrics_collector import MetricName, NoopMetricsCollector  # noqa: E402
from test.utils.remote_executor import RemoteExecutorError  # noqa: E402
from test.utils.scripts import remote_lock_scripts  # noqa: E402

# A realistic unix epoch for the integration clock. Starting the fake clock here
# (rather than ~0) ensures the helper's RELATIVE worst_case_eta is exercised
# independent of the absolute wall clock: an absolute-timestamp ETA at this
# epoch (~1.7e9) would dwarf PATIENCE_SECONDS and force instant rotation.
_REALISTIC_EPOCH = 1_700_000_000.0


class _FakeClock:
    """Deterministic clock: time() returns the current value; sleep() advances it.

    Patched over the global ``time`` module so BOTH the poll loop
    (host_management) and the real lock helper (remote_lock_scripts) observe the
    same monotonic, sleep-driven time. Integration tests start it at a realistic
    epoch (``_REALISTIC_EPOCH``) rather than ~0: the helper now returns
    ``worst_case_eta`` as a RELATIVE wait (seconds from now), so the absolute
    clock value must be irrelevant to the PATIENCE_SECONDS threshold. Starting
    at a real epoch proves that (with an absolute-timestamp ETA every caller
    would instantly exceed patience and rotate).
    """

    def __init__(self, start: float = 0.0) -> None:
        self.t = float(start)

    def time(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt


class _RecordingClock(_FakeClock):
    """A _FakeClock that records every sleep duration in call order.

    Lets wiring tests assert the exact base+jitter the poll loop passes to
    ``time.sleep`` per iteration while still advancing the clock so the loop
    terminates deterministically.
    """

    def __init__(self, start: float = 0.0) -> None:
        super().__init__(start)
        self.sleeps: list[float] = []

    def sleep(self, dt: float) -> None:
        self.sleeps.append(dt)
        super().sleep(dt)


def _cm_collector() -> MagicMock:
    """A MagicMock collector whose timer() is a usable context manager."""
    collector = MagicMock()
    collector.test_name = "test-poll-loop"
    collector.timer = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False)))
    return collector


def _make_ssh_host(tmp_path, total_physical_cores: int = 8) -> SshHost:
    """Construct an SshHost without any real SSH; locking version/core count pre-seeded."""
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("")
    host = SshHost(
        ssh_alias="fakehost",
        test_base_path=str(tmp_path),
        remote_neuron_install_dir="/opt/aws/neuron/bin",
        run_id="run-1",
        ssh_config_path=str(ssh_config),
        s3_config=MagicMock(),
    )
    host._total_physical_cores = total_physical_cores
    host._host_locking_version = 3
    return host


class _FakeManager:
    """Stand-in for CoreLockManager that replays a scripted outcome sequence.

    Once the sequence is exhausted it repeats the final outcome, so the test is
    robust to the exact number of poll iterations the loop performs.
    """

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.acquire_calls = 0
        self.dequeue_calls = 0
        self.metrics_calls = 0
        self.released: list[list[int]] = []
        # Ordered log of terminal-teardown calls so tests can assert that the
        # abandon flush records metrics BEFORE the slot is freed.
        self.events: list[str] = []

    def acquire(self, num_logical_cores, lnc_config, timeout_seconds=None, ready=True):  # noqa: ARG002
        self.acquire_calls += 1
        item = self._outcomes[0] if len(self._outcomes) == 1 else self._outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def dequeue(self) -> None:
        self.dequeue_calls += 1
        self.events.append("dequeue")

    def release(self, core_ids) -> None:
        self.released.append(list(core_ids))

    def record_contention_metrics(self) -> None:
        self.metrics_calls += 1
        self.events.append("metrics")


class TestShouldRotate:
    """The pure caller-patience predicate."""

    def test_patience_seconds_is_180(self) -> None:
        assert PATIENCE_SECONDS == 180

    def test_rotate_when_eta_exceeds_patience_and_not_draining(self) -> None:
        assert should_rotate(PATIENCE_SECONDS + 1, draining=False) is True

    def test_no_rotate_when_draining_even_if_eta_large(self) -> None:
        # ETA is meaningless mid-drain; stay-and-probe, never rotate.
        assert should_rotate(PATIENCE_SECONDS + 1, draining=True) is False

    def test_no_rotate_when_eta_is_none(self) -> None:
        assert should_rotate(None, draining=False) is False

    def test_no_rotate_when_eta_within_patience(self) -> None:
        assert should_rotate(PATIENCE_SECONDS - 1, draining=False) is False
        assert should_rotate(PATIENCE_SECONDS, draining=False) is False


class TestSelectPollBase:
    """The pure poll-cadence selector for a queued caller."""

    def test_unknown_position_uses_slow(self) -> None:
        assert select_poll_base(None) == POLL_PERIOD

    def test_front_position_uses_fast(self) -> None:
        assert select_poll_base(0) == FAST_POLL_PERIOD

    def test_one_slot_from_front_uses_fast(self) -> None:
        assert select_poll_base(1) == FAST_POLL_PERIOD

    def test_position_two_uses_slow(self) -> None:
        assert select_poll_base(2) == POLL_PERIOD

    def test_large_position_uses_slow(self) -> None:
        assert select_poll_base(50) == POLL_PERIOD

    def test_negative_position_uses_fast(self) -> None:
        # The rule is literal `position <= 1`, so a negative position is fast.
        assert select_poll_base(-1) == FAST_POLL_PERIOD

    def test_injectable_fast_and_slow(self) -> None:
        assert select_poll_base(0, fast=2, slow=9) == 2
        assert select_poll_base(5, fast=2, slow=9) == 9


class TestPollLoopConstants:
    """POLL_PERIOD is the single source of truth for the poll-period default."""

    def test_fast_poll_period_is_one_and_below_slow(self) -> None:
        assert FAST_POLL_PERIOD == 1
        assert FAST_POLL_PERIOD < POLL_PERIOD

    def test_poll_period_default_reflects_constant(self) -> None:
        import inspect

        for cls in (host_management.Host, host_management.LocalHost, SshHost):
            sig = inspect.signature(cls.get_core_allocation)
            assert sig.parameters["poll_period_seconds"].default == POLL_PERIOD

    def test_retry_budget_fits_within_stale_threshold(self) -> None:
        # The loop tolerates 10 consecutive retryable errors spaced at
        # POLL_PERIOD + up to POLL_JITTER_MAX (plus RPC latency); that full
        # budget must not exceed STALE_THRESHOLD, with headroom over jitter.
        max_retryable_errors = 10
        assert max_retryable_errors * (POLL_PERIOD + POLL_JITTER_MAX) <= STALE_THRESHOLD


class TestSshHostPollLoop:
    """Unit tests for the SshHost.get_core_allocation poll loop with a faked manager."""

    @contextlib.contextmanager
    def _run(self, host, fake_mgr, clock=None, **kwargs):
        clock = clock or _FakeClock()
        with (
            patch.object(host_management, "CoreLockManager", return_value=fake_mgr),
            patch.object(host, "_get_remote_executor", return_value=MagicMock()),
            patch.object(host, "get_instance_type", return_value="trn1"),
            patch("time.sleep", clock.sleep),
            patch("time.time", clock.time),
        ):
            with host.get_core_allocation(collector=_cm_collector(), **kwargs) as alloc:
                yield alloc

    def test_draining_keeps_polling_then_allocates(self, tmp_path) -> None:
        """DRAINING does not abort, does not rotate, and is not a retryable error."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [
                # Large ETA while draining must NOT rotate (patience suppressed).
                AllocationOutcome(status=AllocationStatus.DRAINING, position=0, worst_case_eta=99999),
                AllocationOutcome(status=AllocationStatus.QUEUED, position=0, worst_case_eta=10),
                AllocationOutcome(
                    status=AllocationStatus.ALLOCATED,
                    logical_cores=[0, 1],
                    physical_cores=[0, 1, 2, 3],
                ),
            ]
        )
        with self._run(host, fake_mgr, collective_ranks=2, lnc_config=2) as alloc:
            assert isinstance(alloc, CoreAllocation)
            assert alloc.logical_core_ids == [0, 1]
        # Polled through DRAINING + QUEUED before ALLOCATED; never dequeued (committed).
        assert fake_mgr.acquire_calls == 3
        assert fake_mgr.dequeue_calls == 0
        assert fake_mgr.released == [[0, 1, 2, 3]]

    def test_queued_within_patience_keeps_polling_then_allocates(self, tmp_path) -> None:
        """QUEUED with ETA within patience must NOT rotate; it keeps polling."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [
                AllocationOutcome(status=AllocationStatus.QUEUED, position=1, worst_case_eta=PATIENCE_SECONDS),
                AllocationOutcome(
                    status=AllocationStatus.ALLOCATED,
                    logical_cores=[0],
                    physical_cores=[0, 1],
                ),
            ]
        )
        with self._run(host, fake_mgr, collective_ranks=1, lnc_config=2) as alloc:
            assert alloc.logical_core_ids == [0]
        assert fake_mgr.acquire_calls == 2
        assert fake_mgr.dequeue_calls == 0

    def test_poll_base_position_one_uses_fast_cadence(self, tmp_path) -> None:
        """Wiring: a near-front waiter (position=1) sleeps the FAST base + jitter."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [
                AllocationOutcome(status=AllocationStatus.QUEUED, position=1, worst_case_eta=10),
                AllocationOutcome(
                    status=AllocationStatus.ALLOCATED,
                    logical_cores=[0],
                    physical_cores=[0, 1],
                ),
            ]
        )
        clock = _RecordingClock()
        with patch("random.uniform", return_value=POLL_JITTER_MAX):
            with self._run(host, fake_mgr, clock=clock, collective_ranks=1, lnc_config=2):
                pass
        assert clock.sleeps[0] == FAST_POLL_PERIOD + POLL_JITTER_MAX

    def test_poll_base_position_two_uses_slow_cadence(self, tmp_path) -> None:
        """Wiring: a deeper waiter (position=2) sleeps the SLOW base + jitter."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [
                AllocationOutcome(status=AllocationStatus.QUEUED, position=2, worst_case_eta=10),
                AllocationOutcome(
                    status=AllocationStatus.ALLOCATED,
                    logical_cores=[0],
                    physical_cores=[0, 1],
                ),
            ]
        )
        clock = _RecordingClock()
        with patch("random.uniform", return_value=POLL_JITTER_MAX):
            with self._run(host, fake_mgr, clock=clock, collective_ranks=1, lnc_config=2):
                pass
        assert clock.sleeps[0] == POLL_PERIOD + POLL_JITTER_MAX

    def test_poll_base_draining_front_uses_fast_cadence_no_status_gating(self, tmp_path) -> None:
        """Wiring: a DRAINING front waiter (position=0) still fast-polls.

        Proves the fast cadence is driven purely by position and is NOT
        suppressed while the host is draining (no status gating).
        """
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [
                AllocationOutcome(status=AllocationStatus.DRAINING, position=0, worst_case_eta=10),
                AllocationOutcome(
                    status=AllocationStatus.ALLOCATED,
                    logical_cores=[0],
                    physical_cores=[0, 1],
                ),
            ]
        )
        clock = _RecordingClock()
        with patch("random.uniform", return_value=POLL_JITTER_MAX):
            with self._run(host, fake_mgr, clock=clock, collective_ranks=1, lnc_config=2):
                pass
        assert clock.sleeps[0] == FAST_POLL_PERIOD + POLL_JITTER_MAX

    def test_poll_base_honors_caller_slow_period(self, tmp_path) -> None:
        """Wiring: a deep waiter's slow base honors the caller's poll_period_seconds,
        proving slow=poll_period_seconds is threaded into select_poll_base."""
        host = _make_ssh_host(tmp_path)
        custom_slow = POLL_PERIOD + 7
        fake_mgr = _FakeManager(
            [
                AllocationOutcome(status=AllocationStatus.QUEUED, position=5, worst_case_eta=10),
                AllocationOutcome(
                    status=AllocationStatus.ALLOCATED,
                    logical_cores=[0],
                    physical_cores=[0, 1],
                ),
            ]
        )
        clock = _RecordingClock()
        with patch("random.uniform", return_value=POLL_JITTER_MAX):
            with self._run(
                host,
                fake_mgr,
                clock=clock,
                collective_ranks=1,
                lnc_config=2,
                poll_period_seconds=custom_slow,
            ):
                pass
        assert clock.sleeps[0] == custom_slow + POLL_JITTER_MAX

    def test_stale_outcome_does_not_fast_spin_on_retryable_error(self, tmp_path) -> None:
        """Regression: a successful near-front poll (QUEUED position=0) must NOT leave a
        stale near-front position that makes the NEXT (retryable-error) iteration
        fast-spin. The successful poll fast-polls; the post-error poll must use the
        SLOW cadence (an error is not evidence the caller is near the front).

        Before the fix the except branch did not reset ``outcome``, so sleeps[1]
        would be FAST+jitter, burning the 10-error budget ~3.6x faster.
        """
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [
                AllocationOutcome(status=AllocationStatus.QUEUED, position=0, worst_case_eta=10),
                LockAcquisitionError("transient"),
                AllocationOutcome(
                    status=AllocationStatus.ALLOCATED,
                    logical_cores=[0],
                    physical_cores=[0, 1],
                ),
            ]
        )
        clock = _RecordingClock()
        with patch("random.uniform", return_value=POLL_JITTER_MAX):
            with self._run(host, fake_mgr, clock=clock, collective_ranks=1, lnc_config=2):
                pass
        # Successful near-front poll fast-polls (unchanged)...
        assert clock.sleeps[0] == FAST_POLL_PERIOD + POLL_JITTER_MAX
        # ...but the post-error iteration falls back to the SLOW cadence.
        assert clock.sleeps[1] == POLL_PERIOD + POLL_JITTER_MAX

    def test_none_position_routes_to_slow_through_loop(self, tmp_path) -> None:
        """Wiring: a QUEUED outcome with an unknown position (None) sleeps the SLOW
        base + jitter through the real loop (not just at the pure-helper level)."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [
                AllocationOutcome(status=AllocationStatus.QUEUED, position=None, worst_case_eta=10),
                AllocationOutcome(
                    status=AllocationStatus.ALLOCATED,
                    logical_cores=[0],
                    physical_cores=[0, 1],
                ),
            ]
        )
        clock = _RecordingClock()
        with patch("random.uniform", return_value=POLL_JITTER_MAX):
            with self._run(host, fake_mgr, clock=clock, collective_ranks=1, lnc_config=2):
                pass
        assert clock.sleeps[0] == POLL_PERIOD + POLL_JITTER_MAX

    def test_dequeue_on_retryable_giveup(self, tmp_path) -> None:
        """After max_retryable_errors consecutive retryable errors, give up + dequeue."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager([LockAcquisitionError("transient") for _ in range(10)])
        with pytest.raises(OSError):
            with self._run(host, fake_mgr, collective_ranks=1, lnc_config=2):
                pass
        assert fake_mgr.acquire_calls == 10
        assert fake_mgr.dequeue_calls == 1
        assert fake_mgr.released == []  # never committed

    def test_full_retry_budget_elapses_within_stale_threshold(self, tmp_path) -> None:
        """The full retry budget (10 retryable errors spaced at the REAL max gap of
        POLL_PERIOD + POLL_JITTER_MAX + RPC latency) elapses strictly inside
        STALE_THRESHOLD, so an actively-retrying caller refreshes last_seen_ts
        before the prune could ever fire (F5 liveness guarantee)."""
        host = _make_ssh_host(tmp_path)
        clock = _FakeClock(start=0.0)
        rpc_latency = 0.3

        class _SlowFailingManager(_FakeManager):
            def acquire(self, *a, **k):
                clock.sleep(rpc_latency)  # simulate the RPC round-trip per poll
                return super().acquire(*a, **k)

        fake_mgr = _SlowFailingManager([LockAcquisitionError("transient") for _ in range(10)])
        # Force jitter to its ceiling so every inter-poll gap is the worst case.
        with patch("random.uniform", return_value=POLL_JITTER_MAX):
            with pytest.raises(OSError):
                with self._run(host, fake_mgr, clock=clock, collective_ranks=1, lnc_config=2):
                    pass
        assert fake_mgr.acquire_calls == 10
        assert fake_mgr.dequeue_calls == 1
        # Even at max jitter + RPC latency, the whole budget fits with headroom.
        assert clock.time() < STALE_THRESHOLD

    def test_dequeue_on_timeout(self, tmp_path) -> None:
        """A wall-clock timeout while still QUEUED (within patience) dequeues + raises."""
        host = _make_ssh_host(tmp_path)
        # ETA within patience so the loop does NOT rotate; it polls until the
        # injected clock advances past the timeout budget, then dequeues.
        fake_mgr = _FakeManager([AllocationOutcome(status=AllocationStatus.QUEUED, position=2, worst_case_eta=10)])
        with pytest.raises(TimeoutException, match="Unable to allocate"):
            with self._run(host, fake_mgr, collective_ranks=1, lnc_config=2, timeout_seconds=30):
                pass
        assert fake_mgr.acquire_calls >= 2  # polled repeatedly before timing out
        assert fake_mgr.dequeue_calls == 1
        assert fake_mgr.released == []  # never committed

    def test_no_dequeue_after_successful_commit(self, tmp_path) -> None:
        """A fast ALLOCATED must not trigger dequeue; only release on context exit."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [
                AllocationOutcome(
                    status=AllocationStatus.ALLOCATED,
                    logical_cores=[0],
                    physical_cores=[0, 1],
                ),
            ]
        )
        with self._run(host, fake_mgr, collective_ranks=1, lnc_config=2) as alloc:
            assert alloc.logical_core_ids == [0]
        assert fake_mgr.dequeue_calls == 0
        assert fake_mgr.released == [[0, 1]]

    def test_over_patience_queued_keeps_polling_and_does_not_rotate(self, tmp_path) -> None:
        """Post-commit (get_core_allocation) no longer rotates on an over-patience
        QUEUED outcome: the gate moved to soft_join_queue (pre-upload). Once
        committed we keep polling until ALLOCATED with NO QueuePatienceRotation.
        The success path flushes contention metrics exactly once (not as an
        abandon) and never dequeues."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [
                AllocationOutcome(status=AllocationStatus.QUEUED, position=5, worst_case_eta=PATIENCE_SECONDS + 1),
                AllocationOutcome(status=AllocationStatus.ALLOCATED, logical_cores=[0], physical_cores=[0, 1]),
            ]
        )
        with self._run(host, fake_mgr, collective_ranks=1, lnc_config=2) as alloc:
            assert alloc.logical_core_ids == [0]
        # Kept polling through the over-patience QUEUED, then committed: no rotation,
        # no dequeue, metrics flushed exactly once on success (not as an abandon).
        assert fake_mgr.acquire_calls == 2
        assert fake_mgr.dequeue_calls == 0
        assert fake_mgr.events == ["metrics"]
        assert fake_mgr.metrics_calls == 1

    def test_retryable_giveup_and_timeout_abandons_flush_metrics(self, tmp_path) -> None:
        """F11: the retryable give-up and wall-clock timeout abandon paths also flush
        contention metrics before dequeuing."""
        # Retryable give-up.
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager([LockAcquisitionError("transient") for _ in range(10)])
        with pytest.raises(OSError):
            with self._run(host, fake_mgr, collective_ranks=1, lnc_config=2):
                pass
        assert fake_mgr.events == ["metrics", "dequeue"]
        assert fake_mgr.metrics_calls == 1

        # Wall-clock timeout (within-patience QUEUED that never gets cores).
        host2 = _make_ssh_host(tmp_path)
        fake_mgr2 = _FakeManager([AllocationOutcome(status=AllocationStatus.QUEUED, position=2, worst_case_eta=10)])
        with pytest.raises(TimeoutException, match="Unable to allocate"):
            with self._run(host2, fake_mgr2, collective_ranks=1, lnc_config=2, timeout_seconds=30):
                pass
        assert fake_mgr2.events == ["metrics", "dequeue"]
        assert fake_mgr2.metrics_calls == 1

    def test_success_flushes_metrics_once_without_dequeue(self, tmp_path) -> None:
        """The ALLOCATED success path still flushes metrics exactly once and never
        dequeues (success and abandon are mutually exclusive)."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [AllocationOutcome(status=AllocationStatus.ALLOCATED, logical_cores=[0], physical_cores=[0, 1])]
        )
        with self._run(host, fake_mgr, collective_ranks=1, lnc_config=2) as alloc:
            assert alloc.logical_core_ids == [0]
        assert fake_mgr.events == ["metrics"]
        assert fake_mgr.metrics_calls == 1
        assert fake_mgr.dequeue_calls == 0


class _HelperExecutor:
    """Fake RemoteExecutor running the real lock helper against a temp locks.json.

    Mirrors RemoteExecutor.call_function: redirects helper/lock paths and the
    remote locks.json path (args[0]) to local temp files, then runs the real
    helper verb so the deployed helper module is genuinely exercised. An optional
    on_call hook can mutate the locks.json between polls.
    """

    def __init__(self, helpers_file, lock_file, locks_json, on_call=None) -> None:
        self._helpers_file = helpers_file
        self._lock_file = lock_file
        self._locks_json = locks_json
        self._on_call = on_call
        self.calls = 0

    def call_function(self, func, **kwargs):
        self.calls += 1
        if self._on_call is not None:
            self._on_call(self.calls, self._locks_json)
        kwargs = dict(kwargs)
        kwargs["helpers_file"] = self._helpers_file
        kwargs["lock_file"] = self._lock_file
        args = list(kwargs["args"])
        args[0] = self._locks_json
        kwargs["args"] = args
        return func(**kwargs)


class TestSshHostPollLoopIntegration:
    """No-hardware integration: real helper + temp locks.json through get_core_allocation.

    A shared _FakeClock is patched over the global ``time`` module so both the
    poll loop and the real helper share one sleep-driven clock. It starts at a
    realistic epoch (``_REALISTIC_EPOCH``): the helper returns ``worst_case_eta``
    as a RELATIVE wait, so lock expiries are written epoch-relative and the
    PATIENCE_SECONDS threshold is exercised independent of the absolute clock.
    """

    def _paths(self, tmp_path):
        return (
            str(remote_lock_scripts.__file__),
            str(tmp_path / "atomic_lock"),
            str(tmp_path / "locks.json"),
        )

    def _write_locks(self, locks_json, locked_until=None, draining_until=None, ncores=8):
        import json as _json

        state = {"version": 3, "physical_neuron_cores_lock_timeout": {}}
        if locked_until is not None:
            state["physical_neuron_cores_lock_timeout"] = {str(i): locked_until for i in range(ncores)}
        if draining_until is not None:
            state["draining_enabled_with_timeout"] = draining_until
        with open(locks_json, "w") as f:
            _json.dump(state, f)

    def _read_queue(self, locks_json):
        import json as _json

        with open(locks_json) as f:
            return _json.load(f).get("queue", [])

    def test_enqueues_polls_then_yields_once_cores_free(self, tmp_path) -> None:
        """Within-patience QUEUED caller stays, then commits once cores free."""
        import json as _json

        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        # Cores busy only briefly (ETA ~100s <= patience) so the caller does not
        # rotate; freed before the second poll so the head commits. locked_until
        # is epoch-relative so the helper's RELATIVE ETA is ~100s regardless of
        # the (realistic) absolute wall clock.
        self._write_locks(locks_json, locked_until=int(_REALISTIC_EPOCH) + 100)

        def free_cores_before_second_poll(call_n, path):
            if call_n == 2:
                with open(path) as f:
                    state = _json.load(f)
                state["physical_neuron_cores_lock_timeout"] = {}
                with open(path, "w") as f:
                    _json.dump(state, f)

        executor = _HelperExecutor(helpers_file, lock_file, locks_json, on_call=free_cores_before_second_poll)
        host = _make_ssh_host(tmp_path)

        with (
            patch.object(host, "_get_remote_executor", return_value=executor),
            patch.object(host, "get_instance_type", return_value="trn1"),
            patch("time.sleep", clock.sleep),
            patch("time.time", clock.time),
        ):
            with host.get_core_allocation(
                collector=_cm_collector(), collective_ranks=2, lnc_config=2, timeout_seconds=300
            ) as alloc:
                assert isinstance(alloc, CoreAllocation)
                assert len(alloc.logical_core_ids) == 2

        assert executor.calls >= 2  # enqueued, polled, then committed
        assert self._read_queue(locks_json) == []  # committed + released -> empty

    def test_eta_over_patience_rotates_and_dequeues(self, tmp_path) -> None:
        """A deep queue pushes the soft-join caller's ETA past patience -> the
        pre-upload patience gate in soft_join_queue dequeues + raises
        QueuePatienceRotation (rotation now originates pre-upload, not in
        get_core_allocation)."""
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        # All cores busy for a full hold window; helper uses the shared clock.
        self._write_locks(locks_json, locked_until=int(_REALISTIC_EPOCH) + 60)
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        with patch("time.time", clock.time), patch("time.sleep", clock.sleep):
            # Seed three full-window waiters AHEAD of our caller so the forward
            # simulation stacks 3 x 60s -> our ETA (~240s) exceeds patience.
            ahead = [
                CoreLockManager(
                    "fakehost",
                    total_physical_cores=8,
                    collector=NoopMetricsCollector(),
                    executor=executor,
                    host_locking_version=3,
                )
                for _ in range(3)
            ]
            for mgr in ahead:
                assert mgr.acquire(num_logical_cores=4, lnc_config=2).status == AllocationStatus.QUEUED
            assert len(self._read_queue(locks_json)) == 3

            host = _make_ssh_host(tmp_path)
            with (
                patch.object(host, "_get_remote_executor", return_value=executor),
                patch.object(host, "get_instance_type", return_value="trn1"),
            ):
                with pytest.raises(QueuePatienceRotation, match="rotating host"):
                    host.soft_join_queue(collector=_cm_collector(), collective_ranks=4, lnc_config=2)

        # Our caller soft-joined (4 entries) then dequeued on the pre-upload
        # rotation -> only the three seeded waiters remain; the abandoned slot is
        # gone (queue returns to its seeded waiter count).
        assert len(self._read_queue(locks_json)) == 3

    def test_stale_entry_pruned_then_reenqueued_records_reenqueue_not_bump(self, tmp_path) -> None:
        """F14 (no-hardware integration): an entry whose last_seen_ts goes stale (no
        poll for > STALE_THRESHOLD) is pruned by reconcile and transparently
        re-enqueued at the tail on its next poll with re_enqueued=True. The manager
        records a non-zero reenqueue count and a ZERO bump count -- proving a
        prune->re-enqueue is observable and distinct from a bump."""
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        # All cores busy so the sole caller queues (no fast-path / commit) until
        # we free them, keeping it alive in the queue across the stale gap.
        self._write_locks(locks_json, locked_until=int(_REALISTIC_EPOCH) + 100000)
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        with patch("time.time", clock.time), patch("time.sleep", clock.sleep):
            spy = _cm_collector()
            mgr = CoreLockManager(
                "fakehost",
                total_physical_cores=8,
                collector=spy,
                executor=executor,
                host_locking_version=3,
            )
            # Poll #1: enqueue at tail (first-ever -> not a re-enqueue).
            assert mgr.acquire(num_logical_cores=4, lnc_config=2).status == AllocationStatus.QUEUED
            assert mgr._reenqueue_count == 0
            assert len(self._read_queue(locks_json)) == 1

            # Let last_seen_ts go stale past STALE_THRESHOLD (a long upload).
            clock.sleep(STALE_THRESHOLD + POLL_PERIOD)

            # Poll #2: reconcile prunes the stale entry, then this poll re-enqueues
            # it at the tail with re_enqueued=True.
            assert mgr.acquire(num_logical_cores=4, lnc_config=2).status == AllocationStatus.QUEUED
            assert mgr._reenqueue_count == 1
            assert mgr._bump_count == 0

            # Free the cores; the caller re-enqueues and commits on a later poll
            # (the queue is rebuilt fresh), flushing the per-attempt metrics on commit.
            self._write_locks(locks_json, locked_until=None)
            final = None
            for _ in range(5):
                final = mgr.acquire(num_logical_cores=4, lnc_config=2)
                if final.status == AllocationStatus.ALLOCATED:
                    break
            assert final is not None and final.status == AllocationStatus.ALLOCATED

        reenqueue = [c for c in spy.record_metric.call_args_list if c.args[0] == MetricName.CORE_LOCK_REENQUEUE_COUNT]
        assert reenqueue and reenqueue[-1].args[1] == 1
        bump = [c for c in spy.record_metric.call_args_list if c.args[0] == MetricName.CORE_LOCK_BUMP_COUNT]
        assert bump and bump[-1].args[1] == 0

    def test_draining_stays_and_probes_no_rotate(self, tmp_path) -> None:
        """A draining host is never rotated; the caller stays and keeps probing."""
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        # Drain far beyond the run; cores all busy so no fast-path either.
        self._write_locks(
            locks_json,
            locked_until=int(_REALISTIC_EPOCH) + 100000,
            draining_until=int(_REALISTIC_EPOCH) + 100000,
        )
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)
        host = _make_ssh_host(tmp_path)

        with (
            patch.object(host, "_get_remote_executor", return_value=executor),
            patch.object(host, "get_instance_type", return_value="trn1"),
            patch("time.sleep", clock.sleep),
            patch("time.time", clock.time),
        ):
            with pytest.raises(TimeoutException, match="Unable to allocate"):
                with host.get_core_allocation(
                    collector=_cm_collector(), collective_ranks=4, lnc_config=2, timeout_seconds=30
                ):
                    pass

        # Stayed and probed repeatedly (not a single-shot rotation), then the
        # wall-clock timeout dequeued the slot.
        assert executor.calls >= 3
        assert self._read_queue(locks_json) == []

    def test_timeout_calls_dequeue_and_clears_queue(self, tmp_path) -> None:
        """Within-patience QUEUED caller that never gets cores times out + dequeues."""
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        # Cores free at 120s (ETA ~120 <= patience, so no rotate) but the run
        # times out at 30s, so the caller never commits.
        self._write_locks(locks_json, locked_until=int(_REALISTIC_EPOCH) + 120)
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)
        host = _make_ssh_host(tmp_path)

        with (
            patch.object(host, "_get_remote_executor", return_value=executor),
            patch.object(host, "get_instance_type", return_value="trn1"),
            patch("time.sleep", clock.sleep),
            patch("time.time", clock.time),
        ):
            with pytest.raises(TimeoutException, match="Unable to allocate"):
                with host.get_core_allocation(
                    collector=_cm_collector(), collective_ranks=2, lnc_config=2, timeout_seconds=30
                ):
                    pass

        # The poll enqueued us; the timeout path must dequeue so the entry is gone.
        assert self._read_queue(locks_json) == []

    def test_retrying_caller_within_budget_not_pruned_keeps_position(self, tmp_path) -> None:
        """A caller that misses several polls within the retry budget (its RPCs fail
        at the max gap, so last_seen_ts never refreshes) is NOT pruned by a
        concurrent reconcile and keeps its FIFO position behind the head (F5)."""
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        # All cores busy so both callers queue rather than committing.
        self._write_locks(locks_json, locked_until=int(_REALISTIC_EPOCH) + 100000)
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        with patch("time.time", clock.time), patch("time.sleep", clock.sleep):
            head = CoreLockManager(
                "fakehost",
                total_physical_cores=8,
                collector=NoopMetricsCollector(),
                executor=executor,
                host_locking_version=3,
            )
            ours = CoreLockManager(
                "fakehost",
                total_physical_cores=8,
                collector=NoopMetricsCollector(),
                executor=executor,
                host_locking_version=3,
            )
            assert head.acquire(num_logical_cores=4, lnc_config=2).status == AllocationStatus.QUEUED
            assert ours.acquire(num_logical_cores=4, lnc_config=2).status == AllocationStatus.QUEUED
            # FIFO: head enqueued first, ours behind it.
            assert [e["entry_id"] for e in self._read_queue(locks_json)] == [head.entry_id, ours.entry_id]

            # Simulate our caller missing its whole retry budget: advance the clock
            # by 10 errors at the REAL max spacing (POLL_PERIOD + POLL_JITTER_MAX)
            # WITHOUT our manager refreshing last_seen_ts. Stay strictly inside
            # STALE_THRESHOLD -- the prune must not fire for us mid-budget.
            max_retryable_errors = 10
            budget = max_retryable_errors * (POLL_PERIOD + POLL_JITTER_MAX)
            assert budget < STALE_THRESHOLD
            clock.t += budget

            # The head polls again -> runs reconcile/prune over the temp locks.json.
            head.acquire(num_logical_cores=4, lnc_config=2)

        # Our entry survived the reconcile and kept its FIFO position behind head.
        assert [e["entry_id"] for e in self._read_queue(locks_json)] == [head.entry_id, ours.entry_id]

    def test_transient_executor_reset_recovers_without_failing_host(self, tmp_path) -> None:
        """Poll-loop tolerance contract: a SINGLE transient executor error during
        acquisition (the kind a channel reset produces) surfaces as a retryable
        LockAcquisitionError; the loop retries, the next acquire succeeds, the
        attempt completes ALLOCATED, the host is NOT marked failed, and only a tiny
        slice of the 10-error retry budget is consumed.

        Scope note: this pins the poll-loop/retry-budget integration, not the
        executor's channel rebuild itself. The transient error is injected via a
        fake executor; the production RemoteExecutor self-heal (a nulled channel is
        rebuilt on the next call) is guarded directly at the unit level by
        test_remote_executor.test_call_reconnects_after_channel_crash. Together they
        cover the bench root cause: before the self-heal a reset left the cached
        executor permanently closed, so EVERY subsequent acquire raised and burned
        all 10 retries -> OSError -> failed host.
        """
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _RecordingClock(start=_REALISTIC_EPOCH)
        # Cores free from the start so a healthy acquire commits immediately; the
        # only thing standing between the caller and ALLOCATED is the transient
        # reset on the first executor use.
        self._write_locks(locks_json, locked_until=None)
        real_executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        class _TransientResetExecutor:
            """Raises a channel-reset RemoteExecutorError on its FIRST call (a transient
            reset), then self-heals: every later call delegates to the real helper
            executor, exactly as RemoteExecutor.call() rebuilds a crashed channel on
            demand before serving the next request."""

            def __init__(self, inner) -> None:
                self._inner = inner
                self.calls = 0
                self.raises = 0

            def call_function(self, func, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    self.raises += 1
                    raise RemoteExecutorError("Executor is closed")
                return self._inner.call_function(func, **kwargs)

        executor = _TransientResetExecutor(real_executor)
        host = _make_ssh_host(tmp_path)

        with (
            patch.object(host, "_get_remote_executor", return_value=executor),
            patch.object(host, "get_instance_type", return_value="trn1"),
            patch("time.sleep", clock.sleep),
            patch("time.time", clock.time),
        ):
            # No OSError / TimeoutException escapes -> the host is never marked
            # failed by context_manager_wrapper; acquisition SUCCEEDS post-reset.
            with host.get_core_allocation(
                collector=_cm_collector(), collective_ranks=2, lnc_config=2, timeout_seconds=300
            ) as alloc:
                assert isinstance(alloc, CoreAllocation)
                assert len(alloc.logical_core_ids) == 2

        # The transient reset was injected exactly once.
        assert executor.raises == 1
        # The poll loop absorbed that single transient error with only a couple of
        # inter-poll sleeps (the recovered acquire enqueues then commits) -- nowhere
        # near the max_retryable_errors budget of 10. Had the channel stayed dead
        # (OLD behavior) every acquire would raise, forcing 10 retries -> OSError and
        # a failed host; reaching ALLOCATED at all proves recovery.
        assert len(clock.sleeps) < 10
        # Committed + released -> a clean FIFO slot (no abandoned dequeue).
        assert self._read_queue(locks_json) == []


# =============================================================================
# Soft-join during artifact upload.
#
# Covers: SshHost.soft_join_queue enqueues via acquire(ready=False) and caches
# the manager; get_core_allocation reuses the SAME cached manager (same
# entry_id, no second construction); soft_join_queue swallows errors
# (best-effort); base/LocalHost soft_join_queue is a no-op; and an end-to-end,
# no-hardware integration proving the upload-window slot reservation preserves
# FIFO ordering against a real locks.json + helper.
# =============================================================================


class _RecordingManager:
    """Fake CoreLockManager that records acquire(ready=...) calls and replays outcomes."""

    def __init__(self, entry_id: str = "entry-abc") -> None:
        self.entry_id = entry_id
        # Mirrors CoreLockManager.collector so SshHost._ensure_core_lock_manager's
        # collector-identity reuse guard can be exercised with this fake.
        self.collector = None
        self.acquire_ready_args: list[bool] = []
        self._seq: list = []
        self.dequeue_calls = 0
        self.released: list[list[int]] = []

    def acquire(self, num_logical_cores, lnc_config, timeout_seconds=None, ready=True):  # noqa: ARG002
        self.acquire_ready_args.append(ready)
        if self._seq:
            item = self._seq.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return AllocationOutcome(status=AllocationStatus.QUEUED, position=0, worst_case_eta=10)

    def record_contention_metrics(self) -> None:
        pass

    def release(self, core_ids) -> None:
        self.released.append(list(core_ids))

    def dequeue(self) -> None:
        self.dequeue_calls += 1


class TestSoftJoinQueue:
    """SshHost.soft_join_queue + manager caching."""

    def test_soft_join_enqueues_ready_false_and_caches_manager(self, tmp_path) -> None:
        host = _make_ssh_host(tmp_path)
        rec = _RecordingManager()
        ctor = MagicMock(return_value=rec)
        with (
            patch.object(host_management, "CoreLockManager", ctor),
            patch.object(host, "_get_remote_executor", return_value=MagicMock()),
            patch.object(host, "get_instance_type", return_value="trn1"),
        ):
            host.soft_join_queue(collector=_cm_collector(), collective_ranks=2, lnc_config=2)

        # Exactly one ready=False enqueue; manager cached on the host.
        assert rec.acquire_ready_args == [False]
        assert ctor.call_count == 1
        assert host._core_lock_manager is rec

    def test_get_core_allocation_reuses_soft_joined_manager(self, tmp_path) -> None:
        """get_core_allocation must reuse the cached manager (same entry_id, no rebuild)."""
        host = _make_ssh_host(tmp_path)
        rec = _RecordingManager(entry_id="entry-xyz")
        ctor = MagicMock(return_value=rec)
        clock = _FakeClock()
        # Same collector across soft-join and the real allocation (one attempt):
        # the reuse guard binds the cached manager to its owning collector, so
        # rec must report that same collector to be reused.
        collector = _cm_collector()
        rec.collector = collector
        with (
            patch.object(host_management, "CoreLockManager", ctor),
            patch.object(host, "_get_remote_executor", return_value=MagicMock()),
            patch.object(host, "get_instance_type", return_value="trn1"),
            patch("time.sleep", clock.sleep),
            patch("time.time", clock.time),
        ):
            host.soft_join_queue(collector=collector, collective_ranks=1, lnc_config=2)
            entry_after_soft_join = host._core_lock_manager.entry_id
            # Now the real allocation commits ready=True on the SAME manager.
            rec._seq = [AllocationOutcome(status=AllocationStatus.ALLOCATED, logical_cores=[0], physical_cores=[0, 1])]
            with host.get_core_allocation(collector=collector, collective_ranks=1, lnc_config=2) as alloc:
                assert alloc.logical_core_ids == [0]

        # No second construction; same entry_id; first poll ready=False, then ready=True.
        assert ctor.call_count == 1
        assert entry_after_soft_join == "entry-xyz"
        assert rec.acquire_ready_args == [False, True]
        assert rec.released == [[0, 1]]
        # Cache cleared on teardown so the next attempt starts fresh.
        assert host._core_lock_manager is None

    def test_soft_join_swallows_exceptions(self, tmp_path) -> None:
        """Best-effort: any failure during soft-join is logged and swallowed."""
        host = _make_ssh_host(tmp_path)
        rec = _RecordingManager()
        rec._seq = [LockAcquisitionError("boom")]
        with (
            patch.object(host_management, "CoreLockManager", MagicMock(return_value=rec)),
            patch.object(host, "_get_remote_executor", return_value=MagicMock()),
            patch.object(host, "get_instance_type", return_value="trn1"),
        ):
            # Must NOT raise even though acquire raises.
            host.soft_join_queue(collector=_cm_collector(), collective_ranks=1, lnc_config=2)
        assert rec.acquire_ready_args == [False]

    def _soft_join(self, host, fake_mgr, **kwargs) -> None:
        """Drive soft_join_queue with a scripted fake manager (no real SSH)."""
        with (
            patch.object(host_management, "CoreLockManager", MagicMock(return_value=fake_mgr)),
            patch.object(host, "_get_remote_executor", return_value=MagicMock()),
            patch.object(host, "get_instance_type", return_value="trn1"),
        ):
            host.soft_join_queue(collector=_cm_collector(), **kwargs)

    def test_soft_join_over_patience_queued_rotates_and_dequeues(self, tmp_path) -> None:
        """An over-patience QUEUED soft-join dequeues the slot and raises
        QueuePatienceRotation (a TimeoutException) so the host rotates BEFORE the
        upload; no contention metrics are flushed (none have accrued yet)."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [AllocationOutcome(status=AllocationStatus.QUEUED, position=5, worst_case_eta=PATIENCE_SECONDS + 1)]
        )
        with pytest.raises(QueuePatienceRotation, match="rotating host") as exc_info:
            self._soft_join(host, fake_mgr, collective_ranks=1, lnc_config=2)
        assert isinstance(exc_info.value, TimeoutException)
        # Exactly one ready=False poll, the slot freed once, no contention flush.
        assert fake_mgr.acquire_calls == 1
        assert fake_mgr.dequeue_calls == 1
        assert fake_mgr.metrics_calls == 0

    def test_soft_join_over_patience_clears_cached_manager(self, tmp_path) -> None:
        """After an over-patience soft-join raise, the cached manager must be
        cleared so a re-selection of this same host re-mints a fresh manager
        (new entry_id + reset fairness counters) instead of inheriting the
        abandoned soft-join's queue-wait, mirroring get_core_allocation's
        per-attempt teardown."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [AllocationOutcome(status=AllocationStatus.QUEUED, position=5, worst_case_eta=PATIENCE_SECONDS + 1)]
        )
        with pytest.raises(QueuePatienceRotation, match="rotating host"):
            self._soft_join(host, fake_mgr, collective_ranks=1, lnc_config=2)
        # The slot was freed via the cached manager (dequeue ran) ...
        assert fake_mgr.dequeue_calls == 1
        # ... and the cache is cleared so the next attempt starts fresh.
        assert host._core_lock_manager is None

    def test_soft_join_within_patience_queued_does_not_raise(self, tmp_path) -> None:
        """A QUEUED soft-join whose ETA is within patience returns normally and
        leaves the reserved slot in place (no dequeue, no raise)."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [AllocationOutcome(status=AllocationStatus.QUEUED, position=1, worst_case_eta=PATIENCE_SECONDS)]
        )
        self._soft_join(host, fake_mgr, collective_ranks=1, lnc_config=2)
        assert fake_mgr.acquire_calls == 1
        assert fake_mgr.dequeue_calls == 0
        assert fake_mgr.metrics_calls == 0

    def test_soft_join_draining_over_patience_does_not_raise(self, tmp_path) -> None:
        """A DRAINING soft-join never rotates even with an over-patience ETA
        (should_rotate returns False while draining; stay-and-probe)."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [AllocationOutcome(status=AllocationStatus.DRAINING, position=0, worst_case_eta=PATIENCE_SECONDS + 1)]
        )
        self._soft_join(host, fake_mgr, collective_ranks=1, lnc_config=2)
        assert fake_mgr.acquire_calls == 1
        assert fake_mgr.dequeue_calls == 0
        assert fake_mgr.metrics_calls == 0

    def test_soft_join_draining_invokes_should_rotate_with_draining_true(self, tmp_path) -> None:
        """A DRAINING outcome must route the draining state into should_rotate
        (draining=True), making should_rotate the single suppression point
        rather than the status guard short-circuiting before it."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [AllocationOutcome(status=AllocationStatus.DRAINING, position=0, worst_case_eta=PATIENCE_SECONDS + 1)]
        )
        calls: list[bool] = []

        def _spy(worst_case_eta, draining):
            calls.append(draining)
            return should_rotate(worst_case_eta, draining)

        with patch.object(host_management, "should_rotate", _spy):
            self._soft_join(host, fake_mgr, collective_ranks=1, lnc_config=2)
        assert calls == [True]

    def test_soft_join_failed_poll_is_swallowed_without_dequeue(self, tmp_path) -> None:
        """A genuine poll failure is swallowed (no raise) and never triggers the
        patience gate, so the slot is not dequeued."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager([LockAcquisitionError("boom")])
        self._soft_join(host, fake_mgr, collective_ranks=1, lnc_config=2)
        assert fake_mgr.acquire_calls == 1
        assert fake_mgr.dequeue_calls == 0
        assert fake_mgr.metrics_calls == 0

    def test_base_and_local_soft_join_is_noop(self, tmp_path) -> None:
        """LocalHost inherits the base no-op; it must not enqueue or error."""
        # LocalHost does not override the base no-op (uses the file-lock path).
        assert LocalHost.soft_join_queue is host_management.Host.soft_join_queue
        local = LocalHost("/opt/aws/neuron/bin", "localhost", str(tmp_path))
        assert local.soft_join_queue(collector=MagicMock(), collective_ranks=2, lnc_config=2) is None


class TestSoftJoinIntegration(TestSshHostPollLoopIntegration):
    """No-hardware integration: soft-join slot reservation preserves FIFO order."""

    def test_soft_join_reserves_upload_window_slot_and_preserves_fifo(self, tmp_path) -> None:
        """A soft-joins (ready=False) at slot 0 without committing even though cores
        are free; B (ready=True) cannot jump ahead; A then commits (ready=True)."""
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=0.0)
        self._write_locks(locks_json)  # all 8 cores free
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        with patch("time.time", clock.time), patch("time.sleep", clock.sleep):
            mgr_a = CoreLockManager(
                "fakehost",
                total_physical_cores=8,
                collector=NoopMetricsCollector(),
                executor=executor,
                host_locking_version=3,
            )
            mgr_b = CoreLockManager(
                "fakehost",
                total_physical_cores=8,
                collector=NoopMetricsCollector(),
                executor=executor,
                host_locking_version=3,
            )

            # A soft-joins during "upload": enqueues at position 0 but does NOT
            # commit, even though all cores are free (ready=False never grabs cores).
            out_a = mgr_a.acquire(num_logical_cores=4, lnc_config=2, ready=False)
            assert out_a.status == AllocationStatus.QUEUED
            assert out_a.position == 0
            queue = self._read_queue(locks_json)
            assert [e["entry_id"] for e in queue] == [mgr_a.entry_id]
            # No cores claimed by the soft-join (slot is metadata-only).
            import json as _json

            with open(locks_json) as f:
                assert _json.load(f).get("physical_neuron_cores_lock_timeout", {}) == {}

            # B arrives ready-to-commit but must NOT jump ahead of A's reserved slot.
            out_b = mgr_b.acquire(num_logical_cores=4, lnc_config=2, ready=True)
            assert out_b.status == AllocationStatus.QUEUED
            assert [e["entry_id"] for e in self._read_queue(locks_json)] == [mgr_a.entry_id, mgr_b.entry_id]

            # A finishes uploading and commits on the SAME entry_id -> ALLOCATED.
            out_a2 = mgr_a.acquire(num_logical_cores=4, lnc_config=2, ready=True)
            assert out_a2.status == AllocationStatus.ALLOCATED
            assert len(out_a2.logical_cores) == 4


# =============================================================================
# Per-attempt CoreLockManager lifetime (F2).
#
# The HostManager keeps ONE SshHost per alias for the whole pytest-worker
# session. Before the fix, SshHost cached its CoreLockManager once per host and
# never cleared it, so the manager's entry_id, captured collector, and latched
# fairness counters bled across every later test on the
# same host. These tests pin the fix: the cached manager is bound to its owning
# collector and dropped on teardown, so each allocation attempt gets a fresh
# manager (new entry_id, rebound collector, reset counters) while soft-join ->
# commit reuse within one attempt (same collector) is preserved.
# =============================================================================


class _RecordingCollector(NoopMetricsCollector):
    """Captures record_metric/record_timer calls so per-collector attribution is observable."""

    def __init__(self, test_name: str = "rec") -> None:
        self.test_name = test_name
        self.metrics: list[tuple] = []
        self.timers: list[tuple] = []

    def record_metric(self, name, value, unit: str = "None") -> None:  # noqa: D102
        self.metrics.append((name, value, unit))

    def record_timer(self, name, duration_seconds) -> None:  # noqa: D102
        self.timers.append((name, duration_seconds))


class TestManagerLifetimePerAttempt(TestSshHostPollLoopIntegration):
    """No-hardware integration proving per-attempt manager lifetime (F2)."""

    def _metric_values(self, collector, name):
        return [v for (n, v, _u) in collector.metrics if n == name]

    def test_manager_resets_across_attempts_on_same_host(self, tmp_path) -> None:
        """Two attempts on the SAME SshHost (different collectors) each get a fresh
        manager: distinct entry_id, rebound collector, and per-attempt counters
        recorded on the owning collector only.

        Before the fix this was impossible: the cached manager's fairness counters
        latched after attempt #1, attributing attempt #2's metrics onto
        attempt #1's collector. This exercises the exact cross-test bleed F2
        describes over the real helper + a temp locks.json.
        """
        import json as _json

        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        # Attempt #1: cores busy with ETA ~100s (<= patience, so no rotation) ->
        # the caller enqueues, then cores are freed before the 2nd poll so it
        # commits THROUGH the queue.
        self._write_locks(locks_json, locked_until=int(_REALISTIC_EPOCH) + 100)

        def free_cores_on_second_call(call_n, path):
            if call_n == 2:
                with open(path) as f:
                    state = _json.load(f)
                state["physical_neuron_cores_lock_timeout"] = {}
                with open(path, "w") as f:
                    _json.dump(state, f)

        executor = _HelperExecutor(helpers_file, lock_file, locks_json, on_call=free_cores_on_second_call)
        host = _make_ssh_host(tmp_path)
        col1 = _RecordingCollector("attempt-1")
        col2 = _RecordingCollector("attempt-2")

        with (
            patch.object(host, "_get_remote_executor", return_value=executor),
            patch.object(host, "get_instance_type", return_value="trn1"),
            patch("time.sleep", clock.sleep),
            patch("time.time", clock.time),
        ):
            with host.get_core_allocation(collector=col1, collective_ranks=4, lnc_config=2, timeout_seconds=300) as a1:
                entry1 = host._core_lock_manager.entry_id
                assert host._core_lock_manager.collector is col1
                assert len(a1.logical_core_ids) == 4
            # Teardown drops the cache so the next attempt cannot reuse it.
            assert host._core_lock_manager is None

            # Attempt #2: freed cores -> commits THROUGH the queue on a later poll.
            with host.get_core_allocation(collector=col2, collective_ranks=4, lnc_config=2, timeout_seconds=300) as a2:
                entry2 = host._core_lock_manager.entry_id
                assert host._core_lock_manager.collector is col2
                assert len(a2.logical_core_ids) == 4
            assert host._core_lock_manager is None

        # Distinct entry_id minted per attempt (not once per host).
        assert entry1 != entry2
        # Each attempt's contention counter is recorded exactly once on its OWN
        # collector (proving the manager + collector binding reset per attempt).
        assert len(self._metric_values(col1, MetricName.CORE_LOCK_NO_CORES_COUNT)) == 1
        assert len(self._metric_values(col2, MetricName.CORE_LOCK_NO_CORES_COUNT)) == 1

    def test_distinct_collectors_get_distinct_managers_no_cross_attribution(self, tmp_path) -> None:
        """Two back-to-back attempts (different collectors) on the same host
        mint distinct managers/entry_ids; neither collector receives the other's metrics."""
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        self._write_locks(locks_json)  # all cores free -> instant ALLOCATED
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)
        host = _make_ssh_host(tmp_path)
        col1 = _RecordingCollector("attempt-1")
        col2 = _RecordingCollector("attempt-2")

        with (
            patch.object(host, "_get_remote_executor", return_value=executor),
            patch.object(host, "get_instance_type", return_value="trn1"),
            patch("time.sleep", clock.sleep),
            patch("time.time", clock.time),
        ):
            with host.get_core_allocation(collector=col1, collective_ranks=4, lnc_config=2) as a1:
                mgr1 = host._core_lock_manager
                entry1 = mgr1.entry_id
                assert mgr1.collector is col1
                assert len(a1.logical_core_ids) == 4
            with host.get_core_allocation(collector=col2, collective_ranks=4, lnc_config=2) as a2:
                mgr2 = host._core_lock_manager
                entry2 = mgr2.entry_id
                assert mgr2.collector is col2
                assert len(a2.logical_core_ids) == 4

        assert mgr1 is not mgr2
        assert entry1 != entry2
        # Each attempt's metrics are attributed to its own collector only.
        assert len(self._metric_values(col1, MetricName.CORE_LOCK_NO_CORES_COUNT)) == 1
        assert len(self._metric_values(col2, MetricName.CORE_LOCK_NO_CORES_COUNT)) == 1


# =============================================================================
# End-to-end manager <-> helper integration under concurrent reservations (IT-4 / T9).
#
# These tests drive one or more REAL CoreLockManager instances (and, where the
# stay/rotate decision is under test, the real SshHost poll loop) against the
# REAL lock helper (remote_lock_scripts) over a single shared temp locks.json
# via _HelperExecutor. They prove the concurrent per-entry reservation model
# behaves correctly end-to-end: disjoint concurrent landing, soft-join entry_id
# reuse, manager-layer W-lazy (a ready waiter is not blocked by an uploading
# predecessor), patience that stays on a within-patience concurrent ETA,
# long-upload bump + eventual commit, RESERVE_EXPIRED observability in the event
# log, and drain-then-grant in FIFO order.
#
# T9: the manager consumes the SAME wire LockResult surface as before, so no
# manager source change is required -- these tests pass against the unmodified
# manager/poll loop, confirming transparency.
# =============================================================================


class TestManagerConcurrentReservationE2E(TestSshHostPollLoopIntegration):
    """Multiple managers over one shared locks.json under concurrent reservations."""

    def _mk_mgr(self, executor) -> CoreLockManager:
        """A real CoreLockManager bound to the shared helper executor (8-core host)."""
        return CoreLockManager(
            "fakehost",
            total_physical_cores=8,
            collector=NoopMetricsCollector(),
            executor=executor,
            host_locking_version=3,
        )

    def _read_events(self, locks_json) -> list:
        """Read the helper's daily-rotated event log(s) beside locks.json."""
        import glob
        import json as _json

        d = os.path.dirname(locks_json)
        events = []
        for fn in sorted(glob.glob(os.path.join(d, "events-*.log"))):
            with open(fn) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(_json.loads(line))
        return events

    def _free_all_cores(self, locks_json) -> None:
        """Clear all live core locks while preserving the queue/drain fields."""
        import json as _json

        with open(locks_json) as f:
            state = _json.load(f)
        state["physical_neuron_cores_lock_timeout"] = {}
        with open(locks_json, "w") as f:
            _json.dump(state, f)

    def _clear_drain(self, locks_json) -> None:
        """Disable drain while preserving the queue (does not wipe the file)."""
        import json as _json

        with open(locks_json) as f:
            state = _json.load(f)
        state.pop("draining_enabled_with_timeout", None)
        with open(locks_json, "w") as f:
            _json.dump(state, f)

    def _assert_disjoint(self, allocations) -> None:
        """No physical core may appear in two managers' allocations simultaneously."""
        seen: set[int] = set()
        for cores in allocations:
            block = set(cores or [])
            overlap = block & seen
            assert not overlap, f"physical core double-allocated across managers: {sorted(overlap)}"
            seen |= block

    def test_multiple_managers_land_concurrently_disjoint(self, tmp_path) -> None:
        """N managers (distinct entry_ids) over ONE locks.json each request w-cores
        that together fit (4 x 2 cores on an 8-core host). Each soft-joins
        (ready=False) then commits (ready=True). Every manager reaches ALLOCATED on
        DISJOINT physical cores, no core is double-allocated, and the helper event
        log contains only ENQUEUE/COMMIT (no ACQUIRE -- the fast path is gone)."""
        import random

        random.seed(42)
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        self._write_locks(locks_json)  # all 8 cores free
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        with patch("time.time", clock.time), patch("time.sleep", clock.sleep):
            managers = [self._mk_mgr(executor) for _ in range(4)]
            # Soft-join: each anchors a FIFO slot (ready=False) without grabbing cores.
            for mgr in managers:
                out = mgr.acquire(num_logical_cores=1, lnc_config=2, ready=False)
                assert out.status == AllocationStatus.QUEUED
            # Four distinct entry_ids enqueued, no cores claimed by the soft-joins.
            queue = self._read_queue(locks_json)
            assert {e["entry_id"] for e in queue} == {m.entry_id for m in managers}
            import json as _json

            with open(locks_json) as f:
                assert _json.load(f).get("physical_neuron_cores_lock_timeout", {}) == {}

            # Commit: one ready=True poll per manager (the soft-joined entry already
            # exists, so reconcile reserves its block and it commits the same poll).
            allocations = []
            for mgr in managers:
                out = mgr.acquire(num_logical_cores=1, lnc_config=2, ready=True)
                assert out.status == AllocationStatus.ALLOCATED
                assert out.physical_cores is not None and len(out.physical_cores) == 2
                allocations.append(out.physical_cores)

        # Every manager landed on a disjoint pair; all 8 cores used exactly once.
        self._assert_disjoint(allocations)
        assert sorted(c for block in allocations for c in block) == list(range(8))

        # Event log: only ENQUEUE + COMMIT transitions -- the fast-path ACQUIRE is gone.
        events = self._read_events(locks_json)
        kinds = {e["event"] for e in events}
        assert "ACQUIRE" not in kinds
        assert kinds == {"ENQUEUE", "COMMIT"}
        assert sum(e["event"] == "ENQUEUE" for e in events) == 4
        assert sum(e["event"] == "COMMIT" for e in events) == 4

    def test_soft_join_reuses_entry_id_through_commit(self, tmp_path) -> None:
        """One manager soft-joins (ready=False -> QUEUED) then commits
        (ready=True -> ALLOCATED) on the SAME entry_id: no re-enqueue on the happy
        path (re_enqueued/bump counters stay 0) and the queue never holds two
        entries for it."""
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        self._write_locks(locks_json)  # all cores free
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        with patch("time.time", clock.time), patch("time.sleep", clock.sleep):
            mgr = self._mk_mgr(executor)
            entry_id = mgr.entry_id

            # Poll #1: soft-join (ready=False) -> IN_QUEUE, exactly one queue entry.
            out1 = mgr.acquire(num_logical_cores=2, lnc_config=2, ready=False)
            assert out1.status == AllocationStatus.QUEUED
            queue = self._read_queue(locks_json)
            assert [e["entry_id"] for e in queue] == [entry_id]

            # Poll #2: commit (ready=True) on the SAME cached entry_id -> ALLOCATED.
            out2 = mgr.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            assert out2.status == AllocationStatus.ALLOCATED
            assert out2.physical_cores is not None and len(out2.physical_cores) == 4

            # Same entry_id throughout; no re-enqueue and no bump on the happy path.
            assert mgr.entry_id == entry_id
            assert mgr._reenqueue_count == 0
            assert mgr._bump_count == 0
            # Committed + removed -> queue empty (never two entries for this id).
            assert self._read_queue(locks_json) == []

        # Event log: a single ENQUEUE for this entry_id then its COMMIT; no
        # second ENQUEUE, no BUMP/PRUNE/RESERVE_EXPIRED (no re-enqueue happened).
        events = self._read_events(locks_json)
        enqueues = [e for e in events if e["event"] == "ENQUEUE" and e.get("entry_id") == entry_id]
        commits = [e for e in events if e["event"] == "COMMIT" and str(e.get("caller", "")).startswith(entry_id)]
        assert len(enqueues) == 1
        assert len(commits) == 1
        assert not any(e["event"] in {"BUMP", "PRUNE", "RESERVE_EXPIRED"} for e in events)

    def test_ready_waiter_not_blocked_by_uploading_manager(self, tmp_path) -> None:
        """Manager A soft-joins and stays ready=False (a long upload); manager B
        (behind A) soft-joins then polls ready=True. B reaches ALLOCATED while A is
        still not-ready (manager-layer W-lazy: a ready waiter backfills past a
        not-ready predecessor), then A commits onto disjoint cores."""
        import random

        random.seed(42)
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        self._write_locks(locks_json)  # all cores free
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        with patch("time.time", clock.time), patch("time.sleep", clock.sleep):
            mgr_a = self._mk_mgr(executor)
            mgr_b = self._mk_mgr(executor)

            # A soft-joins at the head and stays not-ready (uploading).
            out_a = mgr_a.acquire(num_logical_cores=2, lnc_config=2, ready=False)
            assert out_a.status == AllocationStatus.QUEUED
            assert out_a.position == 0
            # B soft-joins behind A, then commits ready=True.
            out_b0 = mgr_b.acquire(num_logical_cores=2, lnc_config=2, ready=False)
            assert out_b0.status == AllocationStatus.QUEUED
            assert out_b0.position == 1
            out_b = mgr_b.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            assert out_b.status == AllocationStatus.ALLOCATED
            assert out_b.physical_cores is not None and len(out_b.physical_cores) == 4

            # A is STILL in the queue and STILL not-ready: B backfilled past it.
            queue = self._read_queue(locks_json)
            a_entry = next(e for e in queue if e["entry_id"] == mgr_a.entry_id)
            assert a_entry.get("ready", False) is False

            # A finishes uploading and commits on the SAME entry_id -> disjoint cores.
            out_a2 = mgr_a.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            assert out_a2.status == AllocationStatus.ALLOCATED
            assert out_a2.physical_cores is not None and len(out_a2.physical_cores) == 4

        self._assert_disjoint([out_b.physical_cores, out_a2.physical_cores])

    def test_patience_stays_on_host_with_concurrent_eta(self, tmp_path) -> None:
        """A board where the OLD serial ETA would exceed patience (so the caller
        would rotate) but the NEW concurrent ETA is within patience because the
        caller fits alongside its predecessor. The manager stays (does not rotate)
        and eventually ALLOCATEs onto cores disjoint from its predecessor."""
        import random

        random.seed(42)
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        # All 8 cores busy until +170s. Concurrent ETA for a 4-core caller that
        # backfills past a 4-core predecessor is ~170s (<= patience 180s). The OLD
        # serial model would stack the predecessor's 60s hold on top (~230s),
        # exceeding patience and forcing a rotation.
        self._write_locks(locks_json, locked_until=int(_REALISTIC_EPOCH) + 170)
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        with patch("time.time", clock.time), patch("time.sleep", clock.sleep):
            pred = self._mk_mgr(executor)
            caller = self._mk_mgr(executor)

            # Predecessor anchors the head (4 cores). Caller enqueues behind it.
            assert pred.acquire(num_logical_cores=2, lnc_config=2, ready=True).status == AllocationStatus.QUEUED
            out = caller.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            assert out.status == AllocationStatus.QUEUED
            assert out.position == 1
            # Concurrent-aware ETA is within patience -> the manager would NOT rotate.
            assert out.worst_case_eta is not None
            assert out.worst_case_eta <= PATIENCE_SECONDS
            assert not should_rotate(out.worst_case_eta, draining=False)
            # The serial-model ETA (predecessor hold stacked on the lock) would have
            # exceeded patience, proving the difference is the concurrent model.
            serial_eta = out.worst_case_eta + 60
            assert serial_eta > PATIENCE_SECONDS

            # Cores free; both reserve disjoint blocks and commit in FIFO order.
            self._free_all_cores(locks_json)
            out_pred = pred.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            out_caller = caller.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            assert out_pred.status == AllocationStatus.ALLOCATED
            assert out_caller.status == AllocationStatus.ALLOCATED

        self._assert_disjoint([out_pred.physical_cores, out_caller.physical_cores])

    def test_long_upload_bumps_then_reenqueues(self, tmp_path) -> None:
        """A manager that stays ready=False past its one-shot readiness grace is
        bumped to the tail by reconcile (a sole-entry bump is a tail re-enqueue).
        The manager observes the bump (bump_count increments; reenqueue_count stays
        0 -- a bump is distinct from a stale-prune re-enqueue) and ultimately still
        commits once ready."""
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        self._write_locks(locks_json)  # all cores free
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        with patch("time.time", clock.time), patch("time.sleep", clock.sleep):
            mgr = self._mk_mgr(executor)

            # Poll #1: soft-join at the head (ready=False); no grace opened yet.
            assert mgr.acquire(num_logical_cores=2, lnc_config=2, ready=False).status == AllocationStatus.QUEUED
            # Poll #2: reconcile opens the head's one-shot readiness grace.
            out2 = mgr.acquire(num_logical_cores=2, lnc_config=2, ready=False)
            assert out2.status == AllocationStatus.QUEUED
            assert mgr._bump_count == 0

            # Let the readiness grace lapse (still well within STALE_THRESHOLD so the
            # entry is bumped to the tail, NOT pruned).
            clock.sleep(remote_lock_scripts.COMMIT_WINDOW + 1)

            # Poll #3: reconcile bumps the stuck not-ready head to the tail.
            out3 = mgr.acquire(num_logical_cores=2, lnc_config=2, ready=False)
            assert out3.status == AllocationStatus.QUEUED
            assert mgr._bump_count >= 1
            assert mgr._reenqueue_count == 0

            # Poll #4: now ready -> reserve + commit on the same entry_id.
            out4 = mgr.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            assert out4.status == AllocationStatus.ALLOCATED
            assert out4.physical_cores is not None and len(out4.physical_cores) == 4

        # The bump is observable in the event log; no prune occurred.
        events = self._read_events(locks_json)
        assert any(e["event"] == "BUMP" and e.get("entry_id") == mgr.entry_id for e in events)
        assert not any(e["event"] == "PRUNE" for e in events)
        # Single allocation -> trivially no double-allocation.
        self._assert_disjoint([out4.physical_cores])

    def test_vanished_reserver_reserve_expired_in_event_log(self, tmp_path) -> None:
        """A vanisher V is granted a reservation (by a follower's reconcile) then
        stops polling. Advancing the clock past its commit deadline makes reconcile
        reclaim the tiles and log RESERVE_EXPIRED (observability = event log, not a
        manager metric). A live manager B behind it then commits the freed cores
        once V is finally pruned."""
        import random

        random.seed(42)
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        self._write_locks(locks_json)  # all cores free
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        with patch("time.time", clock.time), patch("time.sleep", clock.sleep):
            vanisher = self._mk_mgr(executor)  # wants ALL 8 cores -> blocks the follower
            follower = self._mk_mgr(executor)  # wants 4 cores, behind the vanisher

            # V enqueues at the head (ready, unreserved).
            assert vanisher.acquire(num_logical_cores=4, lnc_config=2, ready=True).status == AllocationStatus.QUEUED
            # B's poll runs reconcile, which reserves V its 8-core block; B is a
            # barrier behind it. V now holds a reservation it must commit soon.
            out_b = follower.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            assert out_b.status == AllocationStatus.QUEUED
            v_entry = next(e for e in self._read_queue(locks_json) if e["entry_id"] == vanisher.entry_id)
            assert v_entry.get("reserved_cores")

            # V vanishes. Advance past its commit deadline; B's next poll reclaims
            # the reservation (RESERVE_EXPIRED) -- V keeps its FIFO slot for now.
            clock.sleep(remote_lock_scripts.COMMIT_WINDOW + 1)
            out_b2 = follower.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            assert out_b2.status == AllocationStatus.QUEUED
            events = self._read_events(locks_json)
            assert any(e["event"] == "RESERVE_EXPIRED" and e.get("entry_id") == vanisher.entry_id for e in events)

            # B keeps polling (refreshing its own liveness). V never polls, so once
            # it goes stale it is pruned and B commits the freed cores.
            out = out_b2
            for _ in range(20):
                clock.sleep(POLL_PERIOD)
                out = follower.acquire(num_logical_cores=2, lnc_config=2, ready=True)
                if out.status == AllocationStatus.ALLOCATED:
                    break
            assert out.status == AllocationStatus.ALLOCATED
            assert out.physical_cores is not None and len(out.physical_cores) == 4

        # V is gone from the queue and a PRUNE was logged for it.
        assert all(e["entry_id"] != vanisher.entry_id for e in self._read_queue(locks_json))
        events = self._read_events(locks_json)
        assert any(e["event"] == "PRUNE" and e.get("entry_id") == vanisher.entry_id for e in events)
        # The committed cores are B's alone (V vanished) -> no double-allocation.
        self._assert_disjoint([out.physical_cores])

    def test_drain_keeps_managers_polling_then_grants(self, tmp_path) -> None:
        """While draining, managers receive DRAINING and hold their FIFO positions;
        once drain is disabled they commit in FIFO order onto disjoint cores."""
        import random

        random.seed(42)
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        # All cores free but the host is draining (far beyond the test).
        self._write_locks(locks_json, draining_until=int(_REALISTIC_EPOCH) + 100000)
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        with patch("time.time", clock.time), patch("time.sleep", clock.sleep):
            mgr_a = self._mk_mgr(executor)
            mgr_b = self._mk_mgr(executor)

            # Both enqueue while draining: DRAINING (grants withheld), FIFO positions.
            out_a = mgr_a.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            out_b = mgr_b.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            assert out_a.status == AllocationStatus.DRAINING and out_a.position == 0
            assert out_b.status == AllocationStatus.DRAINING and out_b.position == 1

            # Re-poll: still draining, positions preserved (stay-and-probe, no grant).
            out_a2 = mgr_a.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            out_b2 = mgr_b.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            assert out_a2.status == AllocationStatus.DRAINING and out_a2.position == 0
            assert out_b2.status == AllocationStatus.DRAINING and out_b2.position == 1

            # Disable drain -> next polls commit in FIFO order (head first).
            self._clear_drain(locks_json)
            out_a3 = mgr_a.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            out_b3 = mgr_b.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            assert out_a3.status == AllocationStatus.ALLOCATED
            assert out_b3.status == AllocationStatus.ALLOCATED

        self._assert_disjoint([out_a3.physical_cores, out_b3.physical_cores])
