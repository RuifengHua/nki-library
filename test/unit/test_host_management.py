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
from test.utils.host_management import HostManager, LocalHost, detect_local_neuron_devices, temporary_random_seed


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
                    max_retries=2,
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
                    max_retries=3,
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
                    max_retries=3,
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
