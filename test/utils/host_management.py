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
import contextlib
import functools
import json
import logging
import os
import random
import re
import shutil
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Generator, final

import fabric2
from attr import dataclass
from filelock import FileLock
from paramiko import SSHException
from typing_extensions import override

from . import core_lock_client as lock_client
from .common_dataclasses import INF_ARTIFACT_DIR_NAME, NeuronDeviceInfo, Platforms, TargetHost
from .core_lock_client import POLL_JITTER_MAX, POLL_PERIOD, REMOTE_LOCKS_JSON, STALE_THRESHOLD
from .core_lock_manager import (
    AllocationStatus,
    CoreAllocation,
    CoreLockManager,
    LockAcquisitionError,
    LockVersionError,
)
from .exceptions import (
    InferenceException,
    LocalExecutionException,
    NoNeuronDevicesException,
    QueuePatienceRotation,
    RemoteExecutionException,
    TimeoutException,
    UnimplementedException,
)
from .metrics_collector import IMetricsCollector, MetricName
from .remote_executor import RemoteExecutor
from .resources import RemoteDirectory
from .s3_utils import S3ArtifactUploadConfig
from .scripts.remote_lock_scripts import LockState, find_contiguous_cores

# Caller-side patience for FIFO core allocation.
# If a host's worst-case ETA exceeds this, the caller dequeues and rotates to
# another host rather than camping. Capped well under the per-test pytest
# timeout so the caller can cycle through a few hosts before the test is killed.
PATIENCE_SECONDS = 3 * lock_client.DEFAULT_LOCK_TIMEOUT_SECONDS  # = 180

# Wall-clock deadline for acquiring a host + core allocation. A busy FIFO queue
# is transient (not a host failure), so the caller keeps rotating across hosts
# until this deadline elapses rather than giving up after a fixed attempt count.
# ~2.5h, capped under the overall suite budget.
DEFAULT_ACQUISITION_DEADLINE_SECONDS = 9000
# Backoff between host rotations when the whole fleet is momentarily busy, so a
# fully-busy fleet re-cycles at a bounded rate instead of hot-spinning.
HOST_ROTATION_BACKOFF_SECONDS = 1

# Short poll cadence for a near-front waiter. 1 s is the empirical FCFS hand-off
# floor (bounded by the SSH round-trip), so polling this fast near the front
# closes the per-hand-off idle gap without spending extra load for no gain.
# Must stay below POLL_PERIOD (FAST_POLL_PERIOD < POLL_PERIOD).
FAST_POLL_PERIOD = 1


def select_poll_base(position: int | None, *, fast: int = FAST_POLL_PERIOD, slow: int = POLL_PERIOD) -> int:
    """Return the poll-loop sleep base for a queued caller.

    A caller at or one slot from the front of the queue (``position <= 1``)
    polls at the short ``fast`` cadence so it commits its reserved block
    within ~1 s of a hand-off instead of waiting a full slow poll cycle; any
    deeper position (or an unknown position) uses the normal ``slow`` cadence.
    The ``fast``/``slow`` arguments are injectable so callers can pass a
    customized slow period.
    """
    if position is not None and position <= 1:
        return fast
    return slow


def should_rotate(worst_case_eta: int | None, draining: bool) -> bool:
    """Pure predicate: should a queued caller abandon this host and rotate away?

    Returns True only for a non-draining caller whose worst-case ETA exceeds
    ``PATIENCE_SECONDS``. A draining host never triggers rotation (ETA is
    meaningless mid-drain, so the caller stays and keeps probing),
    and a missing ETA is treated as "stay". Kept pure/host-free for unit testing.
    """
    return (not draining) and worst_case_eta is not None and worst_case_eta > PATIENCE_SECONDS


@contextlib.contextmanager
def temporary_random_seed(seed: int) -> Generator[None, None, None]:
    """Temporarily reseed the global RNG, restoring the original state on exit."""
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


@functools.lru_cache(maxsize=1)
def _run_neuron_ls(neuron_installation_path: str) -> list[dict] | None:
    """Run neuron-ls --json-output and return parsed JSON, or None on failure."""
    try:
        neuron_ls_path = os.path.join(neuron_installation_path, "neuron-ls")
        if not os.path.isfile(neuron_ls_path):
            return None
        result = subprocess.run(
            [neuron_ls_path, "--json-output"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if data:
                return data
        return None
    except Exception:
        return None


def detect_local_neuron_devices(neuron_installation_path: str) -> bool:
    """Check if local Neuron devices are available via neuron-ls.

    Returns False gracefully if neuron-ls is not installed, times out, or finds no devices.
    """
    return _run_neuron_ls(neuron_installation_path) is not None


def detect_local_platform(neuron_installation_path: str) -> Platforms | None:
    """Detect the local Neuron platform type from neuron-ls instance_type.

    Returns the Platforms enum value, or None if detection fails.
    """
    data = _run_neuron_ls(neuron_installation_path)
    if not data:
        return None

    instance_type = data[0].get("instance_type", "")
    # instance_type is e.g. "trn2.48xlarge" or "trn3pds.48xlarge" — extract "trn" + digits
    match = re.match(r"(trn\d+)", instance_type)
    if not match:
        logging.warning(f"Unknown platform from instance_type '{instance_type}', cannot auto-detect")
        return None
    try:
        return Platforms(match.group(1))
    except ValueError:
        logging.warning(f"Unknown platform from instance_type '{instance_type}', cannot auto-detect")
        return None


class Host(ABC):
    def execute_command(
        self,
        command: str,
        target_directory: str,
        collector: IMetricsCollector,
        collective_ranks: int,
        lnc_config: int,
        do_copy_artifacts: bool = False,
        get_list_of_files_to_copy: Callable[[str], list[str]] | None = None,
        post_lock_command: str | None = None,
    ) -> str | None:
        with self.get_core_allocation(
            collective_ranks=collective_ranks, lnc_config=lnc_config, collector=collector
        ) as core_allocation:
            with collector.timer(MetricName.CORE_LOCK_HOLD_TIME):
                unique_port = self._get_unique_collectives_port(core_allocation.logical_core_ids[0])
                neuron_env = self._build_neuron_env(
                    core_allocation,
                    lnc_config,
                    unique_port,
                    self._get_debug_output_dir(target_directory),
                )

                neuron_env.update(self._setup_neuron_tags())

                stdout = self._run_command(command, target_directory, neuron_env, collector)

        if post_lock_command:
            if self._should_run_post_lock():
                post_lock_stdout = self._run_post_lock_command(post_lock_command, target_directory, collector)
                stdout = stdout + "\n" + post_lock_stdout
            else:
                logging.error("Skipping post-lock commands: host is draining")

        if do_copy_artifacts:
            return self._collect_artifacts(target_directory, stdout, get_list_of_files_to_copy, collector)
        return None

    @abstractmethod
    def _get_debug_output_dir(self, target_directory: str) -> str:
        raise UnimplementedException()

    @abstractmethod
    def _run_command(
        self,
        command: str,
        target_directory: str,
        neuron_env: dict[str, str],
        collector: IMetricsCollector,
    ) -> str:
        """Run the command on the host and return stdout."""
        raise UnimplementedException()

    @abstractmethod
    def _collect_artifacts(
        self,
        target_directory: str,
        stdout: str,
        get_list_of_files_to_copy: Callable[[str], list[str]] | None,
        collector: IMetricsCollector,
    ) -> str:
        """Download/copy artifacts and return the local artifact directory path."""
        raise UnimplementedException()

    @abstractmethod
    def prepare_host(
        self,
        target_directory: str,
        collector: IMetricsCollector,
        skip_remote_cleanup: bool = False,
        force_local_cleanup: bool = False,
    ) -> contextlib.AbstractContextManager[Any, Any]:
        raise UnimplementedException()

    @abstractmethod
    def get_core_allocation(
        self,
        collector: IMetricsCollector,
        collective_ranks: int = 1,
        lnc_config: int = 2,
        timeout_seconds: int = 9000,
        poll_period_seconds: int = POLL_PERIOD,
    ) -> contextlib.AbstractContextManager[CoreAllocation, Any]:
        """
        Allocate logical cores for execution by locking physical cores.

        Prefers aligned allocations for better packing, with randomization to reduce
        contention when multiple workers allocate simultaneously.

        Args:
            collective_ranks: Number of logical cores to allocate
            lnc_config: LNC configuration (1 or 2) - determines physical cores per logical core
            timeout_seconds: Maximum time to wait for core allocation
            poll_period_seconds: Time between allocation attempts

        Returns:
            Context manager yielding CoreAllocation with allocated logical core IDs
        """
        raise UnimplementedException()

    @abstractmethod
    def _run_post_lock_command(
        self,
        command: str,
        target_directory: str,
        collector: IMetricsCollector,
    ) -> str:
        """Run a command that does NOT require Neuron hardware, after core lock release."""
        raise UnimplementedException()

    def _should_run_post_lock(self) -> bool:
        """Check if post-lock commands should run. Override to check drain state."""
        return True

    def soft_join_queue(
        self,
        collector: IMetricsCollector,
        collective_ranks: int = 1,
        lnc_config: int = 2,
    ) -> None:
        """Reserve a FIFO core-queue slot before/while artifacts upload.

        Default no-op. Only SshHost participates in the remote FIFO queue;
        LocalHost uses the file-lock path and must NOT enqueue.
        """
        return None

    @abstractmethod
    def get_neuron_device_info(self) -> list[NeuronDeviceInfo]:
        raise UnimplementedException()

    @abstractmethod
    def get_host_id(self) -> str:
        raise UnimplementedException()

    @staticmethod
    def _get_unique_collectives_port(core_id: int) -> int:
        """Generate unique port for collectives coordination based on first allocated core.

        This ensures parallel tests don't conflict on the same port.
        E.g., test on cores [8, 9] gets port 61242, test on cores [16, 17] gets port 61250.
        """
        return 61234 + core_id

    @staticmethod
    def _build_neuron_env(
        core_allocation: CoreAllocation,
        lnc_config: int,
        unique_port: int,
        debug_output_dir: str,
    ) -> dict[str, str]:
        """Build the common Neuron runtime environment variables for execution."""
        return {
            "NEURON_RT_ENABLE_OCP": "1",
            "NEURON_RT_ENABLE_OCP_SATURATION": "1",
            "NEURON_RT_VISIBLE_CORES": core_allocation.get_core_list_str(),
            "NEURON_LOGICAL_NC_CONFIG": str(lnc_config),
            "NEURON_RT_ROOT_COMM_ID": f"localhost:{unique_port}",
            "NEURON_RT_DEBUG_OUTPUT_DIR": debug_output_dir,
        }

    @staticmethod
    def _setup_neuron_tags() -> dict[str, str]:
        """Build the NEURON_TAG_* dict that ships to the remote workload:
        every NEURON_TAG_* in the local env, plus a fresh per-invocation
        REQUEST_ID. Future tags are picked up automatically.
        """
        tags = {k: v for k, v in os.environ.items() if k.startswith("NEURON_TAG_")}
        tags["NEURON_TAG_REQUEST_ID"] = str(uuid.uuid4())
        return tags

    def __lock__(self, lock_file_path: str, timeout_seconds: int):
        return FileLock(f"{lock_file_path}.lock", timeout=timeout_seconds * 1000)


@final
class LocalHost(Host):
    def __init__(self, local_neuron_installation_path: str, host_id: str, core_allocation_dir: str):
        super().__init__()
        self.neuron_ls_path: str = os.path.join(local_neuron_installation_path, "neuron-ls")
        self.host_id: str = host_id
        self.core_allocation_dir: str = core_allocation_dir
        self._core_state_path: str = os.path.join(core_allocation_dir, "local_core_locks.json")

    @override
    def get_host_id(self) -> str:
        return self.host_id

    @staticmethod
    def _check_no_remote_locks() -> None:
        """Raise if this machine has active remote SSH-based core locks (locks.json)."""
        if not os.path.isfile(REMOTE_LOCKS_JSON):
            return
        try:
            with open(REMOTE_LOCKS_JSON, "r") as f:
                data = json.load(f)
            state = LockState.from_dict(data, default_version=2)
            active = state.get_locked_cores(int(time.time()))
            if active:
                raise RuntimeError(
                    f"Active remote core locks found in {REMOTE_LOCKS_JSON} (cores: {active}). "
                    "This machine appears to be in use as a shared fleet host via SSH. "
                    "Local testing is not supported on shared fleet instances to avoid core allocation conflicts."
                )
        except (json.JSONDecodeError, OSError):
            pass

    @override
    def _get_debug_output_dir(self, target_directory: str) -> str:
        return os.path.join(target_directory, "debug_output")

    @override
    def _run_command(
        self,
        command: str,
        target_directory: str,
        neuron_env: dict[str, str],
        collector: IMetricsCollector,
    ) -> str:
        env = os.environ.copy()
        env.update(neuron_env)

        full_command = f"set -o pipefail; cd {target_directory} && {command}"
        logging.info(
            f"Executing local command: {full_command} "
            f"with NEURON_RT_VISIBLE_CORES={env.get('NEURON_RT_VISIBLE_CORES')} "
            f"NEURON_LOGICAL_NC_CONFIG={env.get('NEURON_LOGICAL_NC_CONFIG')}"
        )

        with collector.timer(MetricName.INFERENCE_TIME):
            result = subprocess.run(
                ["bash", "-c", full_command],
                env=env,
                capture_output=True,
                text=True,
            )

        if result.returncode != 0:
            raise LocalExecutionException(
                f"Unable to execute {command} in {target_directory}",
                result,
            )
        return result.stdout

    @override
    def _run_post_lock_command(
        self,
        command: str,
        target_directory: str,
        collector: IMetricsCollector,
    ) -> str:
        full_command = f"set -o pipefail; cd {target_directory} && {command}"
        logging.info(f"Executing local post-lock command: {full_command}")

        result = subprocess.run(
            ["bash", "-c", full_command],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise LocalExecutionException(
                f"Post-lock command failed in {target_directory}",
                result,
            )
        return result.stdout

    @override
    def _collect_artifacts(
        self,
        target_directory: str,
        stdout: str,
        get_list_of_files_to_copy: Callable[[str], list[str]] | None,
        collector: IMetricsCollector,
    ) -> str:
        local_download_location = os.path.join(target_directory, INF_ARTIFACT_DIR_NAME)
        os.makedirs(local_download_location, exist_ok=True)

        if get_list_of_files_to_copy:
            files_to_copy = get_list_of_files_to_copy(stdout)
            for f in files_to_copy:
                src = os.path.join(target_directory, f)
                dst = os.path.join(local_download_location, f)
                if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.move(src, dst)

        return local_download_location

    @override
    @contextlib.contextmanager
    def prepare_host(
        self,
        target_directory: str,
        collector: IMetricsCollector,
        skip_remote_cleanup: bool = False,
        force_local_cleanup: bool = False,
    ):
        yield

    @override
    @contextlib.contextmanager
    def get_core_allocation(
        self,
        collector: IMetricsCollector,
        collective_ranks: int = 1,
        lnc_config: int = 2,
        timeout_seconds: int = 9000,
        poll_period_seconds: int = POLL_PERIOD,
    ) -> Generator[CoreAllocation, None, None]:
        # Guard against running LocalHost on a machine also used as a remote SshHost target.
        # SshHost uses locks.json for core locking — if it has active (non-expired) locks,
        # another user is running tests via SSH and our local locks won't coordinate with theirs.
        self._check_no_remote_locks()

        with collector.timer(MetricName.CORE_ALLOCATION_TIME):
            devices = self.get_neuron_device_info()
            # neuron-ls returns logical core IDs; multiply by lnc_config to get physical count
            device_lnc = devices[0].logical_neuroncore_config if devices else lnc_config
            collector.add_dimension({MetricName.INSTANCE_TYPE: devices[0].instance_type})
            total_physical_cores = sum(len(d.neuroncore_ids) * device_lnc for d in devices)

            # We lock at the physical core level to prevent conflicts between
            # LNC1 and LNC2 tests (same approach as SshHost/CoreLockManager).
            num_physical_needed = collective_ranks * lnc_config

            if num_physical_needed > total_physical_cores:
                raise NoNeuronDevicesException(
                    f"Requested {collective_ranks} logical cores (lnc{lnc_config} = {num_physical_needed} physical) "
                    f"but only {total_physical_cores} physical cores available on {self.host_id}"
                )

            os.makedirs(self.core_allocation_dir, exist_ok=True)
            lock = self.__lock__(self._core_state_path, timeout_seconds)
            allocated_physical: list[int] = []
            pid = os.getpid()

            deadline = time.time() + timeout_seconds
            while time.time() < deadline:
                with lock.acquire():
                    state = self._read_core_state(total_physical_cores)
                    self._purge_stale_owners(state)
                    all_physical = list(range(total_physical_cores))
                    available = [c for c in all_physical if c not in state["in_use"]]

                    # Find a contiguous, aligned block of physical cores
                    result = find_contiguous_cores(available, num_physical_needed, total_physical_cores)
                    if result:
                        allocated_physical = result
                        state["in_use"].extend(allocated_physical)
                        state.setdefault("owners", {})[str(pid)] = allocated_physical
                        self._write_core_state(state)
                        break

                logging.info(
                    f"Waiting for {num_physical_needed} physical cores (lnc{lnc_config}), "
                    f"{len(available)} available. Retrying in {poll_period_seconds}s..."
                )
                time.sleep(poll_period_seconds)
            else:
                raise TimeoutException(
                    f"Timed out waiting for {num_physical_needed} physical cores on {self.host_id} "
                    f"after {timeout_seconds}s"
                )

        # Convert physical core IDs to logical core IDs (same logic as CoreLockManager)
        logical_cores = CoreLockManager._physical_to_logical_cores(allocated_physical, lnc_config)

        try:
            yield CoreAllocation(host_id=self.host_id, logical_core_ids=logical_cores, lnc_config=lnc_config)
        finally:
            with lock.acquire():
                state = self._read_core_state(total_physical_cores)
                for core in allocated_physical:
                    if core in state["in_use"]:
                        state["in_use"].remove(core)
                state.get("owners", {}).pop(str(pid), None)
                self._write_core_state(state)

    @staticmethod
    def _purge_stale_owners(state: dict) -> None:
        """Remove core reservations from PIDs that no longer exist."""
        owners = state.get("owners", {})
        stale_pids = []
        for pid_str, cores in owners.items():
            try:
                os.kill(int(pid_str), 0)
            except OSError:
                stale_pids.append(pid_str)
        for pid_str in stale_pids:
            stale_cores = owners.pop(pid_str)
            for core in stale_cores:
                if core in state["in_use"]:
                    state["in_use"].remove(core)
            logging.warning(f"Purged stale core locks from dead PID {pid_str}: cores {stale_cores}")

    def _read_core_state(self, total_physical_cores: int) -> dict:
        """Read or initialize the local core allocation state file."""
        if os.path.exists(self._core_state_path):
            with open(self._core_state_path, "r") as f:
                return json.load(f)
        return {"total_physical_cores": total_physical_cores, "in_use": []}

    def _write_core_state(self, state: dict) -> None:
        """Write the local core allocation state file."""
        with open(self._core_state_path, "w") as f:
            json.dump(state, f)

    @override
    def get_neuron_device_info(self) -> list[NeuronDeviceInfo]:
        result = subprocess.run([self.neuron_ls_path, "--json-output"], capture_output=True, text=True)
        data = json.loads(result.stdout)
        if not data:
            raise NoNeuronDevicesException("localhost")
        return [NeuronDeviceInfo.from_dict(device) for device in data]


@final
class SshHost(Host):
    SSH_CONNECT_TIMEOUT_SECONDS = 10

    def __init__(
        self,
        ssh_alias: str,
        test_base_path: str,
        remote_neuron_install_dir: str,
        run_id: str,
        ssh_config_path: str,
        s3_config: S3ArtifactUploadConfig,
        remote_base_path: str = "/tmp/neuronx-cc/tests",
    ):
        super().__init__()
        self.ssh_alias: str = ssh_alias
        self.run_id: str = run_id
        self.s3_config = s3_config

        config_overrides = {"run": {"in_stream": False, "warn": True, "pty": True}}

        self.connection: fabric2.Connection = fabric2.Connection(
            host=ssh_alias,
            connect_timeout=SshHost.SSH_CONNECT_TIMEOUT_SECONDS,
            config=fabric2.Config(
                runtime_ssh_path=ssh_config_path,
                overrides=config_overrides,
            ),
        )
        self.remote_base_path: str = remote_base_path
        self.remote_full_path: str | None = None

        self.lock_file_path: str = os.path.join(test_base_path, self.ssh_alias)
        self._total_physical_cores: int | None = None
        self._remote_executor = None
        self._host_locking_version: int | None = None
        # Cached during one allocation attempt so a manager created by
        # soft_join_queue during artifact upload is reused (same entry_id) at
        # commit time. Bound to the owning collector and cleared on teardown so
        # it never bleeds across tests/attempts (see _ensure_core_lock_manager).
        self._core_lock_manager: CoreLockManager | None = None
        self.neuron_ls_path: str = os.path.join(remote_neuron_install_dir, "neuron-ls")
        # Cached for the lifetime of this SshHost — device topology is assumed stable during a test run.
        self._cached_device_info: list[NeuronDeviceInfo] | None = None

    @override
    def get_host_id(self) -> str:
        return self.ssh_alias

    @override
    def _get_debug_output_dir(self, target_directory: str) -> str:
        return "$(pwd)/debug_output"

    def _get_remote_executor(self) -> RemoteExecutor:
        """Lazily create a RemoteExecutor for this host's connection."""
        if self._remote_executor is None:
            self._remote_executor = RemoteExecutor(self.connection)
        return self._remote_executor

    def _reconnect(self):
        """Reconnect SSH and rebuild the remote executor."""
        self.connection.close()
        self.connection.open()
        self._remote_executor = RemoteExecutor(self.connection)

    @override
    def _run_command(
        self,
        command: str,
        target_directory: str,
        neuron_env: dict[str, str],
        collector: IMetricsCollector,
    ) -> str:
        assert self.remote_full_path is not None, "You have to prepare host first!"

        env_var = " && ".join(f'export {k}="{v}"' for k, v in neuron_env.items())
        full_command = self.inside_venv(f"set -o pipefail; cd {self.remote_full_path} && {env_var} && {command}")
        logging.info(f"Executing remote command: {full_command}")

        result: fabric2.Result = self.connection.run(full_command)

        if result.failed:
            # Log PATH on remote host to help debug missing tools issues
            path_result = self.connection.run("echo DIAGNOSTIC: PATH=$PATH", warn=True)
            logging.warning(
                f"Remote PATH on {self.ssh_alias}: {path_result.stdout.strip() if path_result.ok else 'FAILED TO GET PATH'}"
            )
            raise RemoteExecutionException(f"Unable to execute {command} in {self.remote_full_path}", result)

        return result.stdout

    @override
    def _run_post_lock_command(
        self,
        command: str,
        target_directory: str,
        collector: IMetricsCollector,
    ) -> str:
        assert self.remote_full_path is not None, "You have to prepare host first!"

        full_command = self.inside_venv(f"set -o pipefail; cd {self.remote_full_path} && {command}")
        logging.info(f"Executing remote post-lock command: {full_command}")

        result: fabric2.Result = self.connection.run(full_command)

        if result.failed:
            raise RemoteExecutionException(f"Post-lock command failed in {self.remote_full_path}", result)

        return result.stdout

    @override
    def _should_run_post_lock(self) -> bool:
        if self._host_locking_version is None:
            return True
        try:
            return not lock_client.is_draining(self._get_remote_executor(), self._host_locking_version)
        except Exception as e:
            logging.warning(f"[{self.ssh_alias}] Failed to check drain state: {e}")
            return True

    @override
    def _collect_artifacts(
        self,
        target_directory: str,
        stdout: str,
        get_list_of_files_to_copy: Callable[[str], list[str]] | None,
        collector: IMetricsCollector,
    ) -> str:
        assert self.remote_full_path is not None, "You have to prepare host first!"

        local_download_location = os.path.join(target_directory, INF_ARTIFACT_DIR_NAME)
        # Clean up any existing infer_result directory from previous failed/retry attempts
        shutil.rmtree(local_download_location, ignore_errors=True)

        return self.__download_artifacts__(
            remote_path=self.remote_full_path,
            local_path=local_download_location,
            list_of_files_to_copy=(get_list_of_files_to_copy(stdout) if get_list_of_files_to_copy else None),
            collector=collector,
        )

    def __download_artifacts__(
        self,
        remote_path: str,
        local_path: str,
        collector: IMetricsCollector,
        list_of_files_to_copy: list[str] | None = None,
    ):
        with collector.timer(MetricName.FILE_TRANSFER_DOWNLOAD_TIME):
            return RemoteDirectory(remote_path, self.connection).download(
                destination_dir_path=local_path,
                s3_config=self.s3_config,
                list_of_files=list_of_files_to_copy,
                collector=collector,
            )

    def __cleanup_remote_paths__(self, *remote_path_list: str, base_exception: Exception | None = None):
        exceptions: list[Exception] = [base_exception] if base_exception else []
        for remote_path in remote_path_list:
            try:
                self.connection.run(f"rm -rf {remote_path}")
            except Exception as e:
                exceptions.append(e)

        if len(exceptions) > 0:
            raise Exception(*exceptions)

    def __cleanup_local_paths__(self, *local_path_list: str, base_exception: Exception | None = None):
        exceptions: list[Exception] = [base_exception] if base_exception else []
        for local_path in local_path_list:
            command_result = subprocess.run(["rm", "-rf", local_path])
            if command_result.returncode != 0:
                exceptions.append(LocalExecutionException(f"Unable to delete {local_path}", command_result))

        if len(exceptions) > 0:
            raise Exception(*exceptions)

    @override
    @contextlib.contextmanager
    def prepare_host(
        self,
        target_directory: str,
        collector: IMetricsCollector,
        skip_remote_cleanup: bool = False,
        force_local_cleanup: bool = False,
    ):
        # Add PID to make remote path unique per process
        # allows multiple machines to run the same test on the same host
        pid = os.getpid()
        test_base_dir = f"{os.path.basename(target_directory)}_pid{pid}"
        remote_full_path = os.path.join(self.remote_base_path, test_base_dir)

        remote_dir = RemoteDirectory(remote_full_path, self.connection)

        with collector.timer(MetricName.FILE_TRANSFER_UPLOAD_TIME):
            remote_dir.upload(target_directory, collector, self.s3_config, force_local_cleanup=force_local_cleanup)

        self.__install_prerequisites__(remote_path=remote_full_path)

        self.remote_full_path = remote_full_path

        try:
            yield remote_full_path
        finally:
            if not skip_remote_cleanup:
                remote_dir.cleanup()

    def __install_prerequisites__(self, remote_path: str):
        result: fabric2.Result = self.connection.run(f"python3 -m venv {remote_path}/.venv")
        if result.failed:
            raise RemoteExecutionException(
                f"Unable to initialize python virtual env at {remote_path}/.venv",
                result,
            )

    def inside_venv(self, command):
        assert self.remote_full_path
        return f"source {self.remote_full_path}/.venv/bin/activate && {command}"

    def __run_with_retry__(
        self, command: str, max_retries: int = 5, base_delay: float = 1.0, hide: bool = False
    ) -> fabric2.Result:
        """
        Execute SSH command with exponential backoff retry logic to avoid SSH rate limiting.
        """
        import random

        for attempt in range(max_retries):
            try:
                logging.info(f"Executing remote command (attempt {attempt + 1}): {command}")
                result = self.connection.run(command, hide=hide)
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                if isinstance(e, (SSHException, OSError)):
                    logging.warning(f"SSH connection error (attempt {attempt + 1}): {e}. Reconnecting...")
                    try:
                        self._reconnect()
                    except Exception:
                        pass
                else:
                    logging.warning(f"Non-SSH error (attempt {attempt + 1}): {e}. Retrying...")
                # Exponential backoff with jitter
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                time.sleep(delay)

        raise Exception("Retry logic failed unexpectedly")

    def _ensure_core_lock_manager(self, collector: IMetricsCollector) -> CoreLockManager:
        """Create or return a CoreLockManager bound to the given collector.

        Resolves the host's physical-core count and locking version (creating
        infra_version.json with the default if missing), then returns a manager.
        The cached manager is reused ONLY while it belongs to the same
        ``collector`` (the per-test/per-attempt owner): a manager created by
        ``soft_join_queue`` during artifact upload is the SAME instance
        ``get_core_allocation`` later commits with, so the FIFO slot anchored at
        soft-join time is reused (same ``entry_id``). When the collector differs
        (a new test/attempt) the cache is rebuilt, which re-mints ``entry_id``,
        rebinds the collector, and resets all fairness counters — making
        cross-test metric bleed impossible.
        """
        if self._core_lock_manager is not None and self._core_lock_manager.collector is collector:
            return self._core_lock_manager

        if self._total_physical_cores is None:
            devices = self.get_neuron_device_info()
            self._total_physical_cores = sum(len(d.neuroncore_ids) * d.logical_neuroncore_config for d in devices)
        if self._host_locking_version is None:
            executor = self._get_remote_executor()
            with collector.timer(MetricName.CORE_LOCK_INIT_TIME):
                self._host_locking_version = lock_client.initialize_and_deploy(executor)
            if self._host_locking_version > lock_client.DEFAULT_LOCKING_PROTOCOL_VERSION:
                raise LockVersionError(
                    required_version=self._host_locking_version,
                    current_version=lock_client.DEFAULT_LOCKING_PROTOCOL_VERSION,
                )
        logging.info(f"[{self.ssh_alias}] Using locking protocol v{self._host_locking_version}")

        self._core_lock_manager = CoreLockManager(
            self.ssh_alias,
            self._total_physical_cores,
            collector,
            executor=self._get_remote_executor(),
            host_locking_version=self._host_locking_version,
        )
        return self._core_lock_manager

    @override
    def soft_join_queue(
        self,
        collector: IMetricsCollector,
        collective_ranks: int = 1,
        lnc_config: int = 2,
    ) -> None:
        """Reserve a FIFO core-queue slot before/while artifacts upload.

        Synchronous pre-enqueue (no background threads): construct/cache the
        ``CoreLockManager`` and perform ONE ``ready=False`` poll so this attempt
        anchors its FIFO position at soft-join time rather than paying the
        upload latency. This ONLY reserves the slot — the real commit happens
        later in ``get_core_allocation``, which reuses the SAME cached
        manager/``entry_id``. If the slot is pruned/bumped during a long upload,
        the first ``ready=True`` poll transparently re-enqueues at tail (handled
        by the ``poll`` verb), so no resume code is needed here.

        Best-effort poll: a genuine soft-join (poll) failure is logged at
        warning and swallowed; normal acquisition in ``get_core_allocation``
        still works. However, this is NOT a pure no-op on control flow: if the
        just-reserved slot already has a worst-case ETA exceeding patience, the
        slot is freed and ``QueuePatienceRotation`` is raised so the surrounding
        host-assignment retry rotates to another host BEFORE paying the upload
        cost. No contention metrics are flushed on this early rotation because no
        queue-wait has accrued yet at soft-join time.
        """
        outcome = None
        manager = None
        try:
            manager = self._ensure_core_lock_manager(collector)
            outcome = manager.acquire(collective_ranks, lnc_config, ready=False)
            logging.info(
                f"[{self.ssh_alias}] Soft-joined core queue (ready=False): "
                f"status={outcome.status} position={outcome.position}"
            )
        except Exception as e:  # noqa: BLE001
            logging.warning(f"[{self.ssh_alias}] Best-effort soft-join failed: {e}")
            outcome = None  # genuine poll failure: swallow and let get_core_allocation run normally

        # Patience gate — OUTSIDE the best-effort swallow so it propagates (the
        # whole point of pre-upload rotation). A successful poll that landed an
        # over-patience QUEUED slot triggers rotation; DRAINING is suppressed
        # inside should_rotate (ETA is meaningless mid-drain, so it returns
        # False and the caller stays and keeps probing).
        if (
            outcome is not None
            and outcome.status in (AllocationStatus.QUEUED, AllocationStatus.DRAINING)
            and should_rotate(
                outcome.worst_case_eta,
                draining=(outcome.status == AllocationStatus.DRAINING),
            )
        ):
            try:
                manager.dequeue()  # free the just-joined slot (TTL prune is the safety net)
            except Exception as dq_err:  # noqa: BLE001
                logging.warning(f"[{self.ssh_alias}] Best-effort dequeue failed: {dq_err}")
            # Drop the cached manager so the next attempt — even a re-selection
            # of THIS host (a patience rotation does not mark it failed) —
            # re-mints a fresh manager (new entry_id + reset fairness counters)
            # instead of inheriting this abandoned soft-join's queue-wait.
            # Mirrors the per-attempt teardown in get_core_allocation.
            self._core_lock_manager = None
            raise QueuePatienceRotation(
                f"[{self.ssh_alias}] Queue ETA {outcome.worst_case_eta}s exceeds "
                f"patience {PATIENCE_SECONDS}s, rotating host"
            )

    @override
    @contextlib.contextmanager
    def get_core_allocation(
        self,
        collector: IMetricsCollector,
        collective_ranks: int = 1,
        lnc_config: int = 2,
        timeout_seconds: int = 9000,
        poll_period_seconds: int = POLL_PERIOD,
    ) -> Generator[CoreAllocation, None, None]:
        """
        Allocate logical cores for execution by locking physical cores.

        Physical cores are locked to prevent conflicts between LNC1 and LNC2 tests.
        Logical core IDs are returned for use with NEURON_RT_VISIBLE_CORES.
        """
        collector.add_dimension({MetricName.INSTANCE_TYPE: self.get_instance_type()})

        # Reuse the cached manager (and its entry_id) if soft_join_queue already
        # created one during artifact upload; otherwise build it now.
        core_lock_manager = self._ensure_core_lock_manager(collector)

        with collector.timer(MetricName.CORE_ALLOCATION_TIME):
            logging.info(
                f"[{self.ssh_alias}] Trying to acquire {collective_ranks} logical cores (lnc_config={lnc_config})"
            )

            # Poll until we get cores or timeout
            max_retryable_errors = 10
            # Liveness invariant: STALE_THRESHOLD must cover the poll loop's full
            # retry budget *including jitter* so a caller actively retrying
            # through transient SSH disruption is never pruned out from under
            # itself mid-budget. The real inter-refresh gap is POLL_PERIOD +
            # up to POLL_JITTER_MAX, so the budget is max_retryable_errors
            # refreshes at that spacing.
            assert STALE_THRESHOLD >= max_retryable_errors * (POLL_PERIOD + POLL_JITTER_MAX)
            consecutive_retryable_errors = 0
            start_time = time.time()
            outcome = None

            def _dequeue_best_effort() -> None:
                # Free the abandoned host's FIFO slot. Best-effort: a failed
                # dequeue is non-fatal (TTL prune is the crash safety net).
                try:
                    core_lock_manager.dequeue()
                except Exception as dq_err:  # noqa: BLE001
                    logging.warning(f"[{self.ssh_alias}] Best-effort dequeue failed: {dq_err}")

            def _abandon_attempt() -> None:
                # Terminal abandon teardown. Flush this attempt's queue-wait /
                # contention / drain-wait metrics BEFORE freeing the slot so the
                # starved attempts the observability was built to measure are not
                # invisible (success and abandon are mutually exclusive within an
                # attempt, so this never double-counts the success path's flush).
                core_lock_manager.record_contention_metrics()
                _dequeue_best_effort()

            while time.time() - start_time < timeout_seconds:
                try:
                    # Artifacts are uploaded before acquisition in the current
                    # ordering, so the caller is always ready to commit.
                    outcome = core_lock_manager.acquire(
                        collective_ranks,
                        lnc_config,
                        timeout_seconds=lock_client.INFERENCE_LOCK_TIMEOUT_SECONDS,
                        ready=True,
                    )
                    consecutive_retryable_errors = 0
                    if outcome.status == AllocationStatus.ALLOCATED:
                        break
                    if outcome.status == AllocationStatus.DRAINING:
                        # Host draining: stay-and-probe. Keep polling
                        # the same host without counting this as a retryable error
                        # and without aborting. Drains are typically fleet-wide, so
                        # rotating off would just land on another draining host, so
                        # position is preserved organically as last_seen_ts keeps
                        # refreshing.
                        logging.info(
                            f"[{self.ssh_alias}] Host draining; continuing to poll (position={outcome.position})"
                        )
                    # QUEUED / DRAINING: no cores yet, keep polling.
                except (LockAcquisitionError, LockVersionError) as e:
                    logging.warning(f"[{self.ssh_alias}] Lock error (retryable={e.retryable}): {e}")
                    if not e.retryable:
                        raise
                    consecutive_retryable_errors += 1
                    # An error is not evidence the caller is near the front, so
                    # discard any stale near-front position from a prior
                    # successful poll; the cadence below must fall back to the
                    # SLOW base (select_poll_base(None) -> slow) instead of
                    # fast-spinning and burning the retry budget early.
                    outcome = None
                    if consecutive_retryable_errors >= max_retryable_errors:
                        _abandon_attempt()
                        raise OSError(
                            f"[{self.ssh_alias}] Lock acquisition failed after {max_retryable_errors} "
                            f"consecutive retryable errors. Last error: {e}"
                        ) from e
                base = select_poll_base(
                    outcome.position if outcome is not None else None,
                    slow=poll_period_seconds,
                )
                jitter = random.uniform(0, POLL_JITTER_MAX)
                time.sleep(base + jitter)

            if outcome is None or outcome.status != AllocationStatus.ALLOCATED:
                _abandon_attempt()
                raise TimeoutException(
                    f"[{self.ssh_alias}] Unable to allocate {collective_ranks} logical cores within {timeout_seconds}s"
                )

            logical_cores = outcome.logical_cores
            physical_cores = outcome.physical_cores
            core_lock_manager.record_contention_metrics()
            logging.info(f"[{self.ssh_alias}] Allocated logical cores {logical_cores} (physical: {physical_cores})")

        try:
            yield CoreAllocation(host_id=self.ssh_alias, logical_core_ids=logical_cores, lnc_config=lnc_config)
        finally:
            core_lock_manager.release(physical_cores)
            # Drop the cached manager so the next attempt always starts fresh
            # (re-minted entry_id, rebound collector, reset counters) even if it
            # skips soft-join. The collector-identity guard in
            # _ensure_core_lock_manager already prevents cross-test reuse; this
            # is defense in depth and a clean per-attempt teardown hook.
            self._core_lock_manager = None

    @override
    def get_neuron_device_info(self) -> list[NeuronDeviceInfo]:
        """Get Neuron device information from remote host with retry logic. Cached after first call."""
        if self._cached_device_info is not None:
            return list(self._cached_device_info)
        result: fabric2.Result = self.__run_with_retry__(
            f"NEURON_LOGICAL_NC_CONFIG={NeuronDeviceInfo.logical_neuroncore_config} {self.neuron_ls_path} --json-output",
            hide=True,
        )
        if result.failed:
            raise RemoteExecutionException(f"Unable to find neuron device on {self.ssh_alias}", result)
        logging.debug("neuron-ls output: %s", result.stdout)
        data = json.loads(result.stdout)
        if not data:
            raise NoNeuronDevicesException(self.ssh_alias)
        self._cached_device_info = [NeuronDeviceInfo.from_dict(device) for device in data]
        return self._cached_device_info

    def get_instance_type(self) -> str:
        return self.get_neuron_device_info()[0].instance_type


@dataclass
class HostInfo:
    host_alias: str
    work_queue_depth: int
    run_id: str
    host_type: str | None

    def to_json(self):
        return {
            "host_alias": self.host_alias,
            "work_queue_depth": self.work_queue_depth,
            "run_id": self.run_id,
            "host_type": self.host_type,
        }

    @classmethod
    def from_json(cls, input: Any):
        return HostInfo(
            input["host_alias"],
            input["work_queue_depth"],
            input.get("run_id", ""),
            input.get("host_type", ""),
        )


@final
class HostManager:
    def __init__(
        self,
        base_host_info_path: str,
        target_hosts: list[TargetHost],
        neuron_installation_path: str,
        ssh_config_path: str,
        s3_config: S3ArtifactUploadConfig | None = None,
    ) -> None:
        self.run_id = str(os.getppid())
        self.ssh_config_path = ssh_config_path
        self.s3_config = s3_config or S3ArtifactUploadConfig()
        self.target_hosts, self.host_types = self.__derive_hosts__(
            target_hosts,
            neuron_installation_path=neuron_installation_path,
            base_host_info_path=base_host_info_path,
        )
        self.is_local: bool = len(target_hosts) < 1
        self.host_info_path: str = os.path.join(base_host_info_path, "host_stats.json")
        self.failed_hosts: set[str] = set()  # Track hosts that have timed out

    def __derive_hosts__(
        self,
        target_hosts: list[TargetHost],
        neuron_installation_path: str,
        base_host_info_path: str,
    ) -> tuple[dict[str, Host], dict[str, Platforms | None]]:
        if len(target_hosts) == 0:
            local_host_id = "localhost"
            detected_platform = detect_local_platform(neuron_installation_path)
            if detected_platform:
                logging.info(f"Auto-detected local platform: {detected_platform.value}")
            return (
                {local_host_id: LocalHost(neuron_installation_path, local_host_id, base_host_info_path)},
                {local_host_id: detected_platform},
            )
        else:
            hosts: dict[str, Host] = dict()
            host_types: dict[str, Platforms] = dict()

            for target_host in target_hosts:
                # Note: ssh_alias format must match fetch_shared_fleet_metadata.sh which has to derive the same alias from the JSON
                ssh_alias = target_host.ssh_host
                hosts[ssh_alias] = SshHost(
                    ssh_alias,
                    test_base_path=base_host_info_path,
                    remote_neuron_install_dir=neuron_installation_path,
                    run_id=self.run_id,
                    ssh_config_path=self.ssh_config_path,
                    s3_config=self.s3_config,
                )
                host_types[ssh_alias] = target_host.host_type

            return hosts, host_types

    def __construct_lock_file_name(self):
        return f"{self.host_info_path}.lock"

    def __create_lockfile__(self, timeout_seconds: int):
        return FileLock(self.__construct_lock_file_name(), timeout=timeout_seconds * 1000)

    @contextlib.contextmanager
    def __read_host_file__(self) -> Generator[list[HostInfo], list[HostInfo], None]:
        with self.__create_lockfile__(10).acquire():
            with open(self.host_info_path, "r+") as fp:
                j = json.load(fp)

                assert isinstance(j, list)

                hosts = [HostInfo.from_json(json_host_info) for json_host_info in j]

                yield hosts

                # Seek to beginning and truncate before writing
                _ = fp.seek(0)
                _ = fp.truncate()
                # Convert HostInfo objects to dictionaries for JSON serialization
                json.dump(
                    [h.to_json() for h in hosts],
                    fp,
                )

    def initialize_host_stats(self):
        # Each pytest worker tries to create this file, but only one needs to.
        # All others use the file created by the first to reach this point.
        # If file exists from a previous run (different run_id), reset all work_queue_depth to 0.
        with self.__create_lockfile__(10).acquire():
            if not os.path.isfile(self.host_info_path):
                with open(self.host_info_path, "w+") as fp:
                    hosts_info = [
                        HostInfo(
                            host_alias,
                            work_queue_depth=0,
                            run_id=self.run_id,
                            host_type=pt.value if (pt := self.host_types.get(host_alias)) else None,
                        )
                        for host_alias in self.target_hosts.keys()
                    ]
                    # randomly shuffle hosts, so that different test suites don't always hammer the
                    # same hosts first
                    random.shuffle(hosts_info)
                    json.dump([h.to_json() for h in hosts_info], fp)
            else:
                # File exists - validate hosts and check run_id
                with open(self.host_info_path, "r+") as fp:
                    existing_host_infos = [HostInfo.from_json(h) for h in json.load(fp)]
                    existing_hosts = {h.host_alias for h in existing_host_infos}
                    current_hosts = set(self.target_hosts.keys())

                    # Check if run_id matches
                    needs_reset = False
                    if existing_host_infos and existing_host_infos[0].run_id != self.run_id:
                        needs_reset = True

                    # Reset if hosts changed or stale run_id
                    if existing_hosts != current_hosts or needs_reset:
                        _ = fp.seek(0)
                        _ = fp.truncate()
                        hosts_info = [
                            HostInfo(
                                host_alias,
                                work_queue_depth=0,
                                run_id=self.run_id,
                                host_type=pt.value if (pt := self.host_types.get(host_alias)) else None,
                            )
                            for host_alias in current_hosts
                        ]
                        json.dump([h.to_json() for h in hosts_info], fp)

    def mark_host_as_failed(self, host_id: str):
        """Mark a host as failed to exclude it from future assignments."""
        self.failed_hosts.add(host_id)
        logging.warning(
            f"Host {host_id} marked as failed. Failed hosts: {len(self.failed_hosts)}/{len(self.target_hosts)}"
        )

    def get_failed_host_count(self) -> int:
        """Get the number of hosts that have failed during this run."""
        return len(self.failed_hosts)

    def __get_host_assignment__(self, platform_target: Platforms, timeout_seconds: int = 10) -> Host:
        generator = self.__read_host_file__()
        with generator as hosts:
            # Filter out failed hosts and hosts that don't match the platform target
            available_hosts = [
                h for h in hosts if h.host_alias not in self.failed_hosts and h.host_type == platform_target.value
            ]

            if not available_hosts:
                matching_hosts = [h for h in hosts if h.host_type == platform_target.value]
                if not matching_hosts:
                    raise Exception(f"No hosts available for platform {platform_target.value}")
                raise Exception(
                    f"No available hosts for platform {platform_target.value} - "
                    f"all {len(matching_hosts)} matching hosts have failed"
                )

            asc_host_infos = sorted(
                available_hosts,
                key=lambda host: host.work_queue_depth,
            )

            # Pick randomly from top 3 hosts with lowest queue depth
            top_n = min(3, len(asc_host_infos))
            with temporary_random_seed(time.time_ns()):
                selected_host_info = random.choice(asc_host_infos[:top_n])

            selected_host_info.work_queue_depth += 1
            host = self.target_hosts[selected_host_info.host_alias]

            return host

    def release_host(self, host: Host):
        host_id = host.get_host_id()

        with self.__read_host_file__() as hosts:
            for h in hosts:
                if h.host_alias == host_id:
                    h.work_queue_depth -= 1
                    break

    def get_host_assignment_with_retry(
        self,
        platform_target: Platforms,
        collector: IMetricsCollector,
        *,
        deadline_seconds: float = DEFAULT_ACQUISITION_DEADLINE_SECONDS,
        connection_failure_cap: int = 3,
        backoff_seconds: float = HOST_ROTATION_BACKOFF_SECONDS,
    ):
        """Execute a function with automatic retry on different hosts.

        A busy FIFO queue (``QueuePatienceRotation``) is transient: the host is NOT
        marked failed and stays eligible, the caller backs off briefly and keeps
        rotating across hosts until ``deadline_seconds`` elapses -- it never
        self-terminates merely because all matching hosts are currently busy.
        Genuine connection failures keep the prior behavior: mark the host failed,
        exclude it, and stop after ``connection_failure_cap`` of them.
        """
        # Track errors from each host attempt for better debugging
        host_errors: dict[str, str] = {}

        def format_host_errors() -> str:
            """Format all captured host errors for the exception message."""
            return "\n".join(f"  - {host}: {error}" for host, error in host_errors.items())

        # in case code block that's yielded to by context_manager_wrapper does not directly return
        # make sure that we record successes and terminate retries
        success = False
        # Genuine connection failures for THIS allocation; bounded by connection_failure_cap.
        connection_failures = 0
        # Transient busy-queue rotations across hosts for THIS allocation (no failure).
        rotation_count = 0

        def succeeded():
            nonlocal success
            success = True

        @contextlib.contextmanager
        def context_manager_wrapper(notify_success: Callable[[], None], execution_host: Host):
            nonlocal connection_failures, rotation_count
            try:
                yield execution_host
            except QueuePatienceRotation as e:
                # A busy FIFO queue is transient, NOT a host failure: do not mark the
                # host failed, do not count it toward the connection-failure cap, and
                # leave it eligible for re-selection. Record the rotation metric and
                # back off so a fully-busy fleet re-cycles at a bounded rate.
                host_id = execution_host.get_host_id() if execution_host else "unknown"
                rotation_count += 1
                logging.info(f"Host {host_id} queue busy (patience rotation #{rotation_count}); rotating: {e}")
                if collector:
                    collector.record_metric(
                        MetricName.CORE_LOCK_HOST_ROTATION_COUNT,
                        1.0,
                        "Count",
                    )
                time.sleep(backoff_seconds)
            except (OSError, TimeoutError, SSHException, TimeoutException) as e:
                host_id = execution_host.get_host_id() if execution_host else "unknown"
                error_msg = f"{type(e).__name__}: {e}"
                host_errors[host_id] = error_msg

                connection_failures += 1
                logging.error(
                    f"Connection error on host {host_id}, failure {connection_failures}/{connection_failure_cap}: {e}"
                )
                self.mark_host_as_failed(host_id)

                if collector:
                    # Emit one datapoint of 1.0 per rotation (a delta) so that
                    # sum/avg aggregations over EMF datapoints reflect the true
                    # rotation count instead of over-counting the cumulative value.
                    collector.record_metric(
                        MetricName.CORE_LOCK_HOST_ROTATION_COUNT,
                        1.0,
                        "Count",
                    )
                    collector.record_metric(
                        MetricName.FAILED_HOSTS_COUNT,
                        float(self.get_failed_host_count()),
                        "Count",
                    )

                if connection_failures >= connection_failure_cap:
                    error_details = format_host_errors()
                    raise InferenceException(
                        f"Connection error after {connection_failures} attempts. "
                        f"Hosts attempted: {', '.join(attempted_hosts)}\n"
                        f"Errors from each host:\n{error_details}"
                    ) from e

                logging.warning("Retrying on a different host")
            else:
                notify_success()

        attempted_hosts = []

        deadline = time.time() + deadline_seconds
        while not success and time.time() < deadline:
            execution_host = None

            try:
                if collector:
                    with collector.timer(MetricName.HOST_LOCK_TIME):
                        execution_host = self.__get_host_assignment__(platform_target)
                else:
                    execution_host = self.__get_host_assignment__(platform_target)

                host_id = execution_host.get_host_id()
                attempted_hosts.append(host_id)

                yield context_manager_wrapper(succeeded, execution_host)

            except Exception as e:
                # Report error with details from previous failures
                if host_errors and not isinstance(e, InferenceException):
                    error_details = format_host_errors()
                    raise InferenceException(f"{e}\n\nErrors from previous host attempts:\n{error_details}") from e
                raise
            finally:
                if execution_host:
                    self.release_host(execution_host)

        if not success:
            # Distinct from "No available hosts" and "Connection error after N attempts":
            # the fleet was reachable but every matching host stayed busy past the deadline.
            raise InferenceException(
                f"Core allocation deadline ({deadline_seconds}s) exceeded after rotating across hosts; "
                f"all matching hosts remained busy. "
                f"Patience rotations: {rotation_count}. Hosts attempted: {', '.join(attempted_hosts)}"
            )
