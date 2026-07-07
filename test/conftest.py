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
"""nkilib internal test configuration — private behavior on top of the shared plugin.

The nkilib_testing pytest plugin (auto-registered via pytest11) provides shared CLI
options, fixtures, and hooks.  This conftest adds nkilib-internal behavior only.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pytest_timeout
from _pytest.config import Config

from .utils import cpu_timeout

if TYPE_CHECKING:
    from .utils.test_orchestrator import Orchestrator

# Set consistent hash seed for xdist workers to ensure identical test collection
_RNG_SEED_ENV_KEY = "NEURON_PYTHONHASHSEED"
if "PYTEST_XDIST_WORKER" in os.environ and _RNG_SEED_ENV_KEY not in os.environ:
    os.environ[_RNG_SEED_ENV_KEY] = "0"


import pytest
import pytest_html.extras

from .utils.artifact_manager import (
    UploadOutcome,
    should_upload_artifacts,
    upload_test_artifacts_to_s3,
    validate_s3_credentials,
)
from .utils.common_dataclasses import (
    NKICompilationMode,
    Platforms,
    TargetHost,
    TraceMode,
)
from .utils.composite_emitter import CompositeEmitter
from .utils.coverage_utils import get_coverage_data
from .utils.csv_monitor import DURATION_MONITOR, MEMORY_MONITOR
from .utils.feature_flag_helper import (
    construct_test_output_directory_name,
    get_feature_flag,
    resolve_base_output_directory,
)
from .utils.host_management import HostManager
from .utils.metrics_collector import IMetricsCollector
from .utils.metrics_emitter import IMetricsEmitter, OutputMode, SessionContext
from .utils.pytest_plugin import (
    get_platform_targets,
    make_collector,
    make_emitter,
    make_host_manager,
    make_test_manager,
    resolve_current_user,
    resolve_session_trace_mode,
)
from .utils.pytest_test_metadata import derive_labeled_kernel_name
from .utils.qor_collector import collect_qor_from_test_dir
from .utils.s3_utils import S3ArtifactUploadConfig, prefetch_and_cache_credentials
from .utils.sqs_emitter import SQSEmitter

# Key for storing relevant test directories on pytest config (used by --run-relevant-tests)
RELEVANT_TEST_DIRS_KEY = "_relevant_test_dirs"


def pytest_addoption(parser):
    group = parser.getgroup("nkilib-internal", "nkilib internal options")
    group.addoption(
        "--target-host-file",
        action="store",
        default=None,
        help="Path to JSON file containing MLA accelerator host definitions.  This is used in place of --target-host if the set of hosts are not homogeneous with respect to host type.",
    )
    group.addoption(
        "--metrics-namespace",
        action="store",
        default="NeuronCompiler",
        help="CloudWatch namespace for metrics. Default: NeuronCompiler",
    )
    group.addoption(
        "--sqs-queue-url",
        action="store",
        default=None,
        help="SQS Standard queue URL for metrics ingestion (enables OpenSearch storage)",
    )

    group.addoption(
        "--disable-no-tests-failure",
        action="store_true",
        default=False,
        help="Log warning instead of failing (exit code 5) when no tests are collected",
    )

    parser.addini(
        "artifacts_dir",
        type="string",
        help="Path where supporting artifacts are going to be stored e.g. test reports, log files etc.",
    )
    group.addoption(
        "--test-output-s3-bucket",
        action="store",
        default="",
        help="S3 bucket name to upload test output artifacts (e.g., my-bucket-name)",
    )

    group.addoption(
        "--test-output-s3-prefix",
        action="store",
        default="",
        help="S3 prefix/path within bucket for test output artifacts (e.g., test-artifacts/)",
    )

    group.addoption(
        "--upload-test-outcomes",
        action="store",
        default="",
        choices=[""] + [e.value for e in UploadOutcome],
        help=f"Which test outcomes to upload to S3: {', '.join(repr(e.value) for e in UploadOutcome)}. Requires --test-output-s3-bucket",
    )

    group.addoption(
        "--run-relevant-tests",
        action="store",
        nargs="?",
        const="HEAD",
        default=None,
        help="Run only tests relevant to changes in the specified commit(s). "
        "Accepts a single commit or comma-separated list (default: HEAD)",
    )
    group.addoption(
        "--monitor-memory",
        type=int,
        nargs="?",
        const=20,
        default=None,
        help="Track per-test peak RSS and print the N slowest (default 20)",
    )
    group.addoption(
        "--memory-limit",
        type=float,
        default=None,
        help="Kill a test if its process tree RSS exceeds this many MB",
    )
    group.addoption(
        "--monitor-durations",
        type=int,
        nargs="?",
        const=20,
        default=None,
        help="Track per-test wall-clock duration and print the N slowest (default 20)",
    )
    group.addoption(
        "--cpu-timeout",
        type=float,
        default=None,
        help="CPU time budget in seconds (self + children). Override per-test with @pytest.mark.cpu_timeout(N).",
    )


@pytest.fixture(scope="session")
def artifacts_output_directory(request: pytest.FixtureRequest) -> str:
    artifacts_dir_path = str(request.config.getini("artifacts_dir"))
    Path(artifacts_dir_path).mkdir(exist_ok=True, parents=True)
    return artifacts_dir_path


@pytest.fixture(scope="session")
def test_worker_id() -> str:
    return os.environ.get("PYTEST_XDIST_WORKER", default="gw_master")


# Logging from xdist is very tricky, as workers can only write to stderr and ignore most of the logging configuration.
@pytest.fixture(scope="session", autouse=True)
def setup_logging(request: pytest.FixtureRequest, test_worker_id: str, artifacts_output_directory: str):
    log_dir: str = os.path.join(artifacts_output_directory, "log")
    Path(log_dir).mkdir(exist_ok=True)
    log_file: str = os.path.join(log_dir, f"pytest_{test_worker_id}.txt")

    # need to set up logger even for the 'default' thread.
    log_level: str = str(request.config.getini("log_level"))
    log_format = str(request.config.getini("log_format")).replace("__worker_id__", test_worker_id)
    log_date_format: str = str(request.config.getini("log_date_format"))

    # Create file handler to output logs into corresponding worker file
    file_handler = logging.FileHandler(log_file, mode="w", delay=True)
    file_handler.setFormatter(
        logging.Formatter(
            fmt=log_format,
            datefmt=log_date_format,
            style="%",
        )
    )

    # Create stream handler to output logs on console
    # This is a workaround for a known limitation:
    # https://pytest-xdist.readthedocs.io/en/latest/known-limitations.html
    console_handler = logging.StreamHandler(sys.stderr)  # pytest only prints error logs
    console_handler.setFormatter(
        logging.Formatter(
            fmt=log_format,
            datefmt=log_date_format,
            style="%",
        )
    )
    logging.basicConfig(
        handlers=[file_handler, console_handler],
        level=log_level,
        # critical here, as otherwise config doesn't update as the result of this call.
        force=True,
    )
    logging.info(f'Logging format in worker: {log_format=} {log_file=} {log_level=}')


@pytest.fixture(scope="session")
def host_manager(request: pytest.FixtureRequest) -> HostManager:
    platforms = get_platform_targets(request.config)
    target_host_file: str | None = get_feature_flag(request.config, "target_host_file")

    # Build target hosts list
    target_hosts: list[TargetHost] = []
    if target_host_file:
        # Load hosts from JSON file (includes host types, defaults to platform_target if missing)
        with open(target_host_file, "r") as f:
            data = json.load(f)
        for i, host in enumerate(data["sharedFleet"]):
            # sshHost takes precedence over publicIp.
            # TODO: Support for publicIp can be removed once we stop supporting static host lists in the pipeline.
            if "sshHost" in host:
                ssh_host = host["sshHost"]
            elif "publicIp" in host:
                ssh_host = host["publicIp"]
            else:
                raise ValueError(
                    f"Host entry {i} in {target_host_file} is missing required 'sshHost' or 'publicIp' field"
                )
            if "hostType" not in host:
                logging.warning(
                    "Host entry %d in %s is missing 'hostType', falling back to %s",
                    i,
                    target_host_file,
                    platforms[0].value,
                )
            host_type = Platforms(host["hostType"]) if "hostType" in host else platforms[0]
            target_hosts.append(
                TargetHost(
                    ssh_host=ssh_host,
                    host_type=host_type,
                )
            )

    # Filter out "dice" pseudo-host — DICE inference is handled via DICE_ENDPOINT env var
    target_hosts = [th for th in target_hosts if th.ssh_host != "dice"]

    # When no host file, let make_host_manager handle --target-host CLI (including single-platform assertion)
    if not target_host_file:
        return make_host_manager(request.config)

    return make_host_manager(request.config, target_hosts=target_hosts)


@pytest.fixture(scope="session")
def session_trace_mode(request: pytest.FixtureRequest) -> TraceMode:
    """Session-wide trace mode from CLI flags. Individual tests may override via markers."""
    # --target-host-file implies CompileAndInfer (internal option not known to plugin)
    test_mode = get_feature_flag(request.config, "test_mode")
    if test_mode:
        return TraceMode.create(test_mode)
    if get_feature_flag(request.config, "target_host_file"):
        return TraceMode.CompileAndInfer
    return resolve_session_trace_mode(request.config)


@pytest.fixture
def collector(request: pytest.FixtureRequest, metric_output_mode: OutputMode | None) -> IMetricsCollector:
    """Create metrics collector, delegating to shared make_collector with internal namespace config."""
    namespace = get_feature_flag(request.config, "metrics_namespace", default_value="NeuronCompiler")
    collector = make_collector(request, metric_output_mode, namespace=namespace)
    if metric_output_mode is not None:
        request.node._collector = collector
    return collector


@pytest.fixture
def emitter(
    request: pytest.FixtureRequest,
    metric_output_mode: OutputMode | None,
) -> IMetricsEmitter:
    """Create metrics emitter with SQS support for OpenSearch ingestion."""
    if metric_output_mode is None:
        emitter = make_emitter(metric_output_mode)
    else:
        session_ctx = getattr(request.config, "_session_context", None)
        emitters: list[IMetricsEmitter] = [make_emitter(metric_output_mode)]
        if session_ctx and session_ctx.sqs_queue_url:
            emitters.append(SQSEmitter(session=session_ctx))
        emitter = CompositeEmitter(emitters=emitters)

    request.node._emitter = emitter
    return emitter


@pytest.fixture
def test_manager(
    request: pytest.FixtureRequest,
    trace_mode: TraceMode,
    host_manager: HostManager,
    collector: IMetricsCollector,
    emitter: IMetricsEmitter,  # noqa: ARG001 — triggers fixture to stash on item for makereport hook
    perf_analysis_enabled: bool,
    hw_profile_enabled: bool,
) -> Orchestrator:
    metadata_name = None
    if request.cls and hasattr(request.cls, "__pytest_test_metadata__"):
        metadata_name = request.cls.__pytest_test_metadata__.get("name")
    kernel_name = derive_labeled_kernel_name(request.fspath, metadata_name)

    dice_endpoint = os.environ.get("DICE_ENDPOINT")
    if dice_endpoint:
        from .utils.container_service_private import DiceOrchestrator

        return DiceOrchestrator(
            request.config,
            trace_mode,
            host_manager,
            collector,
            perf_analysis_enabled=perf_analysis_enabled,
            hw_profile_enabled=hw_profile_enabled,
            kernel_name=kernel_name,
            dice_endpoint=dice_endpoint,
            nki_compilation_mode=NKICompilationMode[get_feature_flag(request.config, "nki_compilation_mode")],
        )

    return make_test_manager(
        request.config,
        trace_mode,
        host_manager,
        collector,
        perf_analysis_enabled=perf_analysis_enabled,
        hw_profile_enabled=hw_profile_enabled,
        kernel_name=kernel_name,
    )


@pytest.hookimpl(tryfirst=True)
def pytest_timeout_set_timer(item: pytest.Item, settings: pytest_timeout.Settings) -> bool | None:
    return cpu_timeout.prepare_timeout_watchdog(item, settings)


# Store S3 config for use in makereport hook (set in fixture before test runs)
_s3_upload_config: dict[str, tuple[S3ArtifactUploadConfig, str]] = {}


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test outcome and add S3 link to HTML report."""
    outcome = yield
    rep = outcome.get_result()

    # Store the report on the item for access in fixture teardown
    setattr(item, f"rep_{rep.when}", rep)

    # Add S3 link to report during "call" phase (when test outcome is known)
    if rep.when == "call":
        extra = getattr(rep, "extra", [])

        # Check if S3 upload is configured for this test
        if item.nodeid in _s3_upload_config:
            s3_config, upload_test_outcomes = _s3_upload_config[item.nodeid]
            test_outcome = "passed" if rep.passed else "failed"

            if should_upload_artifacts(test_outcome, upload_test_outcomes):
                # Get the test output directory
                test_dir_path = construct_test_output_directory_name()
                output_directory = resolve_base_output_directory(item.config)
                test_dir_full_path = os.path.join(output_directory, test_dir_path)

                s3_result = upload_test_artifacts_to_s3(test_dir_full_path, s3_config, item.name)
                if s3_result.s3_url:
                    extra.append(pytest_html.extras.url(s3_result.s3_url, name="S3 Artifacts"))
                # Store for potential use elsewhere
                item._s3_upload_result = s3_result

        rep.extra = extra

        # Emit test results for all outcomes (passed, failed, skipped)
        collector = getattr(item, "_collector", None)
        if collector:
            if rep.skipped:
                # For pytest.skip() calls, longrepr is a (filename, lineno, reason) tuple.
                # For other skip scenarios (e.g. skipIf, collection-level skips), it may be
                # a string or TerminalRepr. See: _pytest/runner.py pytest_runtest_makereport
                skip_reason = rep.longrepr[2] if isinstance(rep.longrepr, tuple) else str(rep.longrepr)
                collector.add_dimension(
                    {
                        "Status": "skipped",
                        "SkipReason": skip_reason,
                    }
                )
            elif rep.failed and "Status" not in collector.dimensions:
                # Status is already set by the orchestrator for compilation/inference/validation
                # failures. This fallback covers tests that fail before reaching the orchestrator
                # (e.g. assertion in test setup, fixture error, or pre-orchestrator validation).
                collector.add_dimension({"Status": "TEST_EXECUTION_FAILURE"})

            emitter = getattr(item, "_emitter", None)
            if emitter:
                emitter.emit(collector)


@pytest.fixture(autouse=True)
def run_after_every_test(
    request: pytest.FixtureRequest,
):
    # Store S3 config BEFORE test runs so makereport hook can use it
    upload_test_outcomes = get_feature_flag(request.config, "upload_test_outcomes")
    if upload_test_outcomes:
        s3_config = S3ArtifactUploadConfig(
            bucket=get_feature_flag(request.config, "test_output_s3_bucket"),
            prefix=get_feature_flag(request.config, "test_output_s3_prefix"),
            profile=get_feature_flag(request.config, "aws_profile"),
        )
        _s3_upload_config[request.node.nodeid] = (s3_config, upload_test_outcomes)

    try:
        # immediately yield as there is no setup needed
        yield
    finally:
        try:
            # regardless of the test outcome, we want to have a chance to clean up
            # below code is executed right after test has finished running
            test_dir_path = construct_test_output_directory_name()
            output_directory = resolve_base_output_directory(request.config)
            test_dir_full_path = os.path.join(output_directory, test_dir_path)

            # Collect QoR data BEFORE cleanup (only if test dir exists)
            if os.path.isdir(test_dir_full_path):
                # Get session ID from master or worker
                if hasattr(request.config, "_qor_session_id"):
                    session_id = request.config._qor_session_id
                elif hasattr(request.config, "workerinput"):
                    session_id = request.config.workerinput.get("qor_session_id")
                else:
                    session_id = None
                collect_qor_from_test_dir(test_dir_full_path, output_directory, session_id)

            # Cleanup
            force_cleanup: bool = get_feature_flag(request.config, "force_local_cleanup")

            if force_cleanup:
                preserve = set(get_feature_flag(request.config, "force_local_cleanup_keep") or [])
                if preserve and os.path.isdir(test_dir_full_path):
                    # Selectively delete, preserving specified artifact types
                    for item in os.listdir(test_dir_full_path):
                        if item not in preserve:
                            item_path = os.path.join(test_dir_full_path, item)
                            if os.path.isdir(item_path):
                                shutil.rmtree(item_path, ignore_errors=True)
                            else:
                                os.remove(item_path)
                else:
                    shutil.rmtree(test_dir_full_path, ignore_errors=True)
        except Exception as e:
            logging.warning(f"Received exception during teardown, skipping re-throw.\n {e}")


@pytest.fixture(autouse=True)
def _memory_monitor(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Track per-test peak RSS and enforce memory limits.

    --monitor-memory: record peak and delta RSS per test to memory_monitor.csv
    --memory-limit N: kill the test if its delta RSS exceeds N MB
    """
    from test.utils.memory_monitor import ProcessTreeMemoryMonitor

    monitor_memory = get_feature_flag(request.config, "monitor_memory")
    memory_limit_mb = request.config.getoption("memory_limit", default=None)

    if monitor_memory is None and memory_limit_mb is None:
        yield
        return

    limit_bytes = int(memory_limit_mb * 1024 * 1024) if memory_limit_mb is not None else None
    monitor = ProcessTreeMemoryMonitor(os.getpid(), memory_limit_bytes=limit_bytes)
    monitor.start()
    try:
        yield
    finally:
        snapshot = monitor.stop()

        if monitor_memory:
            output_dir = Path(resolve_base_output_directory(request.config))
            MEMORY_MONITOR.append(
                output_dir, f"{request.node.nodeid},{snapshot.peak_rss_mb:.1f},{snapshot.delta_rss_mb:.1f}"
            )


@pytest.fixture(autouse=True)
def _duration_monitor(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Track per-test CPU time and wall-clock duration across xdist workers.

    --monitor-durations N: record CPU time + wall time per test, print the N slowest
    (sorted by CPU time) in the summary.
    """
    import time

    top_n = get_feature_flag(request.config, "monitor_durations")
    if top_n is None:
        yield
        return

    start_wall = time.perf_counter()
    start_cpu = cpu_timeout._get_total_cpu()
    yield
    elapsed_wall = time.perf_counter() - start_wall
    elapsed_cpu = cpu_timeout._get_total_cpu() - start_cpu

    output_dir = Path(resolve_base_output_directory(request.config))
    DURATION_MONITOR.append(output_dir, f"{request.node.nodeid},{elapsed_cpu:.3f},{elapsed_wall:.3f}")


def pytest_ignore_collect(collection_path, config):
    """Skip collecting model_config modules directly when --skip-model-tests is active."""
    if config.getoption("--skip-model-tests", default=False):
        if "model_config" in collection_path.name:
            return True


def pytest_configure(config: Config):
    # Set env var early so model_config modules see it at import time during collection
    if get_feature_flag(config, "skip_model_tests", False):
        os.environ["SKIP_MODEL_TESTS"] = "1"
        logging.info("SKIP_MODEL_TESTS enabled: model config modules will not be loaded")

    sqs_queue_url = get_feature_flag(config, "sqs_queue_url")

    # Pre-fetch AWS credentials before xdist workers spawn to avoid Isengard rate limiting
    # This only runs in the main process; workers will inherit the env vars
    # Workers have 'workerinput' attribute set on config, master does not
    if not hasattr(config, "workerinput"):
        # Generate QoR session ID for this test run
        config._qor_session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Build relevant test directory set when --run-relevant-tests is active
        commit_ids = get_feature_flag(config, "run_relevant_tests")
        if commit_ids:
            from .utils.relevant_test_finder import RelevantTestFinder

            finder = RelevantTestFinder(repo_root=Path(__file__).parent.parent)
            setattr(config, RELEVANT_TEST_DIRS_KEY, finder.get_relevant_test_dirs(commit_ids))
            relevant_dirs = getattr(config, RELEVANT_TEST_DIRS_KEY)
            if relevant_dirs is not None:
                logging.info("Relevant test directories: %s", relevant_dirs)
            else:
                logging.info("Running all tests (infrastructure change or fallback)")

        artifact_bucket = get_feature_flag(config, "artifact_upload_s3_bucket")
        test_output_bucket = get_feature_flag(config, "test_output_s3_bucket")
        profile = get_feature_flag(config, "aws_profile")
        # Prefetch credentials for S3 or SQS operations
        if artifact_bucket or test_output_bucket or sqs_queue_url:
            prefetch_and_cache_credentials(profile)

    # Create session context (single source of truth for session-level fields).
    # Must be outside the master-only block so xdist workers also build it.
    config._session_context = SessionContext(
        target=",".join(p.value for p in get_platform_targets(config)),
        trace_mode=resolve_session_trace_mode(config).value,
        nki_compilation_mode=get_feature_flag(config, "nki_compilation_mode"),
        kernel_name=os.environ.get("KERNEL_NAME"),
        run_type=os.environ.get("RUN_TYPE"),
        is_release=os.environ.get("IS_RELEASE", "").lower() == "true",
        sqs_queue_url=sqs_queue_url,
        username=resolve_current_user(),
        version_set_eid=os.environ.get("VERSION_SET_EID"),
    )

    # Validate S3 credentials for test output upload
    # This runs after pre-fetch so it uses the cached credentials
    # Validate S3 credentials before tests run
    upload_outcomes = get_feature_flag(config, "upload_test_outcomes")
    if upload_outcomes:
        s3_config = S3ArtifactUploadConfig(
            bucket=get_feature_flag(config, "test_output_s3_bucket"),
            prefix=get_feature_flag(config, "test_output_s3_prefix"),
            profile=get_feature_flag(config, "aws_profile"),
        )
        validate_s3_credentials(s3_config)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: Config, items: list[pytest.Item]):
    # Filter to relevant tests when --run-relevant-tests is active
    relevant_dirs = getattr(config, RELEVANT_TEST_DIRS_KEY, "NOT_SET")
    if relevant_dirs != "NOT_SET" and relevant_dirs is not None:
        original_count = len(items)
        kept = []
        deselected = []
        for item in items:
            if any(str(item.fspath).startswith(d) for d in relevant_dirs):
                kept.append(item)
            else:
                deselected.append(item)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
        items[:] = kept
        logging.info(
            "Relevant test filter: %d -> %d tests (%d deselected)", original_count, len(items), len(deselected)
        )


def pytest_configure_node(node):
    """Pass session state from master to workers (xdist hook)."""
    if hasattr(node.config, "_qor_session_id"):
        node.workerinput["qor_session_id"] = node.config._qor_session_id
    if hasattr(node.config, RELEVANT_TEST_DIRS_KEY):
        node.workerinput["relevant_test_dirs"] = (
            list(getattr(node.config, RELEVANT_TEST_DIRS_KEY))
            if getattr(node.config, RELEVANT_TEST_DIRS_KEY) is not None
            else None
        )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """End of test session - emit run_complete and log summaries (master only)."""
    # Handle --disable-no-tests-failure: convert exit code 5 (no tests collected) to 0 with warning
    if exitstatus == 5 and get_feature_flag(session.config, "disable_no_tests_failure", False):
        logging.warning("No tests were collected, but --disable-no-tests-failure is set - exiting with code 0")
        session.exitstatus = 0

    # Only run on master (not xdist workers)
    if hasattr(session.config, "workerinput"):
        return

    # Emit run_complete via SQS
    session_ctx = getattr(session.config, "_session_context", None)
    if session_ctx and session_ctx.sqs_queue_url:
        terminalreporter = session.config.pluginmanager.get_plugin("terminalreporter")
        # A skipped test has a "passed" setup phase, so exclude skipped nodeids from passed.
        skipped_ids = {r.nodeid for r in terminalreporter.stats.get("skipped", [])}
        failed_ids = {r.nodeid for r in terminalreporter.stats.get("failed", [])}
        xfailed_ids = {r.nodeid for r in terminalreporter.stats.get("xfailed", [])}
        passed_ids = {r.nodeid for r in terminalreporter.stats.get("passed", [])} - skipped_ids

        SQSEmitter(session=session_ctx).emit_run_complete(
            tests_passed=len(passed_ids),
            tests_total=len(passed_ids) + len(failed_ids) + len(skipped_ids),
            tests_skipped=len(skipped_ids),
            tests_xfailed=len(xfailed_ids),
            coverage_data=get_coverage_data(session.config),
        )

    # Log QoR CSV file path
    if hasattr(session.config, "_qor_session_id"):
        output_dir = resolve_base_output_directory(session.config)
        filepath = os.path.join(output_dir, f"qor_data_{session.config._qor_session_id}.csv")
        if os.path.exists(filepath):
            print(f"\nQoR data collected to {filepath}")

    # Print memory monitoring summary
    memory_top_n = get_feature_flag(session.config, "monitor_memory")
    if memory_top_n is not None:
        output_dir = Path(resolve_base_output_directory(session.config))
        MEMORY_MONITOR.merge_and_report(output_dir, memory_top_n)

    # Print duration monitoring summary
    top_n = get_feature_flag(session.config, "monitor_durations")
    if top_n is not None:
        output_dir = Path(resolve_base_output_directory(session.config))
        DURATION_MONITOR.merge_and_report(output_dir, top_n)


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Inject UTC timestamp into the final summary separator line."""
    if hasattr(config, "workerinput"):
        return

    original_summary_stats = terminalreporter.summary_stats

    def patched_summary_stats():
        original_write_sep = terminalreporter.write_sep

        def write_sep_with_timestamp(sep, title="", **kw):
            if title:
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                title += f" by {ts}"
            original_write_sep(sep, title, **kw)

        terminalreporter.write_sep = write_sep_with_timestamp
        try:
            original_summary_stats()
        finally:
            terminalreporter.write_sep = original_write_sep

    terminalreporter.summary_stats = patched_summary_stats


def pytest_sessionstart(session: pytest.Session) -> None:
    """nkilib-internal: clean stale monitor CSVs."""
    if not hasattr(session.config, "workerinput"):
        output_dir = Path(resolve_base_output_directory(session.config))
        if get_feature_flag(session.config, "monitor_memory") is not None:
            MEMORY_MONITOR.cleanup_stale(output_dir)
        if get_feature_flag(session.config, "monitor_durations") is not None:
            DURATION_MONITOR.cleanup_stale(output_dir)
