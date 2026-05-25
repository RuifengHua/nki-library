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
"""NKI Library Testing — pytest plugin.

Auto-registered via pytest11 entry point. Provides shared CLI options, fixtures,
and hooks for NKI kernel testing.

Fixtures:
    platform_target, trace_mode, output_directory, metric_output_mode,
    collector, emitter, host_manager, perf_analysis_enabled, test_manager

Hooks:
    pytest_addoption, pytest_configure, pytest_generate_tests,
    pytest_collection_modifyitems, pytest_sessionstart, pytest_collection_finish
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import pytest
from _pytest.config import Config
from _pytest.python import Metafunc

from .common_dataclasses import (
    NKICompilationMode,
    PlatformAware,
    Platforms,
    TargetHost,
    TraceMode,
    UploadProfileMode,
    get_test_tier,
)
from .coverage_parametrized_tests import extract_parametrize_args, generate_parametrized_test_case
from .feature_flag_helper import derive_pytest_test_id, get_feature_flag, resolve_base_output_directory
from .host_management import HostManager, detect_local_neuron_devices
from .metrics_collector import IMetricsCollector, MetricsCollector, NoopMetricsCollector
from .metrics_emitter import IMetricsEmitter, MetricsEmitter, NoopMetricsEmitter, OutputMode
from .param_extractor import extract_pytest_params, normalize_param_names
from .pytest_test_metadata import derive_labeled_kernel_name, discover_pytest_test_metadata_marks
from .s3_utils import S3ArtifactUploadConfig
from .simulation_setup import setup_simulation_mode
from .test_orchestrator import Orchestrator

_RNG_SEED_ENV_KEY = "NEURON_PYTHONHASHSEED"

# ─── Helper functions ───


def get_platform_targets(config: Config) -> list[Platforms]:
    """Return the list of target platforms from CLI (or default to [TRN2])."""
    valid_names = [p.value for p in Platforms]
    raw = get_feature_flag(config, "platform_target", None)
    if raw is None:
        return [Platforms.TRN2]
    platforms: list[Platforms] = []
    for token in raw.split(","):
        name = token.strip()
        if not name:
            continue
        try:
            platforms.append(Platforms(name))
        except ValueError:
            raise pytest.UsageError(f"Unknown platform target '{name}'. Valid options: {', '.join(valid_names)}")
    if not platforms:
        raise pytest.UsageError(f"--platform-target resolved to an empty list. Valid options: {', '.join(valid_names)}")
    return platforms


def resolve_session_trace_mode(config: Config) -> TraceMode:
    """Resolve session trace mode from CLI flags. Result is cached on config."""
    if hasattr(config, "_session_trace_mode"):
        return config._session_trace_mode

    test_mode = get_feature_flag(config, "test_mode")
    if test_mode:
        config._session_trace_mode = TraceMode.create(test_mode)
    elif get_feature_flag(config, "target_host"):
        config._session_trace_mode = TraceMode.CompileAndInfer
    else:
        neuron_installation_path = get_feature_flag(config, "neuron_tools_bin_path")
        if detect_local_neuron_devices(neuron_installation_path):
            config._session_trace_mode = TraceMode.CompileAndInfer
            logging.info("Local Neuron devices detected, using CompileAndInfer mode")
        else:
            config._session_trace_mode = TraceMode.CompileOnly

    return config._session_trace_mode


def is_simulation_mode(config: Config) -> bool:
    """Check if simulation mode is active."""
    return resolve_session_trace_mode(config) == TraceMode.Simulator


def is_debugger_mode(config: Config) -> bool:
    """Check if debugger mode is active."""
    return resolve_session_trace_mode(config) == TraceMode.Debugger


def make_collector(
    request: pytest.FixtureRequest,
    metric_output_mode: OutputMode | None,
    namespace: str = "NeuronCompiler",
) -> IMetricsCollector:
    """Create a metrics collector. Returns NoopMetricsCollector when metrics are disabled."""
    if metric_output_mode is None:
        return NoopMetricsCollector()

    collector = MetricsCollector()
    collector.set_namespace(namespace)

    if hasattr(request.node, "callspec"):
        params = extract_pytest_params(request.node.callspec.params)
        params = normalize_param_names(params)
        collector.set_kernel_params(params)

    # Set TestName from nodeid (same derivation as orchestrator)
    collector.set_test_name(derive_pytest_test_id())

    # Set TestTier from pytest tier marks (tier0, optimal, generality, broad)
    tier = get_test_tier(request.node)
    if tier is not None:
        collector.add_dimension({"TestTier": tier.value})

    metadata_name = None
    if request.cls and hasattr(request.cls, "__pytest_test_metadata__"):
        metadata_name = request.cls.__pytest_test_metadata__.get("name")
    labeled_kernel = derive_labeled_kernel_name(request.fspath, metadata_name)
    if labeled_kernel:
        collector.add_dimension({"KernelName": labeled_kernel})

    return collector


def make_emitter(
    metric_output_mode: OutputMode | None,
) -> IMetricsEmitter:
    """Create a basic metrics emitter. Returns NoopMetricsEmitter when metrics are disabled."""
    if metric_output_mode is None:
        return NoopMetricsEmitter()
    return MetricsEmitter(output_mode=metric_output_mode)


def make_host_manager(
    config: Config,
    *,
    target_hosts: list[TargetHost] | None = None,
    s3_config: S3ArtifactUploadConfig | None = None,
) -> HostManager:
    """Create a HostManager from shared CLI options.

    Args:
        target_hosts: Override host list (default: built from --target-host CLI).
        s3_config: Override S3 config (default: built from --artifact-upload-s3-* CLI).
    """
    neuron_installation_path: str = get_feature_flag(config, "neuron_tools_bin_path")
    ssh_config_path: str = os.path.expanduser(get_feature_flag(config, "ssh_config_path", "~/.ssh/config"))

    if s3_config is None:
        s3_config = S3ArtifactUploadConfig(
            bucket=get_feature_flag(config, "artifact_upload_s3_bucket"),
            prefix=get_feature_flag(config, "artifact_upload_s3_prefix"),
            profile=get_feature_flag(config, "aws_profile"),
        )

    if target_hosts is None:
        target_hosts_cli: list[str] = get_feature_flag(config, "target_host", list())
        if target_hosts_cli:
            platforms = get_platform_targets(config)
            assert len(platforms) == 1, f"--target-host requires a single --platform-target, got {platforms}"
            target_hosts = [TargetHost(ssh_host=ip, host_type=platforms[0]) for ip in target_hosts_cli]
        else:
            target_hosts = []

    hm = HostManager(
        base_host_info_path=resolve_base_output_directory(config),
        neuron_installation_path=neuron_installation_path,
        target_hosts=target_hosts,
        ssh_config_path=ssh_config_path,
        s3_config=s3_config,
    )
    hm.initialize_host_stats()
    return hm


def make_test_manager(
    config: Config,
    trace_mode: TraceMode,
    host_manager: HostManager,
    collector: IMetricsCollector,
    perf_analysis_enabled: bool = False,
    kernel_name: str | None = None,
) -> Orchestrator:
    """Create a standard Orchestrator from shared config."""
    return Orchestrator(
        config,
        trace_mode,
        host_manager,
        collector,
        perf_analysis_enabled=perf_analysis_enabled,
        kernel_name=kernel_name,
        nki_compilation_mode=NKICompilationMode[get_feature_flag(config, "nki_compilation_mode")],
    )


# ─── Shared CLI options ───


def pytest_addoption(parser):
    group = parser.getgroup("nkilib", "NKI Library Testing")

    group.addoption(
        "--target-host",
        default=[],
        nargs="+",
        help="Hostname(s) of MLA accelerator hosts to execute tests on remotely",
    )
    group.addoption(
        "--output-directory",
        default="neuron_test_output",
        help="Base directory for artifacts produced by test cases",
    )
    group.addoption(
        "--neuron-tools-bin-path",
        default="/opt/aws/neuron/bin",
        help="Path to directory containing neuron tools (neuron-profile, neuron-ls, etc.) on remote hosts",
    )
    group.addoption(
        "--ssh-config-path",
        help="Path to SSH config file for remote connections (default: ~/.ssh/config)",
    )
    group.addoption(
        "--skip-remote-cleanup",
        action="store_true",
        default=False,
        help="Skip cleanup of remote directories after test execution",
    )
    group.addoption(
        "--debug-kernels",
        action="store_true",
        default=False,
        help="Dump additional debug output inside test directory",
    )
    group.addoption(
        "--metric-output",
        nargs="?",
        const="file",
        default="file",
        choices=["file", "stdout", "stderr"],
        help="Enable metrics collection: 'file' (default), 'stdout', or 'stderr'",
    )
    group.addoption(
        "--test-mode",
        action="store",
        choices=["trace-only", "compile-only", "compile-and-infer", "simulation", "debugger"],
        help="Override default trace mode (markers take precedence)",
    )
    group.addoption(
        "--nki-compilation-mode",
        action="store",
        default=NKICompilationMode.parser.value,
        choices=[m.value for m in NKICompilationMode],
        help="NKI compiler frontend to use",
    )
    group.addoption(
        "--debugger-interactive",
        action="store_true",
        default=False,
        help="Enable interactive mode for nki.debug",
    )
    group.addoption(
        "--debugger-core-id",
        action="store",
        type=int,
        default=0,
        help="NeuronCore ID to debug (default: 0)",
    )
    group.addoption(
        "--debugger-replay",
        action="store_true",
        default=False,
        help="Replay nki.debug() against device dumps from a previous run",
    )
    group.addoption(
        "--platform-target",
        action="store",
        default=None,
        help="Target instance family for test execution. "
        "Single value or comma-separated list (e.g., trn2,trn3_a0). Auto-detected if omitted.",
    )
    group.addoption(
        "--validation-histograms",
        action="store_true",
        default=False,
        help="Dump full report with histograms during validation",
    )
    group.addoption(
        "--enable-perf-analysis",
        action="store_true",
        default=False,
        help="Enable performance analysis with perf sim and detailed profiled JSON",
    )
    group.addoption(
        "--force-local-cleanup",
        action="store_true",
        default=False,
        help="Automatically cleanup test output directory regardless of test outcome",
    )
    group.addoption(
        "--force-local-cleanup-keep",
        nargs="+",
        choices=["metrics"],
        default=[],
        help="Artifact types to preserve when using --force-local-cleanup",
    )
    group.addoption(
        "--enable-dge-notifs",
        action="store_true",
        default=False,
        help="Enable DGE Notifications during profiling",
    )
    group.addoption(
        "--artifact-upload-s3-bucket",
        action="store",
        default=None,
        help="S3 bucket for artifact file transfer to remote hosts",
    )
    group.addoption(
        "--artifact-upload-s3-prefix",
        action="store",
        default="artifacts_tmp",
        help="S3 prefix for artifact file transfer (default: artifacts_tmp)",
    )
    group.addoption(
        "--aws-profile",
        action="store",
        default=None,
        help="AWS profile name for S3 authentication",
    )
    group.addoption(
        "--upload-profile-to-explorer",
        nargs="?",
        const=UploadProfileMode.ALWAYS.value,
        default=None,
        choices=[m.value for m in UploadProfileMode],
        help="Upload profiling artifacts to Neuron Explorer. 'always' (default when flag provided), 'on-fail-only' (upload only on test failure)",
    )

    group.addoption(
        "--skip-model-tests",
        action="store_true",
        default=False,
        help="Exclude model tests from collection",
    )

    coverage_group = parser.getgroup("coverage-parametrize")
    coverage_group.addoption(
        "--coverage",
        action="store",
        default="singles",
        choices=["singles", "pairs", "full"],
        help="Default parameter coverage regime for coverage_parametrize tests",
    )
    coverage_group.addoption(
        "--skip-coverage-parametrize",
        action="store_true",
        help="Exclude coverage_parametrize tests from collection",
    )


# ─── Shared fixtures ───


@pytest.fixture(scope="session")
def output_directory(request: pytest.FixtureRequest) -> str:
    output_dir_path = resolve_base_output_directory(request.config)
    Path(output_dir_path).mkdir(exist_ok=True)
    return output_dir_path


@pytest.fixture(scope="session")
def session_trace_mode(request: pytest.FixtureRequest) -> TraceMode:
    """Session-wide trace mode from CLI flags. Individual tests may override via markers."""
    return resolve_session_trace_mode(request.config)


@pytest.fixture
def trace_mode(request: pytest.FixtureRequest, session_trace_mode: TraceMode) -> TraceMode:
    """Per-test trace mode. Markers override the session default."""
    for mode in TraceMode:
        if request.node.get_closest_marker(mode.value) is not None:
            return mode
    return session_trace_mode


@pytest.fixture
def metric_output_mode(request: pytest.FixtureRequest) -> OutputMode | None:
    metric_output: str | None = get_feature_flag(request.config, "metric_output")
    valid_values = {None, OutputMode.FILE.value, OutputMode.STDOUT.value, OutputMode.STDERR.value}
    assert metric_output in valid_values, (
        f"Invalid --metric-output value: '{metric_output}'. "
        f"Valid options: '{OutputMode.FILE.value}', '{OutputMode.STDOUT.value}', '{OutputMode.STDERR.value}'"
    )
    return OutputMode(metric_output) if metric_output else None


@pytest.fixture
def collector(request: pytest.FixtureRequest, metric_output_mode: OutputMode | None) -> IMetricsCollector:
    """Create metrics collector (Noop when metrics are disabled)."""
    return make_collector(request, metric_output_mode)


@pytest.fixture
def emitter(metric_output_mode: OutputMode | None) -> IMetricsEmitter:
    """Create metrics emitter (Noop when metrics are disabled)."""
    return make_emitter(metric_output_mode)


@pytest.fixture(scope="session")
def host_manager(request: pytest.FixtureRequest) -> HostManager:
    """Host manager using --target-host and --artifact-upload-s3-* CLI options. No host-file support."""
    return make_host_manager(request.config)


@pytest.fixture
def perf_analysis_enabled(request: pytest.FixtureRequest) -> bool:
    return get_feature_flag(request.config, "enable_perf_analysis", False)


@pytest.fixture
def test_manager(
    request: pytest.FixtureRequest,
    trace_mode: TraceMode,
    host_manager: HostManager,
    collector: IMetricsCollector,
    perf_analysis_enabled: bool,
) -> Orchestrator:
    """Standard kernel test orchestrator."""
    metadata_name = None
    if request.cls and hasattr(request.cls, "__pytest_test_metadata__"):
        metadata_name = request.cls.__pytest_test_metadata__.get("name")
    kernel_name = derive_labeled_kernel_name(request.fspath, metadata_name)
    return make_test_manager(
        request.config,
        trace_mode,
        host_manager,
        collector,
        perf_analysis_enabled=perf_analysis_enabled,
        kernel_name=kernel_name,
    )


@pytest.fixture
def platform_target(request: pytest.FixtureRequest) -> Platforms:
    """Injected via indirect parametrization from pytest_generate_tests."""
    return request.param


# ─── Shared hooks ───


def pytest_configure(config: Config):
    """Auto-discover marks, set up simulation mode, register platform markers."""
    # Discover marks from @pytest_test_metadata decorators
    # Use config.rootdir so discovery works whether the plugin is loaded from
    # the source tree or from the installed nkilib_testing wheel.
    test_root = Path(config.rootdir) / "test"
    if test_root.is_dir():
        discovered_marks = discover_pytest_test_metadata_marks(test_root)
        for mark_name, description in discovered_marks.items():
            config.addinivalue_line("markers", f"{mark_name}: {description}")

    if is_simulation_mode(config):
        setup_simulation_mode()

    for p in Platforms:
        config.addinivalue_line(
            "markers",
            f"{p.value}: Dynamically applied to tests targeting the {p.value} platform",
        )

    # Apply platform-target as pytest markexpr for correct test collection
    platforms = get_platform_targets(config)
    platform_expr = "(" + " or ".join(p.value for p in platforms) + ")"
    marker_expr: str | None = config.option.markexpr
    if marker_expr:
        config.option.markexpr = f"{marker_expr} and {platform_expr}"
    else:
        config.option.markexpr = platform_expr


def pytest_sessionstart(session):
    """Seed random generators for deterministic test collection across xdist workers."""
    if _RNG_SEED_ENV_KEY in os.environ:
        seed = int(os.environ[_RNG_SEED_ENV_KEY])
        session._original_random_state = random.getstate()
        session._original_numpy_state = np.random.get_state()
        random.seed(seed)
        np.random.seed(seed)


def pytest_collection_finish(session):
    """Restore random generators after collection and validate debugger mode."""
    if is_debugger_mode(session.config) and len(session.items) != 1:
        raise pytest.UsageError(
            f"Debugger mode requires exactly one test. Got {len(session.items)}. Use -k to select a single test."
        )
    if hasattr(session, "_original_random_state"):
        random.setstate(session._original_random_state)
    if hasattr(session, "_original_numpy_state"):
        np.random.set_state(session._original_numpy_state)


def pytest_collection_modifyitems(config: Config, items: list[pytest.Item]):
    """Apply platform marks to tests and skip slow tests in simulation mode."""
    # Deselect coverage_parametrize tests when --skip-coverage-parametrize is set
    if get_feature_flag(config, "skip_coverage_parametrize", default_value=False):
        items[:] = [item for item in items if not item.get_closest_marker("coverage_parametrize")]

    for item in items:
        platforms_marker = item.get_closest_marker("platforms")
        excluded = set(platforms_marker.kwargs.get("exclude") or []) if platforms_marker else set()

        if hasattr(item, "callspec"):
            for param_val in item.callspec.params.values():
                if isinstance(param_val, PlatformAware) and param_val.supported_platforms is not None:
                    excluded |= set(Platforms) - param_val.supported_platforms

        supported = set(Platforms) - excluded

        if hasattr(item, "callspec") and "platform_target" in item.callspec.params:
            supported &= {item.callspec.params["platform_target"]}

        for p in supported:
            item.add_marker(pytest.mark.__getattr__(p.value))

    if is_simulation_mode(config):
        from .simulation_setup import skip_slow_simulation_tests

        skip_marker = pytest.mark.skip(reason="Skipping slow simulation test (see test/simulation.md)")
        skip_slow_simulation_tests(items, skip_marker)


# =========================
# COVERAGE GENERATORS - @pytest.mark.coverage_parametrize
# =========================
"""
Coverage Parametrize Feature
============================

The coverage_parametrize marker provides intelligent test case generation with configurable
coverage strategies. It generates parameter combinations based on coverage requirements
while supporting filtering and validation.

Usage:
    @pytest.mark.coverage_parametrize(
        param1=[value1, value2, ...],
        param2=[value1, value2, ...],
        coverage="singles|pairs|full",  # Optional: overrides CLI default
        filter=filter_function          # Optional: constraint function
    )

Coverage Strategies:
    - "singles": Each parameter value appears at least once (1-way coverage)
    - "pairs": All parameter pairs are covered (2-way coverage using AllPairs)
    - "full": Complete cartesian product of all parameters

Filter Functions:
    - Must accept parameter names as keyword arguments
    - Return True to include the combination, False to exclude
    - Specifying default values for filter arguments helps create smaller covering sets
    - Example: def filter_func(param1, param2=None): return param1 < param2

Limitations:
    - All parameter values must be hashable (strings, numbers, tuples, etc.)
    - Filter functions with default values work better with AllPairs algorithm
    - Large parameter spaces with "full" coverage can generate many test cases

CLI Options:
    --coverage {singles,pairs,full}  Set default coverage strategy
"""


def pytest_generate_tests(metafunc: Metafunc):
    """Parametrize platform_target and coverage_parametrize tests."""
    # Always parametrize platform_target so test IDs are stable regardless of how many platforms are requested
    platforms = get_platform_targets(metafunc.config)
    if "platform_target" in metafunc.fixturenames:
        metafunc.parametrize("platform_target", platforms, indirect=True, ids=[str(p) for p in platforms])

    # Handle coverage_parametrize (independent of platform parametrization)
    coverage_marker = metafunc.definition.get_closest_marker("coverage_parametrize")
    skip_coverage = get_feature_flag(metafunc.config, "skip_coverage_parametrize", default_value=False)
    if not coverage_marker or skip_coverage:
        return

    # When -m selects fast tests, skip coverage generation for non-fast tests.
    # Coverage-parametrized tests never carry the fast mark, so they'll always be
    # deselected by -m fast. Skipping here avoids expensive AllPairs / combinatorial
    # expansion for ~100+ test functions that would be thrown away during deselection.
    mark_expr = metafunc.config.option.markexpr or ""
    if "fast" in mark_expr and "not fast" not in mark_expr:
        if not metafunc.definition.get_closest_marker("fast"):
            return

    if _RNG_SEED_ENV_KEY in os.environ:
        random.seed(int(os.environ[_RNG_SEED_ENV_KEY]))

    params = coverage_marker.kwargs.copy()
    assert params, "No parameters defined for coverage_parametrize"
    coverage_override = params.pop("coverage", None)
    filter_func = params.pop("filter", None)
    enable_automatic_boundary_tests = params.pop("enable_automatic_boundary_tests", True)
    enable_invalid_combination_tests = params.pop("enable_invalid_combination_tests", True)
    n_tests_per_boundary_value = params.pop("n_tests_per_boundary_value", 3)
    max_invalid_tests = params.pop("max_invalid_tests", 30)
    abbrev = params.pop("abbrev", None)

    coverage = coverage_override if coverage_override is not None else get_feature_flag(metafunc.config, "coverage")

    test_cases = generate_parametrized_test_case(
        params=params,
        coverage=coverage,
        filter_func=filter_func,
        enable_automatic_boundary_tests=enable_automatic_boundary_tests,
        enable_invalid_combination_tests=enable_invalid_combination_tests,
        n_tests_per_boundary_value=n_tests_per_boundary_value,
        max_invalid_tests=max_invalid_tests,
    )

    param_names, values_list, ids_list = extract_parametrize_args(
        params, test_cases, abbrev=abbrev, test_func_name=metafunc.function.__name__
    )
    metafunc.parametrize(param_names, values_list, ids=ids_list)
