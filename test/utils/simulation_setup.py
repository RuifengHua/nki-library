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
"""
nki.simulate integration - all simulator-dependent code in one place.

This module is only imported when simulation mode is active.
"""

import logging
import os

import numpy as np

from .common_dataclasses import GoldenTensorDict, normalize_golden_output
from .simulation_constants import SIMULATION_RUN_ALL_ENV_VAR

# Patterns to identify tests with large shapes that are slow on CPU simulation
_LARGE_SHAPE_PATTERNS = ("4096", "5120", "7168", "8192", "10240", "16384", "32768", "36864")


def setup_simulation_mode():
    """Setup simulation mode"""

    # Limit BLAS threading in xdist workers to avoid contention
    if "PYTEST_XDIST_WORKER" in os.environ:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def skip_slow_simulation_tests(items, skip_marker):
    """Mark tests with large tensor shapes for skipping. Set NKILIB_SIMULATION_RUN_ALL=1 to run all."""
    if os.environ.get(SIMULATION_RUN_ALL_ENV_VAR) == "1":
        return
    for item in items:
        if item.get_closest_marker("slow_simulation") or any(pattern in item.name for pattern in _LARGE_SHAPE_PATTERNS):
            item.add_marker(skip_marker)


def simulate_kernel(kernel_func, kernel_input: dict, lnc_count: int) -> list:
    """Execute kernel using nki.simulator.simulate_kernel."""
    from nki.simulator import simulate_kernel as _simulate_kernel

    # Strip ".must_alias_input" suffix from parameter names - this suffix is added
    # for the graph compiler but simulate_kernel expects original param names
    cleaned_input = {k.removesuffix(".must_alias_input"): v for k, v in kernel_input.items()}
    result = _simulate_kernel(kernel_func, args=[], kwargs=cleaned_input, _lnc=lnc_count)

    if result is None:
        return []
    elif isinstance(result, (list, tuple)):
        return list(result)
    else:
        return [result]


def run_simulator_inference(kernel_under_test) -> dict[str, np.ndarray]:
    """Run kernel using nki.simulate and return outputs.

    Returns dict mapping output names to numpy arrays, ready for dumping/validation.
    """
    logging.info("Running kernel via nki.simulate")

    os.environ["NKI_NC_VERSION"] = kernel_under_test.compiler_input.platform_target.get_nc_gen()
    kernel_outputs = simulate_kernel(
        kernel_under_test.kernel_func,
        kernel_under_test.kernel_input or {},
        kernel_under_test.compiler_input.logical_nc_config,
    )

    if kernel_under_test.validation_args is None:
        raise ValueError("Simulation mode requires validation_args to verify outputs")

    golden_output = kernel_under_test.validation_args.golden_output
    golden_tensors: GoldenTensorDict = normalize_golden_output(golden_output)
    output_names: list[str] = list(golden_tensors.keys())

    if len(kernel_outputs) != len(output_names):
        raise RuntimeError(
            f"Kernel returned {len(kernel_outputs)} outputs but expected {len(output_names)}. "
            f"Got: {[type(o).__name__ for o in kernel_outputs]}, Expected names: {output_names}"
        )

    # Convert kernel outputs to numpy arrays with dtypes matching golden tensors
    return {
        name: np.asarray(output, dtype=golden.dtype)
        for name, output, golden in zip(output_names, kernel_outputs, golden_tensors.values())
    }
