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

"""Unit Test Framework for NKI Kernels.

This framework standardizes test interfaces, enforces consistency, and simplifies unit test implementations.

Key Features:
    - Signature validation: kernel_entry ↔ torch_ref parameter consistency
    - Input validation: kernel_input keys match kernel_entry signature
    - Output validation: output_tensor_descriptor keys match torch_ref returns
    - Unused parameter detection (opt-in): catches forgotten pass-through in wrappers

Usage:
    Developers need to provide:
    1. Kernel input generator function
    2. Torch reference generator function
    3. Test configuration (parameters and dimensions)
    4. For SBUF I/O: a thin wrapper that handles HBM<->SBUF conversions

    The framework handles:
    - Test orchestration and execution
    - Input/output validation
    - Automatic parameter filtering
"""

import functools
import inspect
from inspect import signature
from typing import Callable, Optional

import neuron_dtypes as ndt
import numpy as np
import torch

from .common_dataclasses import (
    CompilerArgs,
    InferenceArgs,
    KernelArgs,
    LazyGoldenGenerator,
    PerRankLazyGoldenGenerator,
    PerRankLazyInputGenerator,
    ValidationArgs,
)
from .coverage_parametrized_tests import assert_negative_test_case
from .metadata_loader import load_model_configs
from .metrics_collector import IMetricsCollector
from .test_orchestrator import Orchestrator


class UnitTestFramework:
    """Framework for executing NKI kernel unit tests.

    Orchestrates the complete test flow:
    1. Generate kernel inputs from test configuration
    2. Execute kernel with generated inputs
    3. Validate outputs against golden reference
    """

    def __init__(
        self,
        test_manager: Orchestrator,
        kernel_entry: Callable,
        kernel_input_generator: Callable,
        torch_ref: Optional[Callable] = None,
        output_tensor_descriptor: Optional[Callable] = None,
        check_unused_params: bool = False,
        collector: Optional[IMetricsCollector] = None,
        trace_only: bool = False,
    ):
        """Initialize the test framework.

        Args:
            test_manager: Test orchestrator for execution
            kernel_entry: Kernel function under test
            kernel_input_generator: Function(test_config) -> dict of inputs
            torch_ref: Torch reference function (signature must match kernel_entry).
                Required unless trace_only=True.
            output_tensor_descriptor: Function(kernel_input) -> dict of output tensors.
                Required unless trace_only=True.
            check_unused_params: Check for unused parameters in kernel_entry
            collector: Optional metrics collector for model coverage tracking
            trace_only: If True, skip torch_ref validation and output comparison.
                Use for tests that only trace/compile the kernel and validate via
                kernel_assert inside the kernel itself.
        """
        self.trace_only = trace_only

        if not trace_only:
            if torch_ref is None:
                raise ValueError("torch_ref is required when trace_only=False")
            if output_tensor_descriptor is None:
                raise ValueError("output_tensor_descriptor is required when trace_only=False")
            validate_torch_ref_signature(kernel_entry, torch_ref)

        if check_unused_params:
            check_unused_parameters(kernel_entry)

        self.test_manager = test_manager
        self.kernel_entry = kernel_entry
        self.torch_ref = torch_ref
        self.kernel_input_generator = kernel_input_generator
        self.output_tensor_descriptor = output_tensor_descriptor
        self.collector = collector

    def run_test(
        self,
        test_config,
        compiler_args: CompilerArgs,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan_inf: bool = False,
        is_negative_test: bool = False,
        inference_args: Optional[InferenceArgs] = None,
        custom_validation_args: Optional[ValidationArgs] = None,
        custom_comparator: Optional[Callable] = None,
        metadata: Optional[dict] = None,
    ):
        """Execute a single test case.

        Args:
            test_config: Test configuration (can be None when using pytest.mark.parametrize)
            compiler_args: Compiler arguments
            rtol: Relative tolerance for validation
            atol: Absolute tolerance for validation
            equal_nan_inf: If True, matching NaN and matching infinity values are treated as equal
            is_negative_test: Whether this is a negative test case
            inference_args: Optional inference arguments (e.g., for determinism checking)
            custom_validation_args: Optional pre-built ValidationArgs that bypasses both
                torch_ref golden generation and the default comparison. Use this only when
                the golden cannot come from torch_ref at all.
            custom_comparator: Optional callable(golden_dict, output_tensors) -> dict mapping
                output names to CustomValidatorWithOutputTensorData. The framework runs
                torch_ref to produce golden_dict, then passes it to this function to build
                custom validators. This keeps golden generation in the framework while
                allowing custom comparison logic (e.g., cosine similarity, scaled tolerances).
                Mutually exclusive with custom_validation_args.
            metadata: Optional dict with 'config_name' (str for load_model_configs) and 'key' (test dimensions dict)
        """
        if custom_validation_args is not None and custom_comparator is not None:
            raise ValueError("custom_validation_args and custom_comparator are mutually exclusive")

        if self.collector is not None and metadata is not None:
            metadata_list = load_model_configs(metadata["config_name"])
            self.collector.match_and_add_metadata_dimensions(metadata["key"], metadata_list)

        with assert_negative_test_case(is_negative_test):
            # Generate kernel inputs
            kernel_input = self.kernel_input_generator(test_config)

            # Validate and filter inputs using shared helpers
            validate_input_keys(kernel_input, self.kernel_entry)
            filtered_kernel_input = filter_kernel_input(kernel_input, self.kernel_entry)

            if self.trace_only:
                # Trace-only mode: no golden comparison, just trace/compile the kernel.
                # Build output_ndarray from .must_alias_input keys so the compiler knows
                # the kernel's output names (required for compilation).
                output_ndarray = {}
                for k, v in kernel_input.items():
                    if k.endswith(".must_alias_input"):
                        base_key = k.rsplit(".must_alias_input", 1)[0]
                        output_ndarray[base_key] = v
                validation_args = ValidationArgs(
                    golden_output=LazyGoldenGenerator(output_ndarray=output_ndarray, lazy_golden_generator=None),
                )
                kernel_args = KernelArgs(
                    kernel_func=self.kernel_entry,
                    compiler_input=compiler_args,
                    kernel_input=filtered_kernel_input,
                    validation_args=validation_args,
                )
                if inference_args is not None:
                    kernel_args.inference_args = inference_args
                self.test_manager.execute(kernel_args)
                return

            ref_input = filter_ref_input(kernel_input, self.torch_ref)

            # Generate output tensors
            output_tensors = self.output_tensor_descriptor(kernel_input)

            # Defer torch_ref computation to validation time via lazy golden generator.
            # This avoids running torch_ref in compile-only/trace-only modes (orchestrator
            # returns early and .golden is never accessed), and in normal mode it runs
            # after the kernel compile+infer, right before output data comparison.
            def compute_ref():
                ref_result = self.torch_ref(**ref_input)
                _validate_key_sets(
                    expected=set(ref_result.keys()),
                    actual=set(output_tensors.keys()),
                    msg_header="Output tensor mismatch:",
                    expected_label="torch_ref returns but output_tensor_descriptor doesn't provide",
                    actual_label="output_tensor_descriptor provides but torch_ref doesn't return",
                )
                if custom_comparator is not None:
                    return custom_comparator(ref_result, output_tensors)
                return ref_result

            lazy_golden = LazyGoldenGenerator(
                lazy_golden_generator=compute_ref,
                output_ndarray=output_tensors,
            )

            # Execute test
            validation_args = custom_validation_args or ValidationArgs(
                golden_output=lazy_golden,
                relative_accuracy=rtol,
                absolute_accuracy=atol,
                equal_nan_inf=equal_nan_inf,
            )
            kernel_args = KernelArgs(
                kernel_func=self.kernel_entry,
                compiler_input=compiler_args,
                kernel_input=filtered_kernel_input,
                validation_args=validation_args,
            )
            if inference_args is not None:
                kernel_args.inference_args = inference_args

            self.test_manager.execute(kernel_args)


class CollectiveUnitTestFramework:
    """Framework for executing multi-rank collective NKI kernel tests.

    Extends the validation features of UnitTestFramework to collective (multi-rank) tests:
    - Signature validation (kernel_entry ↔ torch_ref)
    - .must_alias_input handling
    - Lazy per-rank golden generation (skipped in compile-only mode)
    - Cross-rank shape/dtype consistency check
    - Per-rank torch_ref input override (for KVDP-style tests)
    - Per-rank custom validation via callable custom_comparator
    """

    def __init__(
        self,
        test_manager: Orchestrator,
        kernel_entry: Callable,
        torch_ref: Callable,
        per_rank_input_generator: Callable[[int], dict],
        collective_ranks: int,
        per_rank_torch_ref_input_override: Optional[Callable[[int, dict], dict]] = None,
        check_unused_params: bool = False,
        collector: Optional[IMetricsCollector] = None,
    ):
        """Initialize the collective test framework.

        Args:
            test_manager: Test orchestrator for execution
            kernel_entry: Kernel function under test
            torch_ref: Torch reference function (signature must match kernel_entry)
            per_rank_input_generator: Function(rank_id: int) -> dict of inputs per rank
            collective_ranks: Number of ranks
            per_rank_torch_ref_input_override: Optional function(rank_id, kernel_input) -> dict
                that transforms kernel inputs before passing to torch_ref. Use for cases
                where golden needs different inputs than kernel (e.g., KVDP=1 for golden).
            check_unused_params: Check for unused parameters in kernel_entry
            collector: Optional metrics collector
        """
        validate_torch_ref_signature(kernel_entry, torch_ref)
        if check_unused_params:
            check_unused_parameters(kernel_entry)

        self.test_manager = test_manager
        self.kernel_entry = kernel_entry
        self.torch_ref = torch_ref
        self.per_rank_input_generator = per_rank_input_generator
        self.collective_ranks = collective_ranks
        self.per_rank_torch_ref_input_override = per_rank_torch_ref_input_override
        self.collector = collector

    def run_test(
        self,
        test_config,
        compiler_args: CompilerArgs,
        rtol: float = 1e-2,
        atol: float = 1e-2,
        inference_args: Optional[InferenceArgs] = None,
        custom_comparator: Optional[Callable] = None,
        metadata: Optional[dict] = None,
    ):
        """Execute a collective test case.

        Args:
            test_config: Test configuration (unused, for API consistency)
            compiler_args: Compiler arguments
            rtol: Relative tolerance
            atol: Absolute tolerance
            inference_args: Optional inference arguments (overrides collective_ranks)
            custom_comparator: Optional callable(rank_id, golden_dict) -> dict mapping
                output names to CustomValidatorWithOutputTensorData. The framework runs
                torch_ref per rank to produce golden_dict, then passes it to this function
                to build custom validators. This keeps golden generation in the framework
                while allowing custom comparison logic (e.g., cosine similarity).
            metadata: Optional dict with 'config_name' and 'key' for metrics
        """

        if self.collector is not None and metadata is not None:
            metadata_list = load_model_configs(metadata["config_name"])
            self.collector.match_and_add_metadata_dimensions(metadata["key"], metadata_list)

        # Validate rank-0 inputs
        rank0_input = self.per_rank_input_generator(rank_id=0)
        validate_input_keys(rank0_input, self.kernel_entry)

        # Cross-rank shape/dtype consistency check
        if self.collective_ranks >= 2:
            validate_cross_rank_consistency(self.per_rank_input_generator, rank0_input, self.collective_ranks)

        # Build per-rank input generator with .must_alias_input filtering
        def _filtered_input_generator(rank_id: int) -> dict:
            raw = self.per_rank_input_generator(rank_id=rank_id)
            return filter_kernel_input(raw, self.kernel_entry)

        # Build per-rank golden generator via torch_ref
        torch_ref = self.torch_ref

        def _golden_generator(rank_id: int) -> dict:
            raw_input = self.per_rank_input_generator(rank_id=rank_id)
            if self.per_rank_torch_ref_input_override is not None:
                ref_input_raw = self.per_rank_torch_ref_input_override(rank_id, raw_input)
            else:
                ref_input_raw = raw_input
            ref_input = filter_ref_input(ref_input_raw, torch_ref)
            golden = torch_ref(**ref_input)
            if custom_comparator is not None:
                return custom_comparator(rank_id, golden)
            return golden

        # Resolve validation args
        validation_args = ValidationArgs(
            golden_output=PerRankLazyGoldenGenerator(_golden_generator),
            relative_accuracy=rtol,
            absolute_accuracy=atol,
        )

        per_rank_input = PerRankLazyInputGenerator(_filtered_input_generator)
        per_rank_input.base_input = rank0_input

        kernel_args = KernelArgs(
            kernel_func=self.kernel_entry,
            compiler_input=compiler_args,
            kernel_input=per_rank_input,
            inference_args=inference_args or InferenceArgs(collective_ranks=self.collective_ranks),
            validation_args=validation_args,
        )
        self.test_manager.execute(kernel_args)


# --- Shared Validation Helpers ---


def validate_input_keys(kernel_input: dict, kernel_entry: Callable) -> None:
    """Validate kernel_input keys against kernel_entry signature.

    Checks for extra keys not in the signature and missing required parameters.
    Handles .must_alias_input suffix transparently.
    """
    sig = signature(kernel_entry)
    kernel_params = set(sig.parameters.keys())

    extra_keys = []
    for k in kernel_input.keys():
        base_key = k.rsplit(".must_alias_input", 1)[0] if k.endswith(".must_alias_input") else k
        if base_key not in kernel_params:
            extra_keys.append(k)
    if extra_keys:
        raise ValueError(
            f"kernel_input has keys {extra_keys} that don't match kernel_entry signature. "
            f"Expected parameters: {sorted(kernel_params)}"
        )

    missing_required = []
    for param_name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            if param_name not in kernel_input and f"{param_name}.must_alias_input" not in kernel_input:
                missing_required.append(param_name)
    if missing_required:
        raise ValueError(f"kernel_input missing required parameters: {missing_required}")


def filter_kernel_input(kernel_input: dict, kernel_entry: Callable) -> dict:
    """Filter kernel_input to only parameters accepted by kernel_entry.

    Handles .must_alias_input suffix transparently.
    """
    kernel_params = set(signature(kernel_entry).parameters.keys())
    filtered = {}
    for k, v in kernel_input.items():
        if k in kernel_params:
            filtered[k] = v
        elif k.endswith(".must_alias_input"):
            base_key = k.rsplit(".must_alias_input", 1)[0]
            if base_key in kernel_params:
                filtered[k] = v
    return filtered


def filter_ref_input(kernel_input: dict, torch_ref: Callable) -> dict:
    """Filter kernel_input to only parameters accepted by torch_ref.

    Handles .must_alias_input suffix: strips suffix and copies arrays to avoid aliasing.
    """
    ref_params = set(signature(torch_ref).parameters.keys())
    ref_input = {}
    for k, v in kernel_input.items():
        if k in ref_params:
            ref_input[k] = v
        elif k.endswith(".must_alias_input"):
            base_key = k.rsplit(".must_alias_input", 1)[0]
            if base_key in ref_params:
                ref_input[base_key] = v.copy() if hasattr(v, "copy") else v
    return ref_input


def validate_cross_rank_consistency(
    per_rank_input_generator: Callable[[int], dict], rank0_input: dict, num_ranks: int
) -> None:
    """Validate input tensor shapes/dtypes are consistent across all ranks."""
    for rank_id in range(1, num_ranks):
        rank_input = per_rank_input_generator(rank_id=rank_id)
        for key in rank0_input:
            v0, vr = rank0_input[key], rank_input.get(key)
            if isinstance(v0, np.ndarray) and isinstance(vr, np.ndarray):
                if v0.shape != vr.shape:
                    raise ValueError(
                        f"Input tensor '{key}' shape mismatch across ranks: rank 0 {v0.shape} vs rank {rank_id} {vr.shape}"
                    )
                if v0.dtype != vr.dtype:
                    raise ValueError(
                        f"Input tensor '{key}' dtype mismatch across ranks: rank 0 {v0.dtype} vs rank {rank_id} {vr.dtype}"
                    )


# --- Helper Functions ---


def _validate_key_sets(expected: set, actual: set, msg_header: str, expected_label: str, actual_label: str) -> None:
    """Raise ValueError if two key sets don't match, with a descriptive diff message."""
    if expected != actual:
        missing = expected - actual
        extra = actual - expected
        msg = msg_header
        if missing:
            msg += f"\n  {expected_label}: {sorted(missing)}"
        if extra:
            msg += f"\n  {actual_label}: {sorted(extra)}"
        raise ValueError(msg)


def validate_torch_ref_signature(kernel_entry: Callable, torch_ref: Callable) -> None:
    """Validate torch reference signature matches kernel signature."""
    kernel_params = set(signature(kernel_entry).parameters.keys())
    ref_params = set(signature(torch_ref).parameters.keys())
    _validate_key_sets(
        expected=kernel_params,
        actual=ref_params,
        msg_header="Torch ref signature mismatch with kernel:",
        expected_label="Missing in torch_ref",
        actual_label="Extra in torch_ref",
    )


def check_unused_parameters(func: Callable) -> None:
    """Raise error if any parameter in func's signature appears unused in the function body.

    This helps catch bugs where a wrapper accepts a parameter but forgets to pass it through.

    Raises:
        ValueError: If a parameter appears only in the signature (likely unused).
    """
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return  # Can't get source, skip check

    sig = signature(func)
    unused = []
    for param_name in sig.parameters:
        # Count occurrences - if only 1, it's just in the signature
        if source.count(param_name) == 1:
            unused.append(param_name)

    if unused:
        raise ValueError(
            f"Parameters {unused} may be unused in {func.__name__}. "
            "Ensure all parameters are forwarded to the underlying function."
        )


def torch_ref_wrapper(
    torch_ref_func: Callable,
    preserve_lower_precision: bool = False,
    input_dtype_converter: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
    output_dtype_converter: Optional[Callable[[torch.Tensor], np.ndarray]] = None,
) -> Callable:
    """Wrap a torch reference function to handle numpy<->torch conversion.

    Converts numpy arrays to torch tensors (float16->float32 for CPU compatibility),
    calls the torch reference, and converts results back to numpy.

    Args:
        torch_ref_func: Torch reference function that takes torch tensors as kwargs
        preserve_lower_precision: If True, cast output tensors back to the original
            input dtype (e.g., bfloat16) using neuron_dtypes.static_cast. This matches
            the kernel's output precision, enabling tighter validation tolerances.
            Computation still happens in float32 for numerical stability.
        input_dtype_converter: Optional callback to customize numpy->torch dtype conversion.
            Called with (numpy_array,) for every numpy input, before any default
            conversion. If it returns a torch tensor, that value is used directly
            (overriding all default behavior). Return None to fall back to the default
            conversion. The callback is responsible for converting numpy to torch. This
            allows callers to preserve fp8 or other custom dtypes that the default
            wrapper upcasts to fp32.
        output_dtype_converter: Optional callback to customize torch->numpy dtype conversion.
            Called with (torch_tensor,). Return a numpy array, or None to use default
            behavior. This allows callers to handle fp8 or other custom output dtypes.

    Returns:
        Wrapped function that takes numpy arrays and returns numpy arrays

    Example:
        # Default usage:
        @torch_ref_wrapper
        def my_kernel_torch_ref(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return torch.matmul(input, weight)

        # Can now call with numpy arrays:
        result = my_kernel_torch_ref(input=np_array, weight=np_weight)

        # With custom fp8 handling:
        def fp8_input_converter(value):
            if str(value.dtype) == 'float8_e4m3fn':
                return torch.from_numpy(value.astype(np.float32)).to(torch.float8_e4m3fn)
            return None  # use default

        def fp8_output_converter(tensor):
            if tensor.dtype == torch.float8_e4m3fn:
                return ndt.static_cast(tensor.float().numpy(), 'float8_e4m3fn')
            return None  # use default

        torch_ref=torch_ref_wrapper(my_torch_ref,
            input_dtype_converter=fp8_input_converter,
            output_dtype_converter=fp8_output_converter)
    """

    @functools.wraps(torch_ref_func)
    def wrapped(**kwargs):
        # Convert numpy arrays to torch tensors (float16/bfloat16->float32 for CPU)
        torch_kwargs = {}
        original_dtype = None
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                dtype_str = str(value.dtype)
                # Try custom converter first; it overrides all default behavior
                if input_dtype_converter is not None:
                    converted = input_dtype_converter(value)
                    if converted is not None:
                        torch_kwargs[key] = converted
                        continue
                # Handle MX packed x4 types: pass as numpy for torch ref to handle
                if 'x4' in dtype_str:
                    torch_kwargs[key] = value
                    continue
                # Track original dtype for output cast-back
                if original_dtype is None and (
                    'bfloat16' in dtype_str or 'float8' in dtype_str or 'float16' in dtype_str
                ):
                    original_dtype = dtype_str
                # Make value safe for torch.from_numpy (doesn't support uint32/bfloat16/fp8)
                if value.dtype == np.uint32:
                    value = value.astype(np.int32)
                elif 'bfloat16' in dtype_str or 'float8' in dtype_str:
                    value = value.astype(np.float32)
                tensor = torch.from_numpy(value)
                # Default dtype conversions
                if tensor.dtype == torch.float16:
                    tensor = tensor.float()
                elif preserve_lower_precision and 'bfloat16' in dtype_str:
                    tensor = tensor.to(torch.bfloat16)
                torch_kwargs[key] = tensor
            else:
                torch_kwargs[key] = value

        # Call torch reference
        result = torch_ref_func(**torch_kwargs)

        # Convert result back to numpy
        def _tensor_to_numpy(t):
            if isinstance(t, torch.Tensor):
                # Try custom output converter first
                if output_dtype_converter is not None:
                    converted = output_dtype_converter(t)
                    if converted is not None:
                        return converted
                if t.dtype == torch.bfloat16:
                    t = t.float()
                np_val = t.numpy()
                if preserve_lower_precision and original_dtype is not None:
                    np_val = ndt.static_cast(np_val, original_dtype)
                return np_val
            return t

        if isinstance(result, torch.Tensor):
            return {"out": _tensor_to_numpy(result)}
        elif isinstance(result, dict):
            return {k: _tensor_to_numpy(v) for k, v in result.items()}
        else:
            return result

    return wrapped
