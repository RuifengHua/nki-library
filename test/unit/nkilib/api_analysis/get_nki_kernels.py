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

"""Find all NKI kernels in nkilib/core (NKI functions that call nisa, directly or indirectly)."""

from pathlib import Path

from .get_nki_functions import CORE_DIR, KernelLocation, get_nki_functions
from .get_nki_utils import get_nki_utils


def _is_private_kernel(filepath: str, name: str):
    """A kernel is private if file-private, in a *_utils file, or under a utils/ folder."""
    if name.startswith('_'):
        return True
    p = Path(filepath)
    if p.stem.endswith('_utils'):
        return True
    if 'utils' in p.parts:
        return True
    return False


_cache: dict[str, tuple[list[KernelLocation], list[KernelLocation], list[tuple[str, int, str, str]]]] = {}


def get_nki_kernels():
    """Return list of KernelLocation for NKI kernels, in topological order."""
    utils = set(get_nki_utils())
    return [loc for loc in get_nki_functions() if loc not in utils]


def get_nki_kernels_split():
    """Return (public_kernels, private_kernels, violators) lists."""
    if "kernels_split" not in _cache:
        _cache["kernels_split"] = _compute_nki_kernels_split()
    return _cache["kernels_split"]


def _compute_nki_kernels_split():
    kernels = get_nki_kernels()
    public = [loc for loc in kernels if not _is_private_kernel(loc.filepath, loc.name)]
    private = [loc for loc in kernels if _is_private_kernel(loc.filepath, loc.name)]

    # Violation 1: kernels in core/utils/
    core_utils = str(CORE_DIR / "utils")
    violators: list[tuple[str, int, str, str]] = []
    for loc in kernels:
        if loc.filepath.startswith(core_utils + "/") or loc.filepath.startswith(core_utils + "\\"):
            violators.append((loc.filepath, loc.line, loc.name, "kernel in core/utils"))

    # Violation 2: every file with NKI functions must have at least one public kernel
    # unless it's a _utils file or in a utils/ folder
    all_nki_files: set[str] = {loc.filepath for loc in get_nki_functions()}
    public_kernel_files: set[str] = {loc.filepath for loc in kernels if not _is_private_kernel(loc.filepath, loc.name)}

    for filepath in sorted(all_nki_files):
        p = Path(filepath)
        if p.stem.endswith('_utils') or 'utils' in p.parts:
            continue
        if filepath not in public_kernel_files:
            violators.append((filepath, 0, "[file]", "no public kernels"))

    return public, private, violators
