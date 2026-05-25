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
"""Verify all integration test modules can be imported without errors.

Catches broken cross-module imports (e.g., importing a removed symbol)
at build time, before code is merged.
"""

import importlib
import pkgutil

import test.integration.nkilib as _root


def test_all_integration_modules_importable():
    errors = []
    for _, modname, _ in pkgutil.walk_packages(_root.__path__, _root.__name__ + "."):
        try:
            importlib.import_module(modname)
        except Exception as exc:
            errors.append(f"{modname}: {type(exc).__name__}: {exc}")

    assert not errors, f"{len(errors)} module(s) failed to import:\n" + "\n".join(f"  • {e}" for e in errors)
