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
"""NKI Library Testing — shared test utilities and pytest fixtures."""

import sys


class _NkilibTestingRedirector:
    """Redirect ``test.utils.*`` imports to ``nkilib_testing.*`` when available.

    Installed on ``sys.meta_path`` so that the very first import of any
    ``test.utils.<submodule>`` reuses the module object that the pytest11
    entry point already loaded under the ``nkilib_testing`` namespace.
    """

    def find_module(self, fullname, path=None):
        if fullname.startswith("test.utils."):
            alias = fullname.replace("test.utils.", "nkilib_testing.", 1)
            if alias in sys.modules:
                return self
        return None

    def load_module(self, fullname):
        if fullname not in sys.modules:
            alias = fullname.replace("test.utils.", "nkilib_testing.", 1)
            sys.modules[fullname] = sys.modules[alias]
        return sys.modules[fullname]


# Only install when running from the source tree (loaded as test.utils),
# not from the installed wheel (loaded as nkilib_testing).
#
# The nkilib_testing pytest plugin is auto-loaded from the installed wheel via
# its pytest11 entry point, creating nkilib_testing.* module objects.  Source-tree
# code imports the same modules as test.utils.* via relative imports.  Without the
# redirector Python would create separate module objects for each path, breaking
# isinstance checks, mock.patch targets, and class registries.
if __name__ == "test.utils":
    sys.meta_path.insert(0, _NkilibTestingRedirector())
