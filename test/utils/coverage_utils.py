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
"""Coverage data extraction utilities for pytest-cov integration."""

import logging

from _pytest.config import Config
from coverage.results import Numbers

from .metrics_emitter import CoverageData

# Match the precision used by pytest-cov's terminal report (6 decimal places for rates).
_COVERAGE_PRECISION = 6


def get_coverage_data(config: Config) -> CoverageData | None:
    """Extract coverage metrics from pytest-cov's in-memory data.

    Returns None when:
    - pytest-cov plugin is not loaded (--cov not passed, e.g. in dry-run builds)
    - No coverage data was collected (e.g. no tests ran)
    - Any error occurs during extraction

    This accesses the coverage.Coverage object directly rather than reading the XML
    file (which is written later in pytest_unconfigure by pytest-cov).
    """
    try:
        cov_plugin = config.pluginmanager.get_plugin("_cov")
        if not cov_plugin or not hasattr(cov_plugin, "cov_controller") or not cov_plugin.cov_controller:
            return None

        cov = cov_plugin.cov_controller.cov
        if not cov:
            return None

        totals = Numbers(precision=_COVERAGE_PRECISION)
        for filename in cov.get_data().measured_files():
            analysis = cov._analyze(filename)
            totals += analysis.numbers

        branches_valid = totals.n_branches
        branches_covered = branches_valid - totals.n_missing_branches - totals.n_partial_branches
        lines_valid = totals.n_statements
        lines_covered = totals.n_executed

        return CoverageData(
            BranchRate=round(branches_covered / branches_valid, _COVERAGE_PRECISION) if branches_valid else 0.0,
            LineRate=round(lines_covered / lines_valid, _COVERAGE_PRECISION) if lines_valid else 0.0,
            CoveragePercent=round(totals.pc_covered, 2),
            BranchesCovered=branches_covered,
            BranchesValid=branches_valid,
        )
    except Exception as e:
        logging.warning(f"Failed to extract coverage data from pytest-cov: {e}")
        return None
