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

"""Unit tests for per-attempt rerun-tracking dimensions (add_rerun_dimensions).

Verifies every attempt is individually attributable via AttemptNumber, and a
failed attempt records a (truncated) FailureReason. Flaky-pass detection and
original-failure-cause preservation live in the KaenaKernelsTools processor.
"""

from ..utils.metrics_collector import MAX_FAILURE_REASON_LEN, add_rerun_dimensions


class _FakeCollector:
    def __init__(self):
        self.dimensions: dict[str, str] = {}

    def add_dimension(self, dims: dict[str, str]) -> None:
        self.dimensions.update(dims)


def test_first_attempt_number():
    collector = _FakeCollector()
    add_rerun_dimensions(collector, attempt_number=1, failed=False)

    assert collector.dimensions["AttemptNumber"] == "1"
    assert "FailureReason" not in collector.dimensions


def test_rerun_attempt_number():
    collector = _FakeCollector()
    add_rerun_dimensions(collector, attempt_number=2, failed=False)

    assert collector.dimensions["AttemptNumber"] == "2"


def test_failed_attempt_records_failure_reason():
    collector = _FakeCollector()
    add_rerun_dimensions(collector, attempt_number=1, failed=True, failure_reason="allclose mismatch: max diff 0.5")

    assert collector.dimensions["FailureReason"] == "allclose mismatch: max diff 0.5"


def test_failure_reason_newlines_collapsed():
    collector = _FakeCollector()
    add_rerun_dimensions(collector, attempt_number=1, failed=True, failure_reason="line1\nline2")

    assert collector.dimensions["FailureReason"] == "line1 line2"


def test_failure_reason_is_truncated():
    collector = _FakeCollector()
    add_rerun_dimensions(collector, attempt_number=1, failed=True, failure_reason="x" * 1000)

    assert len(collector.dimensions["FailureReason"]) <= MAX_FAILURE_REASON_LEN


def test_failed_attempt_without_reason_omits_field():
    collector = _FakeCollector()
    add_rerun_dimensions(collector, attempt_number=1, failed=True, failure_reason=None)

    assert collector.dimensions["AttemptNumber"] == "1"
    assert "FailureReason" not in collector.dimensions
