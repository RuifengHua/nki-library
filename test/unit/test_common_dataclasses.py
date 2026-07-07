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
import pytest

from test.utils.common_dataclasses import Platforms


class TestPlatformsIsTrn3:
    @pytest.mark.parametrize(
        "platform,expected",
        [
            (Platforms.TRN1, False),
            (Platforms.TRN2, False),
            (Platforms.TRN3, True),
            (Platforms.TRN3_A0, True),
        ],
    )
    def test_is_trn3(self, platform, expected):
        assert platform.is_trn3() == expected


class TestPlatformsGetCompileTarget:
    @pytest.mark.parametrize(
        "platform,expected",
        [
            (Platforms.TRN1, "trn1"),
            (Platforms.TRN2, "trn2"),
            (Platforms.TRN3, "trn3"),
            (Platforms.TRN3_A0, "trn3pre"),
            (Platforms.TRN3_PDS, "trn3"),
            (Platforms.TRN3_PDS_A0, "trn3pre"),
        ],
    )
    def test_get_compile_target(self, platform, expected):
        assert platform.get_compile_target() == expected


from test.utils.common_dataclasses import (
    ModelTestType,
    Platforms,
    _iter_model_configs,
    is_model_test_type,
    prepare_model_parametrize,
)


class TestModelTestType:
    def test_test_id_prefix(self):
        assert ModelTestType.BROAD.test_id_prefix == "BROAD"
        assert ModelTestType.GENERALITY.test_id_prefix == "GENERALITY"
        assert ModelTestType.OPTIMAL.test_id_prefix == "OPTIMAL"


class TestIsModelTestType:
    @pytest.mark.parametrize(
        "test_type,expected",
        [
            ("BROAD", True),
            ("BROAD_some_test", True),
            ("GENERALITY_test", True),
            ("OPTIMAL_config", True),
            ("TIER0_ln-2", True),
            ("manual", False),
            ("random", False),
        ],
    )
    def test_is_model_test_type(self, test_type, expected):
        assert is_model_test_type(test_type) == expected


class TestIterModelConfigs:
    def test_dict_format(self):
        configs = {
            ModelTestType.BROAD: [[1, 2], [3, 4]],
            ModelTestType.OPTIMAL: [[5, 6]],
        }
        result = list(_iter_model_configs(configs))
        assert result == [
            (ModelTestType.BROAD, [1, 2], None),
            (ModelTestType.BROAD, [3, 4], None),
            (ModelTestType.OPTIMAL, [5, 6], None),
        ]

    def test_flat_list_format(self):
        configs = [[1, 2], [3, 4]]
        result = list(_iter_model_configs(configs))
        assert result == [
            (ModelTestType.BROAD, [1, 2], None),
            (ModelTestType.BROAD, [3, 4], None),
        ]

    def test_platform_restricted_entry(self):
        platforms = {Platforms.TRN3, Platforms.TRN3_A0}
        configs = {
            ModelTestType.TIER0: [
                ([1, 2], platforms),
                [3, 4],
            ],
        }
        result = list(_iter_model_configs(configs))
        assert result == [
            (ModelTestType.TIER0, [1, 2], platforms),
            (ModelTestType.TIER0, [3, 4], None),
        ]


class TestPrepareModelParametrize:
    def test_dict_format(self):
        configs = {
            ModelTestType.BROAD: [[1, 2]],
            ModelTestType.GENERALITY: [[3, 4]],
        }
        params, ids = prepare_model_parametrize(configs)
        assert params == [[1, 2], [3, 4]]
        assert ids == ["BROAD_1-2", "GENERALITY_3-4"]

    def test_flat_list_format(self):
        configs = [[1, 2], [3, 4]]
        params, ids = prepare_model_parametrize(configs)
        assert params == [[1, 2], [3, 4]]
        assert ids == ["BROAD_1-2", "BROAD_3-4"]

    def test_custom_id_formatter(self):
        configs = {ModelTestType.OPTIMAL: [[10, 20]]}
        params, ids = prepare_model_parametrize(configs, id_formatter=lambda p: f"x{p[0]}")
        assert ids == ["OPTIMAL_x10"]

    def test_empty_dict(self):
        params, ids = prepare_model_parametrize({})
        assert params == []
        assert ids == []


from test.utils.common_dataclasses import unpack_model_config


class TestUnpackModelConfig:
    def test_tuple_entry(self):
        mt, params = unpack_model_config((ModelTestType.GENERALITY, [1, 2, 3]))
        assert mt == ModelTestType.GENERALITY
        assert params == [1, 2, 3]

    def test_raw_list_defaults_to_broad(self):
        mt, params = unpack_model_config([1, 2, 3])
        assert mt == ModelTestType.BROAD
        assert params == [1, 2, 3]


from test.utils.common_dataclasses import LazyGoldenGenerator, ValidationArgs


class TestValidationArgsEqualNanInf:
    def test_default_equal_nan_inf_is_false(self):
        golden = LazyGoldenGenerator(lazy_golden_generator=lambda: {}, output_ndarray={})
        args = ValidationArgs(golden_output=golden)
        assert args.equal_nan_inf is False

    def test_equal_nan_inf_can_be_set_true(self):
        golden = LazyGoldenGenerator(lazy_golden_generator=lambda: {}, output_ndarray={})
        args = ValidationArgs(golden_output=golden, equal_nan_inf=True)
        assert args.equal_nan_inf is True
