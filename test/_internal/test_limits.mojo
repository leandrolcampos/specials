# ===----------------------------------------------------------------------=== #
# Copyright 2023 The Specials Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %build_dir %assertion_flag %debug_level %sanitize_checks %s

"""Tests for machine limits of IEEE 754-2008 binary floating-point types."""

from python import Python

from specials._internal.limits import FloatLimits
from test_utils import UnitTest


fn test_eps[type: DType]() raises:
    var unit_test = UnitTest("test_eps_" + str(type))

    var np = Python.import_module("numpy")

    var actual = FloatLimits[type].eps
    var expected = np.finfo(str(type)).eps.to_float64().cast[type]()

    unit_test.assert_equal(actual, expected)


fn test_epsneg[type: DType]() raises:
    var unit_test = UnitTest("test_epsneg_" + str(type))

    var np = Python.import_module("numpy")

    var actual = FloatLimits[type].epsneg
    var expected = np.finfo(str(type)).epsneg.to_float64().cast[type]()

    unit_test.assert_equal(actual, expected)


fn test_min[type: DType]() raises:
    var unit_test = UnitTest("test_min_" + str(type))

    var np = Python.import_module("numpy")

    var actual = FloatLimits[type].min
    var expected = np.finfo(str(type)).smallest_normal.to_float64().cast[type]()

    unit_test.assert_equal(actual, expected)


fn test_max[type: DType]() raises:
    var unit_test = UnitTest("test_max_" + str(type))

    var np = Python.import_module("numpy")

    var actual = FloatLimits[type].max
    var expected = np.finfo(str(type)).max.to_float64().cast[type]()

    unit_test.assert_equal(actual, expected)


fn main() raises:
    test_eps[DType.float16]()
    test_eps[DType.float32]()
    test_eps[DType.float64]()

    test_epsneg[DType.float16]()
    test_epsneg[DType.float32]()
    test_epsneg[DType.float64]()

    test_min[DType.float16]()
    test_min[DType.float32]()
    test_min[DType.float64]()

    test_max[DType.float16]()
    test_max[DType.float32]()
    test_max[DType.float64]()
