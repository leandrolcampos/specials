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

"""Tests for machine limits of IEEE 754-2008 binary floating-point types."""

from python import Python

from specials._internal.limits import FloatLimits
from specials._internal.testing import UnitTest


fn test_eps[dtype: DType]() raises:
    let unit_test = UnitTest("test_eps_" + str(dtype))

    let np = Python.import_module("numpy")

    let actual = FloatLimits[dtype].eps
    let expected = np.finfo(str(dtype)).eps.to_float64().cast[dtype]()

    unit_test.assert_equal(actual, expected)


fn test_epsneg[dtype: DType]() raises:
    let unit_test = UnitTest("test_epsneg_" + str(dtype))

    let np = Python.import_module("numpy")

    let actual = FloatLimits[dtype].epsneg
    let expected = np.finfo(str(dtype)).epsneg.to_float64().cast[dtype]()

    unit_test.assert_equal(actual, expected)


fn test_min[dtype: DType]() raises:
    let unit_test = UnitTest("test_min_" + str(dtype))

    let np = Python.import_module("numpy")

    let actual = FloatLimits[dtype].min
    let expected = np.finfo(str(dtype)).smallest_normal.to_float64().cast[dtype]()

    unit_test.assert_equal(actual, expected)


fn test_max[dtype: DType]() raises:
    let unit_test = UnitTest("test_max_" + str(dtype))

    let np = Python.import_module("numpy")

    let actual = FloatLimits[dtype].max
    let expected = np.finfo(str(dtype)).max.to_float64().cast[dtype]()

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
