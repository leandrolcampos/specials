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

"""Tests for elementary functions."""

import math
import testing

from python import Python
from python.object import PythonObject
from utils.static_tuple import StaticTuple

import specials

from specials._internal.limits import FloatLimits
from specials._internal.testing import UnitTest


fn _mp_exp[dtype: DType](x: SIMD[dtype, 1]) raises -> SIMD[dtype, 1]:
    let mp = Python.import_module("mpmath")
    let result = mp.exp(mp.mpf(x))
    return result.to_float64().cast[dtype]()


fn test_exp[dtype: DType]() raises:
    let unit_test = UnitTest("test_exp_" + str(dtype))

    let xmin = specials.log(FloatLimits[dtype].min)
    var xmax = specials.log(FloatLimits[dtype].max)
    let xeps = FloatLimits[dtype].eps

    if dtype == DType.float32:
        xmax *= 1.0 - FloatLimits[dtype].epsneg
    else:
        pass

    let xs = StaticTuple[5, SIMD[dtype, 1]](xmin, 0.0, 0.5 * xeps, 1.0, xmax)

    let rtol: SIMD[dtype, 1]

    @parameter
    if dtype == DType.float32:
        rtol = 1e-6
    else:  # dtype == DType.float64
        rtol = 1e-12

    for i in range(len(xs)):
        let x = xs[i]
        let expected = _mp_exp[dtype](x)
        let actual = specials.exp(x)

        unit_test.assert_almost_equal(expected, actual, 0.0, rtol)


fn test_exp_special_cases[dtype: DType]() raises:
    let unit_test = UnitTest("test_exp_special_cases_" + str(dtype))

    let xmin = specials.log(FloatLimits[dtype].min)
    let xeps = FloatLimits[dtype].eps
    let nan = math.nan[dtype]()
    let inf = math.limit.inf[dtype]()

    let x = SIMD[dtype, 4](2.0 * xmin, inf, 0.1 * xeps, nan)

    let expected = SIMD[dtype, 4](0.0, inf, 1.0, nan)
    let actual = specials.exp(x)

    # Here NaNs are compared like numbers and no assertion is raised if both objects
    # have NaNs in the same positions.
    let result = (
        (math.isnan(actual) & math.isnan(actual)) | (actual == expected)
    ).reduce_and()
    unit_test.assert_true(result, str(expected) + " is not equal to " + str(actual))


fn _mp_log[dtype: DType](x: SIMD[dtype, 1]) raises -> SIMD[dtype, 1]:
    let mp = Python.import_module("mpmath")
    let result = mp.log(mp.mpf(x))
    return result.to_float64().cast[dtype]()


fn test_log[dtype: DType]() raises:
    let unit_test = UnitTest("test_log_" + str(dtype))

    let xmin = FloatLimits[dtype].min
    let epsneg = FloatLimits[dtype].epsneg
    let eps = FloatLimits[dtype].eps
    let xmax = FloatLimits[dtype].max

    let xs = StaticTuple[7, SIMD[dtype, 1]](
        xmin, epsneg, eps, 1.0 - epsneg, 1.0, 1.0 + eps, xmax
    )

    let rtol: SIMD[dtype, 1]

    @parameter
    if dtype == DType.float32:
        rtol = 1e-6
    else:  # dtype == DType.float64
        rtol = 1e-12

    for i in range(len(xs)):
        let x = xs[i]
        let expected = _mp_log[dtype](x)
        let actual = specials.log(x)

        unit_test.assert_almost_equal(expected, actual, 0.0, rtol)


fn test_log_special_cases[dtype: DType]() raises:
    let unit_test = UnitTest("test_log_special_cases_" + str(dtype))

    let xmin = FloatLimits[dtype].min
    let nan = math.nan[dtype]()
    let inf = math.limit.inf[dtype]()

    let x = SIMD[dtype, 4](-xmin, 0.0, nan, inf)

    let expected = SIMD[dtype, 4](nan, -inf, nan, inf)
    let actual = specials.log(x)

    # Here NaNs are compared like numbers and no assertion is raised if both objects
    # have NaNs in the same positions.
    let result = (
        (math.isnan(actual) & math.isnan(actual)) | (actual == expected)
    ).reduce_and()
    unit_test.assert_true(result, str(expected) + " is not equal to " + str(actual))


fn main() raises:
    # Setting the mpmath precision for this module
    let mp = Python.import_module("mpmath")
    mp.mp.dps = 50

    # exp
    test_exp[DType.float64]()
    test_exp[DType.float32]()

    test_exp_special_cases[DType.float64]()
    test_exp_special_cases[DType.float32]()

    # log
    test_log[DType.float64]()
    test_log[DType.float32]()

    test_log_special_cases[DType.float64]()
    test_log_special_cases[DType.float32]()
