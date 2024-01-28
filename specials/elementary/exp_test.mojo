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

"""Tests for the exponential function."""

import math

from python import Python
from utils.static_tuple import StaticTuple

from specials._internal.limits import FloatLimits
from specials._internal.testing import UnitTest
from specials.elementary.exp import exp
from specials.elementary.log import log


fn _mp_exp[dtype: DType](x: SIMD[dtype, 1]) raises -> SIMD[dtype, 1]:
    let mp = Python.import_module("mpmath")
    let result = mp.exp(mp.mpf(x))
    return result.to_float64().cast[dtype]()


fn test_exp[dtype: DType]() raises:
    let unit_test = UnitTest("test_exp_" + str(dtype))

    let xmin = log(FloatLimits[dtype].min)
    var xmax = log(FloatLimits[dtype].max)
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
        let actual = exp(x)

        unit_test.assert_all_close(actual, expected, 0.0, rtol)


fn test_exp_special_cases[dtype: DType]() raises:
    let unit_test = UnitTest("test_exp_special_cases_" + str(dtype))

    let xmin = log(FloatLimits[dtype].min)
    let xeps = FloatLimits[dtype].eps
    let nan = math.nan[dtype]()
    let inf = math.limit.inf[dtype]()

    let x = SIMD[dtype, 4](2.0 * xmin, inf, 0.1 * xeps, nan)

    let expected = SIMD[dtype, 4](0.0, inf, 1.0, nan)
    let actual = exp(x)

    # Here NaNs are compared like numbers and no assertion is raised if both objects
    # have NaNs in the same positions.
    let result = (
        (math.isnan(actual) & math.isnan(actual)) | (actual == expected)
    ).reduce_and()
    unit_test.assert_true(result, str(actual) + " is not equal to " + str(expected))


fn main() raises:
    # Setting the mpmath precision for this module
    let mp = Python.import_module("mpmath")
    mp.mp.dps = 50

    test_exp[DType.float64]()
    test_exp[DType.float32]()

    test_exp_special_cases[DType.float64]()
    test_exp_special_cases[DType.float32]()
