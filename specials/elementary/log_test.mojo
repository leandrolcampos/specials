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

"""Tests for the natural logarithmic function."""

import math

from python import Python
from utils.static_tuple import StaticTuple

from specials._internal.limits import FloatLimits
from specials._internal.testing import UnitTest
from specials.elementary.log import log


fn _mp_log[dtype: DType](x: Scalar[dtype]) raises -> Scalar[dtype]:
    var mp = Python.import_module("mpmath")
    var result = mp.log(mp.mpf(x))
    return result.to_float64().cast[dtype]()


fn test_log[dtype: DType]() raises:
    var unit_test = UnitTest("test_log_" + str(dtype))

    var xmin = FloatLimits[dtype].min
    var epsneg = FloatLimits[dtype].epsneg
    var eps = FloatLimits[dtype].eps
    var xmax = FloatLimits[dtype].max

    var xs = StaticTuple[7, Scalar[dtype]](
        xmin, epsneg, eps, 1.0 - epsneg, 1.0, 1.0 + eps, xmax
    )

    var rtol: Scalar[dtype]

    @parameter
    if dtype == DType.float32:
        rtol = 1e-6
    else:  # dtype == DType.float64
        rtol = 1e-12

    for i in range(len(xs)):
        var x = xs[i]
        var expected = _mp_log[dtype](x)
        var actual = log(x)

        unit_test.assert_all_close(actual, expected, 0.0, rtol)


fn test_log_special_cases[dtype: DType]() raises:
    var unit_test = UnitTest("test_log_special_cases_" + str(dtype))

    var xmin = FloatLimits[dtype].min
    var nan = math.nan[dtype]()
    var inf = math.limit.inf[dtype]()

    var x = SIMD[dtype, 4](-xmin, 0.0, nan, inf)

    var expected = SIMD[dtype, 4](nan, -inf, nan, inf)
    var actual = log(x)

    # Here NaNs are compared like numbers and no assertion is raised if both objects
    # have NaNs in the same positions.
    var result = (
        (math.isnan(actual) & math.isnan(actual)) | (actual == expected)
    ).reduce_and()
    unit_test.assert_true(result, str(actual) + " is not equal to " + str(expected))


fn main() raises:
    # Setting the mpmath precision for this module
    var mp = Python.import_module("mpmath")
    mp.mp.dps = 50

    test_log[DType.float64]()
    test_log[DType.float32]()

    test_log_special_cases[DType.float64]()
    test_log_special_cases[DType.float32]()
