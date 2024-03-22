# ===----------------------------------------------------------------------=== #
# Copyright 2024 The Specials Authors.
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

"""Tests for the `exp2` function."""

import math

from python import Python
from utils.static_tuple import StaticTuple

from specials._internal.limits import FloatLimits
from specials._internal.testing import UnitTest
from specials.elementary.exp2 import exp2


fn _mp_exp2[dtype: DType](x: Scalar[dtype]) raises -> Scalar[dtype]:
    var mp = Python.import_module("mpmath")
    var result = mp.power(mp.mpf(2), mp.mpf(x))
    return result.to_float64().cast[dtype]()


fn test_exp2[dtype: DType]() raises:
    var unit_test = UnitTest("test_exp2_" + str(dtype))

    var xs = StaticTuple[9, Scalar[dtype]](
        -10.0, -1.0, -0.1, -0.01, 0.0, 0.01, 0.1, 1.0, 10.0
    )

    var rtol: Scalar[dtype]

    @parameter
    if dtype == DType.float32:
        rtol = 1e-6
    else:  # dtype == DType.float64
        rtol = 1e-15

    for i in range(len(xs)):
        var x = xs[i]
        var expected = _mp_exp2[dtype](x)
        var actual = exp2(x)

        unit_test.assert_all_close(actual, expected, 0.0, rtol)


fn test_exp2_special_cases[dtype: DType]() raises:
    var unit_test = UnitTest("test_exp2_special_cases_" + str(dtype))

    var xmin = Scalar[dtype](FloatLimits[dtype].minexp)
    var xeps = 0.5 * FloatLimits[dtype].epsneg
    var xmax = math.nextafter(Scalar[dtype](FloatLimits[dtype].maxexp), 0.0)
    var nan = math.nan[dtype]()
    var inf = math.limit.inf[dtype]()

    var xs = StaticTuple[13, Scalar[dtype]](
        nan,
        -inf,
        xmin - 1.0,
        xmin,
        xmin + 1.0,
        -xeps,
        -0.1 * xeps,
        0.0,
        0.1 * xeps,
        xmax - 1.0,
        xmax,
        xmax + 1.0,
        inf,
    )

    var rtol: Scalar[dtype]

    @parameter
    if dtype == DType.float32:
        rtol = 1e-6
    else:  # dtype == DType.float64
        rtol = 1e-15

    for i in range(len(xs)):
        var x = xs[i]
        var actual = exp2(x)
        var expected: Scalar[dtype]

        if math.isnan(x):
            expected = nan
        elif x > xmax:
            expected = inf
        else:
            expected = _mp_exp2[dtype](x)

        unit_test.assert_all_close(actual, expected, 0.0, rtol)


fn main() raises:
    # Setting the mpmath precision for this module
    var mp = Python.import_module("mpmath")
    mp.mp.dps = 100

    test_exp2[DType.float64]()
    test_exp2[DType.float32]()

    test_exp2_special_cases[DType.float64]()
    test_exp2_special_cases[DType.float32]()
