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

"""Tests for the `expm1` function."""

import math

from memory.unsafe import bitcast

from python import Python
from utils.static_tuple import StaticTuple

from specials._internal.limits import FloatLimits
from specials._internal.testing import UnitTest
from specials.elementary.expm1 import expm1
from specials.elementary.log import log


fn _mp_expm1[dtype: DType](x: Scalar[dtype]) raises -> Scalar[dtype]:
    var mp = Python.import_module("mpmath")
    var result = mp.expm1(mp.mpf(x))
    return result.to_float64().cast[dtype]()


fn test_expm1[dtype: DType]() raises:
    var unit_test = UnitTest("test_expm1_" + str(dtype))

    var xeps = FloatLimits[dtype].eps
    var xs = StaticTuple[5, Scalar[dtype]](0.1 * xeps, 0.01, 0.1, 1.0, 10.0)

    var rtol: Scalar[dtype]

    @parameter
    if dtype == DType.float32:
        rtol = 1e-6
    else:  # dtype == DType.float64
        rtol = 1e-15

    for i in range(len(xs)):
        var x = xs[i]
        var expected = _mp_expm1[dtype](x)
        var actual = expm1(x)

        unit_test.assert_all_close(actual, expected, 0.0, rtol)


fn test_expm1_special_cases[dtype: DType]() raises:
    var unit_test = UnitTest("test_expm1_special_cases_" + str(dtype))

    var xeps: Scalar[dtype]
    var xsml_inf: Scalar[dtype]
    var xsml_sup: Scalar[dtype]
    var xmin: Scalar[dtype]
    var xmax: Scalar[dtype]
    var nan = math.nan[dtype]()
    var inf = math.limit.inf[dtype]()

    @parameter
    if dtype == DType.float32:
        xeps = bitcast[dtype, DType.uint32](0x3300_0000)
        xsml_inf = bitcast[dtype, DType.uint32](0xBE93_4B11)
        xsml_sup = bitcast[dtype, DType.uint32](0x3E64_7FBF)
        xmin = bitcast[dtype, DType.uint32](0xC18A_A122)
        xmax = math.nextafter(log(FloatLimits[dtype].max), 0.0)
    else:  # dtype == DType.float64
        xeps = bitcast[dtype, DType.uint64](0x3C900000_00000000)
        xsml_inf = bitcast[dtype, DType.uint64](0xBFD26962_1134DB93)
        xsml_sup = bitcast[dtype, DType.uint64](0x3FCC8FF7_C79A9A22)
        xmin = bitcast[dtype, DType.uint64](0xC042B708_872320E1)
        xmax = log(FloatLimits[dtype].max)

    var xs = StaticTuple[12, Scalar[dtype]](
        nan,
        -inf,
        xmin - 1.0,
        xmin,
        xsml_inf,
        -xeps,
        0.0,
        xeps,
        xsml_sup,
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
        var actual = expm1(x)
        var expected: Scalar[dtype]

        if math.isnan(x):
            expected = nan
        elif x < xmin:
            expected = -1.0
        elif x > xmax:
            expected = inf
        else:
            expected = _mp_expm1[dtype](x)

        unit_test.assert_all_close(actual, expected, 0.0, rtol)


fn main() raises:
    # Setting the mpmath precision for this module
    var mp = Python.import_module("mpmath")
    mp.mp.dps = 50

    test_expm1[DType.float64]()
    test_expm1[DType.float32]()

    test_expm1_special_cases[DType.float64]()
    test_expm1_special_cases[DType.float32]()
