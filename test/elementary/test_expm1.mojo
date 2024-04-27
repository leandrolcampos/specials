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
# RUN: %mojo %build_dir %assertion_flag %debug_level %sanitize_checks %s

"""Tests for the `expm1` function."""

import math

from memory.unsafe import bitcast

from python import Python
from utils.static_tuple import StaticTuple

from specials._internal.limits import FloatLimits
from specials.elementary.expm1 import expm1
from specials.elementary.log import log
from test_utils import UnitTest


fn _mp_expm1[type: DType](x: Scalar[type]) raises -> Scalar[type]:
    var mp = Python.import_module("mpmath")
    var result = mp.expm1(mp.mpf(x))
    return result.to_float64().cast[type]()


fn test_expm1[type: DType]() raises:
    var unit_test = UnitTest("test_expm1_" + str(type))

    var xeps = FloatLimits[type].eps
    var xs = StaticTuple[Scalar[type], 5](0.1 * xeps, 0.01, 0.1, 1.0, 10.0)

    var rtol: Scalar[type]

    @parameter
    if type == DType.float32:
        rtol = 1e-6
    else:  # type == DType.float64
        rtol = 1e-15

    for i in range(len(xs)):
        var x = xs[i]
        var expected = _mp_expm1[type](x)
        var actual = expm1(x)

        unit_test.assert_all_close(actual, expected, atol=0.0, rtol=rtol)


fn test_expm1_special_cases[type: DType]() raises:
    var unit_test = UnitTest("test_expm1_special_cases_" + str(type))

    var xeps: Scalar[type]
    var xsml_inf: Scalar[type]
    var xsml_sup: Scalar[type]
    var xmin: Scalar[type]
    var xmax: Scalar[type]
    var nan = math.nan[type]()
    var inf = math.limit.inf[type]()

    @parameter
    if type == DType.float32:
        xeps = bitcast[type, DType.uint32](0x3300_0000)
        xsml_inf = bitcast[type, DType.uint32](0xBE93_4B11)
        xsml_sup = bitcast[type, DType.uint32](0x3E64_7FBF)
        xmin = bitcast[type, DType.uint32](0xC18A_A122)
        xmax = math.nextafter(log(FloatLimits[type].max), 0.0)
    else:  # type == DType.float64
        xeps = bitcast[type, DType.uint64](0x3C900000_00000000)
        xsml_inf = bitcast[type, DType.uint64](0xBFD26962_1134DB93)
        xsml_sup = bitcast[type, DType.uint64](0x3FCC8FF7_C79A9A22)
        xmin = bitcast[type, DType.uint64](0xC042B708_872320E1)
        xmax = log(FloatLimits[type].max)

    var xs = StaticTuple[Scalar[type], 12](
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

    var rtol: Scalar[type]

    @parameter
    if type == DType.float32:
        rtol = 1e-6
    else:  # type == DType.float64
        rtol = 1e-15

    for i in range(len(xs)):
        var x = xs[i]
        var actual = expm1(x)
        var expected: Scalar[type]

        if math.isnan(x):
            expected = nan
        elif x < xmin:
            expected = -1.0
        elif x > xmax:
            expected = inf
        else:
            expected = _mp_expm1[type](x)

        unit_test.assert_all_close(actual, expected, atol=0.0, rtol=rtol)


fn main() raises:
    # Setting the mpmath precision for this module
    var mp = Python.import_module("mpmath")
    mp.mp.dps = 50

    test_expm1[DType.float64]()
    test_expm1[DType.float32]()

    test_expm1_special_cases[DType.float64]()
    test_expm1_special_cases[DType.float32]()
