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

"""Tests for the `exp` function."""

import math

from python import Python
from utils.static_tuple import StaticTuple

from specials.elementary.exp import exp
from specials.utils.numerics import FloatLimits
from test_utils import UnitTest


fn _mp_exp[type: DType](x: Scalar[type]) raises -> Scalar[type]:
    var mp = Python.import_module("mpmath")
    var result = mp.exp(mp.mpf(x))
    return result.to_float64().cast[type]()


fn test_exp[type: DType]() raises:
    var unit_test = UnitTest("test_exp_" + str(type))

    var xs = StaticTuple[Scalar[type], 9](
        -10.0, -1.0, -0.1, -0.01, 0.0, 0.01, 0.1, 1.0, 10.0
    )

    var rtol: Scalar[type]

    @parameter
    if type == DType.float32:
        rtol = 1e-6
    else:  # type == DType.float64
        rtol = 1e-15

    for i in range(len(xs)):
        var x = xs[i]
        var expected = _mp_exp[type](x)
        var actual = exp(x)

        unit_test.assert_all_close(actual, expected, atol=0.0, rtol=rtol)


fn test_exp_special_cases[type: DType]() raises:
    var unit_test = UnitTest("test_exp_special_cases_" + str(type))

    var log2 = 0.6931471805599453
    var xmin = Scalar[type](FloatLimits[type].min_exponent - 1) * log2
    var xeps = 0.5 * FloatLimits[type].epsilon_neg()
    var xmax = math.nextafter(
        Scalar[type](FloatLimits[type].max_exponent) * log2, 0.0
    )
    var nan = math.nan[type]()
    var inf = math.inf[type]()

    var xs = StaticTuple[Scalar[type], 13](
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

    var rtol: Scalar[type]

    @parameter
    if type == DType.float32:
        rtol = 1e-6
    else:  # type == DType.float64
        rtol = 1e-15

    for i in range(len(xs)):
        var x = xs[i]
        var actual = exp(x)
        var expected: Scalar[type]

        if math.isnan(x):
            expected = nan
        elif x > xmax:
            expected = inf
        else:
            expected = _mp_exp[type](x)

        unit_test.assert_all_close(actual, expected, atol=0.0, rtol=rtol)


fn main() raises:
    # Setting the mpmath precision for this module
    var mp = Python.import_module("mpmath")
    mp.mp.dps = 50

    test_exp[DType.float64]()
    test_exp[DType.float32]()

    test_exp_special_cases[DType.float64]()
    test_exp_special_cases[DType.float32]()
