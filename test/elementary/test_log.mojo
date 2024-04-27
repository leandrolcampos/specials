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

"""Tests for the logarithm function."""

import math

from python import Python
from utils.static_tuple import StaticTuple

from specials._internal.limits import FloatLimits
from specials.elementary.log import log
from test_utils import UnitTest


fn _mp_log[type: DType](x: Scalar[type]) raises -> Scalar[type]:
    var mp = Python.import_module("mpmath")
    var result = mp.log(mp.mpf(x))
    return result.to_float64().cast[type]()


fn test_log[type: DType]() raises:
    var unit_test = UnitTest("test_log_" + str(type))

    var xs = StaticTuple[Scalar[type], 8](
        1e-16,
        1e-4,
        0.1,
        0.99,
        5.0,
        10.0,
        1e4,
        1e16,
    )

    var rtol: Scalar[type]

    @parameter
    if type == DType.float32:
        rtol = 1e-6
    else:  # type == DType.float64
        rtol = 1e-15

    for i in range(len(xs)):
        var x = xs[i]
        var expected = _mp_log[type](x)
        var actual = log(x)

        unit_test.assert_all_close(actual, expected, atol=0.0, rtol=rtol)


fn test_log_special_cases[type: DType]() raises:
    var unit_test = UnitTest("test_log_special_cases_" + str(type))

    var xmin = FloatLimits[type].min
    var xmax = FloatLimits[type].max
    var nan = math.nan[type]()
    var inf = math.limit.inf[type]()

    var xs = StaticTuple[Scalar[type], 13](
        nan,
        -inf,
        -1.0,
        0.0,
        0.5 * xmin,
        xmin,
        2.0 * xmin,
        1.0 - FloatLimits[type].epsneg,
        1.0,
        1.0 + FloatLimits[type].eps,
        xmax - 1.0,
        xmax,
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
        var actual = log(x)
        var expected: Scalar[type]

        if math.isnan(x) | (x < 0.0):
            expected = nan
        elif x == 0.0:
            expected = -inf
        elif x > xmax:
            expected = inf
        else:
            expected = _mp_log[type](x)

        unit_test.assert_all_close(actual, expected, atol=0.0, rtol=rtol)


fn main() raises:
    # Setting the mpmath precision for this module
    var mp = Python.import_module("mpmath")
    mp.mp.dps = 100

    test_log[DType.float64]()
    test_log[DType.float32]()

    test_log_special_cases[DType.float64]()
    test_log_special_cases[DType.float32]()
