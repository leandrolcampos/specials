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
# RUN: %mojo %build_dir %debug_level %sanitize_checks %s

"""Tests for gamma-related functions."""

import math

from python import Python
from utils.static_tuple import StaticTuple

import specials

from specials._internal.limits import FloatLimits
from test_utils import UnitTest

# TODO: Add tests with `DType.float32` type for `lbeta`.
# TODO: Investigate why defining `MOJO_ENABLE_ASSERTIONS` leads to test failures.


fn test_lgamma_correction[type: DType]() raises:
    var unit_test = UnitTest("test_lgamma_correction_" + str(type))

    var x = SIMD[type, 4](10.0, 100.0, 1000.0, 10000.0)

    # The expected values were computed using `mpmath`.
    var expected = SIMD[type, 4](
        8.330563433362871256469318659629e-3,
        8.333305556349146833812416928147e-4,
        8.333333055555634920575396909572e-5,
        8.333333330555555563492063432540e-6,
    )
    var actual = specials.lgamma_correction(x)

    var rtol: Scalar[type]

    @parameter
    if type == DType.float32:
        rtol = 1e-6
    else:  # type == DType.float64
        rtol = 1e-12

    unit_test.assert_all_close(actual, expected, atol=0.0, rtol=rtol)


fn test_lgamma_correction_special_cases[type: DType]() raises:
    var unit_test = UnitTest(
        "test_lgamma_correction_special_cases_" + str(type)
    )

    var inf = math.limit.inf[type]()
    var nan = math.nan[type]()
    var zero = Scalar[type](0.0)
    var x = SIMD[type, 4](nan, -inf, zero, inf)

    var expected = SIMD[type, 4](nan, nan, nan, zero)
    var actual = specials.lgamma_correction(x)

    unit_test.assert_equal(actual, expected)


fn test_lgamma1p_region1[type: DType]() raises:
    var unit_test = UnitTest("test_lgamma1p_region1_" + str(type))

    alias eps_f32 = FloatLimits[DType.float32].eps.cast[type]()
    var x = SIMD[type, 4](-0.2, 0.0 - eps_f32, 0.0 + eps_f32, 0.60 - eps_f32)

    # The expected values were computed using `mpmath`.
    var expected = SIMD[type, 4](
        1.520596783998375994920398994673e-1,
        6.880948101845375157827140831409e-8,
        -6.88094576425347115116354810460e-8,
        -1.12591780722776981673572240680e-1,
    )

    var rtol: Scalar[type]

    @parameter
    if type == DType.float32:
        rtol = 1e-6
    else:  # type == DType.float64
        rtol = 1e-12

    unit_test.assert_all_close(
        specials.lgamma1p(x), expected, atol=0.0, rtol=rtol
    )


fn test_lgamma1p_region2[type: DType]() raises:
    var unit_test = UnitTest("test_lgamma1p_region2_" + str(type))

    alias eps_f32 = FloatLimits[DType.float32].eps.cast[type]()
    var x = SIMD[type, 4](0.6, 1.0 - eps_f32, 1.0 + eps_f32, 1.25)

    # The expected values were computed using `mpmath`.
    var expected = SIMD[type, 4](
        -1.12591765696755786387475561192e-1,
        -5.03998156377554207114124403668e-8,
        5.039982480281974557594367601653e-8,
        1.248717148923965943024412876132e-1,
    )

    var rtol: Scalar[type]

    @parameter
    if type == DType.float32:
        rtol = 1e-6
    else:  # type == DType.float64
        rtol = 1e-12

    unit_test.assert_all_close(
        specials.lgamma1p(x), expected, atol=0.0, rtol=rtol
    )


fn test_lgamma1p_edge_cases[type: DType]() raises:
    var unit_test = UnitTest("test_lgamma1p_edge_cases_" + str(type))

    var inf = math.limit.inf[type]()
    var nan = math.nan[type]()
    var x = SIMD[type, 4](nan, -1.0, 0.0, inf)

    var expected = math.lgamma(1.0 + x)
    var actual = specials.lgamma1p(x)

    unit_test.assert_equal(actual, expected)


fn test_lbeta() raises:
    var unit_test = UnitTest("test_lbeta")

    var x = SIMD[DType.float64, 4](1.0, 8.0, 1e-5, 1e4)
    var y = SIMD[DType.float64, 4](7.0, 8.0, 1e10, 1e5)

    # The expected values were computed using `mpmath`.
    var expected = SIMD[DType.float64, 4](
        -1.9459101490553133051,
        -10.848948661710062966,
        11.5126894343865267210,
        -33513.609276569981795,
    )

    unit_test.assert_all_close(
        specials.lbeta(x, y), expected, atol=0.0, rtol=1e-12
    )


fn test_lbeta_edge_cases() raises:
    var unit_test = UnitTest("test_lbeta_edge_cases")

    var neg = SIMD[DType.float64, 1](-1.0)
    var inf = math.limit.inf[DType.float64]()
    var nan = math.nan[DType.float64]()
    var x = SIMD[DType.float64, 8](neg, neg, neg, neg, 0.0, 0.0, 1.0, 1.0)
    var y = SIMD[DType.float64, 8](0.0, 2.0, nan, inf, 2.0, inf, nan, inf)

    # The expected values were computed using the R language.
    var expected = SIMD[DType.float64, 8](
        nan, nan, nan, nan, inf, inf, nan, -inf
    )
    var actual = specials.lbeta(x, y)

    unit_test.assert_equal(actual, expected)


fn _mp_rgamma1pm1[type: DType](x: Scalar[type]) raises -> Scalar[type]:
    var mp = Python.import_module("mpmath")
    mp.mp.dps += 300

    var result = mp.mpf(1) / mp.gamma(mp.mpf(1) + mp.mpf(x)) - mp.mpf(1)

    mp.mp.dps -= 300
    return result.to_float64().cast[type]()


fn test_rgamma1pm1[type: DType]() raises:
    var unit_test = UnitTest("test_rgamma1pm1_" + str(type))

    var tiny = FloatLimits[type].min
    var epsneg = FloatLimits[type].epsneg
    var eps = FloatLimits[type].eps

    var xs = StaticTuple[Scalar[type], 10](
        -1.0 + epsneg,
        -0.5,
        -tiny,
        0.0,
        tiny,
        0.5,
        1.0 - epsneg,
        1.0,
        1.5 - eps,
        7.0,
    )

    var rtol: Scalar[type]

    @parameter
    if type == DType.float32:
        rtol = 1e-6
    else:  # type == DType.float64
        rtol = 1e-12

    for i in range(len(xs)):
        var x = xs[i]
        var expected = _mp_rgamma1pm1[type](x)
        var actual = specials.rgamma1pm1(x)

        unit_test.assert_all_close(actual, expected, atol=0.0, rtol=rtol)


fn test_rgamma1pm1_special_cases[type: DType]() raises:
    var unit_test = UnitTest("test_rgamma1pm1_special_cases_" + str(type))

    var inf = math.limit.inf[type]()
    var nan = math.nan[type]()
    var x = SIMD[type, 4](nan, -inf, inf, nan)

    var expected = SIMD[type, 4](nan, nan, -1, nan)
    var actual = specials.rgamma1pm1(x)

    unit_test.assert_equal(actual, expected)


fn main() raises:
    # Setting the mpmath precision for this module
    var mp = Python.import_module("mpmath")
    mp.mp.dps = 50

    # lgamma_correction
    test_lgamma_correction[DType.float64]()
    test_lgamma_correction[DType.float32]()

    test_lgamma_correction_special_cases[DType.float64]()
    test_lgamma_correction_special_cases[DType.float32]()

    # lgamma1p
    test_lgamma1p_region1[DType.float32]()
    test_lgamma1p_region1[DType.float64]()

    test_lgamma1p_region2[DType.float32]()
    test_lgamma1p_region2[DType.float64]()

    test_lgamma1p_edge_cases[DType.float32]()
    test_lgamma1p_edge_cases[DType.float64]()

    # lbeta
    test_lbeta()
    test_lbeta_edge_cases()

    # rgamma1pm1
    test_rgamma1pm1[DType.float64]()
    test_rgamma1pm1[DType.float32]()

    test_rgamma1pm1_special_cases[DType.float64]()
    test_rgamma1pm1_special_cases[DType.float32]()
