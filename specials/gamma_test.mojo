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

"""Tests for gamma-related functions."""

import math
import testing

import specials

from specials._internal.testing import UnitTest

# TODO: Add tests with `DType.float32` data type.


fn test_lgamma_correction() raises:
    let unit_test = UnitTest("test_lgamma_correction")

    let x = SIMD[DType.float64, 4](10.0, 100.0, 1000.0, 10000.0)

    # The expected values were computed using `mpmath`.
    let expected = SIMD[DType.float64, 4](
        8.330563433362871256469318659629e-3,
        8.333305556349146833812416928147e-4,
        8.333333055555634920575396909572e-5,
        8.333333330555555563492063432540e-6,
    )

    unit_test.assert_almost_equal[DType.float64, 4](
        expected, specials.lgamma_correction(x), 0.0, 1e-12
    )


fn test_lgamma_correction_edge_cases() raises:
    let unit_test = UnitTest("test_lgamma_correction_edge_cases")

    let inf = math.limit.inf[DType.float64]()
    let nan = math.nan[DType.float64]()
    let zero = SIMD[DType.float64, 1](0.0)
    let x = SIMD[DType.float64, 4](nan, -inf, zero, inf)

    let expected = SIMD[DType.float64, 4](nan, nan, nan, zero)
    let actual = specials.lgamma_correction(x)

    # Here NaNs are compared like numbers and no assertion is raised if both objects
    # have NaNs in the same positions.
    let result = (
        (math.isnan(actual) & math.isnan(actual)) | (actual == expected)
    ).reduce_and()
    unit_test.assert_true(result, "Some of the results are incorrect.")


fn test_lbeta() raises:
    let unit_test = UnitTest("test_lbeta")

    let x = SIMD[DType.float64, 4](1.0, 8.0, 1e-5, 1e4)
    let y = SIMD[DType.float64, 4](7.0, 8.0, 1e10, 1e5)

    # The expected values were computed using `mpmath`.
    let expected = SIMD[DType.float64, 4](
        -1.9459101490553133051,
        -10.848948661710062966,
        11.5126894343865267210,
        -33513.609276569981795,
    )

    unit_test.assert_almost_equal[DType.float64, 4](
        expected, specials.lbeta(x, y), 0.0, 1e-12
    )
    unit_test.assert_almost_equal[DType.float64, 4](
        expected, specials.lbeta(y, x), 0.0, 1e-12
    )


fn test_lbeta_edge_cases() raises:
    let unit_test = UnitTest("test_lbeta_edge_cases")

    let neg = SIMD[DType.float64, 1](-1.0)
    let inf = math.limit.inf[DType.float64]()
    let nan = math.nan[DType.float64]()
    let x = SIMD[DType.float64, 8](neg, neg, neg, neg, 0.0, 0.0, 1.0, 1.0)
    let y = SIMD[DType.float64, 8](0.0, 2.0, nan, inf, 2.0, inf, nan, inf)

    # The expected values were computed using the R language.
    let expected = SIMD[DType.float64, 8](nan, nan, nan, nan, inf, inf, nan, -inf)
    let actual = specials.lbeta(x, y)

    # Here NaNs are compared like numbers and no assertion is raised if both objects
    # have NaNs in the same positions.
    let result = (
        (math.isnan(actual) & math.isnan(actual)) | (actual == expected)
    ).reduce_and()
    unit_test.assert_true(result, "Some of the results are incorrect.")


fn main() raises:
    # log-gamma correction
    test_lgamma_correction()
    test_lgamma_correction_edge_cases()

    # log-beta function
    test_lbeta()
    test_lbeta_edge_cases()
