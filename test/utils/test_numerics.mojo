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

"""Tests for utilities that work with numeric types."""

import math

from specials.utils.numerics import FloatLimits
from test_utils import UnitTest


fn test_float_limits_digits() raises:
    var unit_test = UnitTest("test_float_limits_digits")

    unit_test.assert_equal(FloatLimits[DType.float16].digits, 11)
    unit_test.assert_equal(FloatLimits[DType.float32].digits, 24)
    unit_test.assert_equal(FloatLimits[DType.float64].digits, 53)


fn test_float_limits_max_exponent() raises:
    var unit_test = UnitTest("test_float_limits_max_exponent")

    unit_test.assert_equal(FloatLimits[DType.float16].max_exponent, 16)
    unit_test.assert_equal(FloatLimits[DType.float32].max_exponent, 128)
    unit_test.assert_equal(FloatLimits[DType.float64].max_exponent, 1024)


fn test_float_limits_min_exponent() raises:
    var unit_test = UnitTest("test_float_limits_min_exponent")

    unit_test.assert_equal(FloatLimits[DType.float16].min_exponent, -13)
    unit_test.assert_equal(FloatLimits[DType.float32].min_exponent, -125)
    unit_test.assert_equal(FloatLimits[DType.float64].min_exponent, -1021)


fn test_float_limits_radix() raises:
    var unit_test = UnitTest("test_float_limits_radix")

    unit_test.assert_equal(FloatLimits[DType.float16].radix, 2)
    unit_test.assert_equal(FloatLimits[DType.float32].radix, 2)
    unit_test.assert_equal(FloatLimits[DType.float64].radix, 2)


fn test_float_limits_denorm_min() raises:
    var unit_test = UnitTest("test_float_limits_denorm_min")

    unit_test.assert_equal(
        FloatLimits[DType.float16].denorm_min(), Float16(5.960464477539063e-08)
    )
    unit_test.assert_equal(
        FloatLimits[DType.float32].denorm_min(), Float32(1.401298464324817e-45)
    )
    unit_test.assert_equal(
        FloatLimits[DType.float64].denorm_min(),
        math.nextafter(Float64(0.0), 1.0),
    )


fn test_float_limits_epsilon() raises:
    var unit_test = UnitTest("test_float_limits_epsilon")

    unit_test.assert_equal(
        FloatLimits[DType.float16].epsilon(), Float16(0.0009765625)
    )
    unit_test.assert_equal(
        FloatLimits[DType.float32].epsilon(), Float32(1.1920928955078125e-07)
    )
    unit_test.assert_equal(
        FloatLimits[DType.float64].epsilon(), Float64(2.220446049250313e-16)
    )


fn test_float_limits_epsilon_neg() raises:
    var unit_test = UnitTest("test_float_limits_epsilon_neg")

    unit_test.assert_equal(
        FloatLimits[DType.float16].epsilon_neg(), Float16(0.00048828125)
    )
    unit_test.assert_equal(
        FloatLimits[DType.float32].epsilon_neg(), Float32(5.960464477539063e-08)
    )
    unit_test.assert_equal(
        FloatLimits[DType.float64].epsilon_neg(),
        Float64(1.1102230246251565e-16),
    )


fn test_float_limits_lowest() raises:
    var unit_test = UnitTest("test_float_limits_lowest")

    unit_test.assert_equal(
        FloatLimits[DType.float16].lowest(), Float16(-65504.0)
    )
    unit_test.assert_equal(
        FloatLimits[DType.float32].lowest(), Float32(-3.4028234663852886e38)
    )
    unit_test.assert_equal(
        FloatLimits[DType.float64].lowest(), Float64(-1.7976931348623157e308)
    )


fn test_float_limits_max() raises:
    var unit_test = UnitTest("test_float_limits_max")

    unit_test.assert_equal(FloatLimits[DType.float16].max(), Float16(65504.0))
    unit_test.assert_equal(
        FloatLimits[DType.float32].max(), Float32(3.4028234663852886e38)
    )
    unit_test.assert_equal(
        FloatLimits[DType.float64].max(), Float64(1.7976931348623157e308)
    )


fn test_float_limits_min() raises:
    var unit_test = UnitTest("test_float_limits_min")

    unit_test.assert_equal(
        FloatLimits[DType.float16].min(), Float16(6.103515625e-05)
    )
    unit_test.assert_equal(
        FloatLimits[DType.float32].min(), Float32(1.1754943508222875e-38)
    )
    unit_test.assert_equal(
        FloatLimits[DType.float64].min(), Float64(2.2250738585072014e-308)
    )


fn main() raises:
    test_float_limits_digits()
    test_float_limits_max_exponent()
    test_float_limits_min_exponent()
    test_float_limits_radix()
    test_float_limits_denorm_min()
    test_float_limits_epsilon()
    test_float_limits_epsilon_neg()
    test_float_limits_lowest()
    test_float_limits_max()
    test_float_limits_min()
