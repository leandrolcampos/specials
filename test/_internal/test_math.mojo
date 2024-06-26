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

"""Tests for mathematical utilities."""

import math

from python import Python

from specials._internal.math import ldexp
from specials.utils.numerics import FloatLimits
from test_utils import UnitTest


fn _np_ldexp[
    type: DType
](x: Scalar[type], exp: Scalar[DType.int32]) raises -> Scalar[type]:
    var np = Python.import_module("numpy")
    var result = np.ldexp(x, exp)
    return result.to_float64().cast[type]()


fn test_ldexp_float_max[type: DType]() raises:
    var unit_test = UnitTest("test_ldexp_float_max_" + str(type))

    var x: Scalar[type] = 1.0 - FloatLimits[type].epsilon_neg()
    var exp: Scalar[DType.int32] = FloatLimits[type].max_exponent

    var expected = _np_ldexp[type](x, exp)
    var actual = ldexp[type](x, exp)

    unit_test.assert_equal(actual, expected)
    unit_test.assert_true(
        math.isfinite(actual).reduce_and(), msg="max should be finite"
    )


fn test_ldexp_float_smallest_normal[type: DType]() raises:
    var unit_test = UnitTest("test_ldexp_float_smallest_normal_" + str(type))

    var x: Scalar[type] = 1.0
    var exp: Scalar[DType.int32] = FloatLimits[type].min_exponent - 1

    var expected = _np_ldexp[type](x, exp)
    var actual = ldexp[type](x, exp)

    unit_test.assert_equal(actual, expected)
    unit_test.assert_true(
        (actual > 0.0).reduce_and(), msg="smallest_normal should be positive"
    )


fn test_ldexp_float_smallest_subnormal[type: DType]() raises:
    var unit_test = UnitTest("test_ldexp_float_smallest_subnormal_" + str(type))

    var x: Scalar[type] = 1.0
    var exp: Scalar[DType.int32] = FloatLimits[type].min_exponent - FloatLimits[
        type
    ].digits

    var expected = _np_ldexp[type](x, exp)
    var actual = ldexp[type](x, exp)

    unit_test.assert_equal(actual, expected)
    unit_test.assert_true(
        (actual > 0.0).reduce_and(), msg="smallest_subnormal should be positive"
    )


fn main() raises:
    test_ldexp_float_max[DType.float64]()
    test_ldexp_float_max[DType.float32]()

    test_ldexp_float_smallest_normal[DType.float64]()
    test_ldexp_float_smallest_normal[DType.float32]()

    test_ldexp_float_smallest_subnormal[DType.float64]()
    test_ldexp_float_smallest_subnormal[DType.float32]()
