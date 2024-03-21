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

"""Tests for mathematical utilities."""

import math

from python import Python

from specials._internal.limits import FloatLimits
from specials._internal.math import ldexp
from specials._internal.testing import UnitTest


fn _np_ldexp[
    dtype: DType
](x: Scalar[dtype], exp: Scalar[DType.int32]) raises -> Scalar[dtype]:
    var np = Python.import_module("numpy")
    var result = np.ldexp(x, exp)
    return result.to_float64().cast[dtype]()


fn test_ldexp_float_max[dtype: DType]() raises:
    var unit_test = UnitTest("test_ldexp_float_max_" + str(dtype))

    var x: Scalar[dtype] = 1.0 - FloatLimits[dtype].epsneg
    var exp: Scalar[DType.int32] = FloatLimits[dtype].maxexp

    var expected = _np_ldexp[dtype](x, exp)
    var actual = ldexp[dtype](x, exp)

    unit_test.assert_equal(actual, expected)
    unit_test.assert_true(
        math.limit.isfinite(actual).reduce_and(), "max should be finite"
    )


fn test_ldexp_float_smallest_normal[dtype: DType]() raises:
    var unit_test = UnitTest("test_ldexp_float_smallest_normal_" + str(dtype))

    var x: Scalar[dtype] = 1.0
    var exp: Scalar[DType.int32] = FloatLimits[dtype].minexp

    var expected = _np_ldexp[dtype](x, exp)
    var actual = ldexp[dtype](x, exp)

    unit_test.assert_equal(actual, expected)
    unit_test.assert_true(
        (actual > 0.0).reduce_and(), "smallest_normal should be positive"
    )


fn test_ldexp_float_smallest_subnormal[dtype: DType]() raises:
    var unit_test = UnitTest("test_ldexp_float_smallest_subnormal_" + str(dtype))

    var x: Scalar[dtype] = 1.0
    var exp: Scalar[DType.int32] = FloatLimits[dtype].minexp + FloatLimits[dtype].machep

    var expected = _np_ldexp[dtype](x, exp)
    var actual = ldexp[dtype](x, exp)

    unit_test.assert_equal(actual, expected)
    unit_test.assert_true(
        (actual > 0.0).reduce_and(), "smallest_subnormal should be positive"
    )


fn main() raises:
    test_ldexp_float_max[DType.float64]()
    test_ldexp_float_max[DType.float32]()

    test_ldexp_float_smallest_normal[DType.float64]()
    test_ldexp_float_smallest_normal[DType.float32]()

    test_ldexp_float_smallest_subnormal[DType.float64]()
    test_ldexp_float_smallest_subnormal[DType.float32]()
