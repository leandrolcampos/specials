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

"""Tests for table utilities."""

import math

from specials._internal.table import FloatTable
from specials._internal.testing import UnitTest


fn test_sized() raises:
    var unit_test = UnitTest("test_sized")

    var table = FloatTable[4, DType.float32].from_values[1.0, 2.0, 3.0, 4.0]()
    var expected = 4
    var actual = len(table)

    unit_test.assert_equal(actual, expected)


fn test_unsafe_lookup[dtype: DType]() raises:
    var unit_test = UnitTest("test_unsafe_lookup_" + str(dtype))

    var table = FloatTable[4, dtype].from_values[1.0, 2.0, 3.0, 4.0]()
    var index = SIMD[DType.int32, 4](3, 0, 2, 1)
    var expected = SIMD[dtype, 4](4.0, 1.0, 3.0, 2.0)
    var actual = table.unsafe_lookup(index)

    unit_test.assert_equal(actual, expected)


fn test_lookup[dtype: DType]() raises:
    var unit_test = UnitTest("test_lookup_" + str(dtype))

    var table = FloatTable[4, dtype].from_values[1.0, 2.0, 3.0, 4.0]()
    var index = SIMD[DType.int32, 4](3, 0, 2, 1)
    var expected = SIMD[dtype, 4](4.0, 1.0, 3.0, 2.0)
    var actual = table.lookup(index)

    unit_test.assert_equal(actual, expected)


fn test_out_of_bound[dtype: DType]() raises:
    var unit_test = UnitTest("test_lookup_out_of_bound_" + str(dtype))

    var table = FloatTable[4, dtype].from_values[1.0, 2.0, 3.0, 4.0]()
    var index = SIMD[DType.int32, 4](4, 0, 2, 1)
    var expected = SIMD[dtype, 4](math.nan[dtype](), 1.0, 3.0, 2.0)
    var actual = table.lookup(index)

    unit_test.assert_equal(actual, expected)


fn test_hexadecimal_values[dtype: DType]() raises:
    var unit_test = UnitTest("test_hexadecimal_values_" + str(dtype))

    var table: FloatTable[2, dtype]

    @parameter
    if dtype == DType.float32:
        table = FloatTable[2, dtype].from_hexadecimal_values[
            0x3F80_0000,
            0x3F00_0000,
        ]()
    else:
        table = FloatTable[2, dtype].from_hexadecimal_values[
            0x3FF00000_00000000,
            0x3FE00000_00000000,
        ]()

    var index = SIMD[DType.int32, 2](0, 1)
    var expected = SIMD[dtype, 2](1.0, 0.5)
    var actual = table.lookup(index)

    unit_test.assert_equal(actual, expected)


fn main() raises:
    test_sized()

    test_unsafe_lookup[DType.float32]()
    test_unsafe_lookup[DType.float64]()

    test_lookup[DType.float32]()
    test_lookup[DType.float64]()

    test_out_of_bound[DType.float32]()
    test_out_of_bound[DType.float64]()

    test_hexadecimal_values[DType.float32]()
    test_hexadecimal_values[DType.float64]()
