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

"""Tests for common constants."""

from python import Python

from specials._internal.testing import UnitTest
from specials.elementary.common_constants import ExpTable, LogConstants


fn _np_exp2[dtype: DType](x: Scalar[dtype]) raises -> Scalar[dtype]:
    var np = Python.import_module("numpy")
    var result = np.exp2(x)
    return result.to_float64().cast[dtype]()


fn test_exp_table[dtype: DType]() raises:
    var unit_test = UnitTest("test_exp_table_" + str(dtype))

    var expected_table_size = 32

    unit_test.assert_equal(len(ExpTable[dtype].lead), expected_table_size)
    unit_test.assert_equal(len(ExpTable[dtype].trail), expected_table_size)

    var rtol: Scalar[dtype]

    @parameter
    if dtype == DType.float32:
        rtol = 1e-6
    else:  # dtype == DType.float64
        rtol = 1e-15

    for i in range(expected_table_size):
        var index = Scalar[DType.int32](i)
        var x_lead = ExpTable[dtype].lead.lookup(index)
        var x_trail = ExpTable[dtype].trail.lookup(index)
        var expected = _np_exp2[dtype](Scalar[dtype](i) / expected_table_size)
        var actual = x_lead + x_trail

        unit_test.assert_all_close(actual, expected, 0.0, rtol)


fn test_log_constants[dtype: DType]() raises:
    var unit_test = UnitTest("test_log_constants_" + str(dtype))

    var expected_table_size = 129

    unit_test.assert_equal(len(LogConstants[dtype].inv_fraction1), expected_table_size)
    unit_test.assert_equal(
        len(LogConstants[dtype].log_fraction1_lead), expected_table_size
    )
    unit_test.assert_equal(
        len(LogConstants[dtype].log_fraction1_trail), expected_table_size
    )

    fn _np_inv_fraction1[dtype: DType](j: Int) raises -> Scalar[dtype]:
        var np = Python.import_module("numpy")
        var result = np.reciprocal(1.0 + np.ldexp(j, -7))
        return result.to_float64().cast[dtype]()

    fn _np_log_fraction1[dtype: DType](j: Int) raises -> Scalar[dtype]:
        var np = Python.import_module("numpy")
        var result = np.log1p(np.ldexp(j, -7))
        return result.to_float64().cast[dtype]()

    var rtol: Scalar[dtype]

    @parameter
    if dtype == DType.float32:
        rtol = 1e-6
    else:  # dtype == DType.float64
        rtol = 1e-15

    for j in range(expected_table_size):
        var index = Scalar[DType.int32](j)

        # inv_fraction1
        var inv_fraction1 = LogConstants[dtype].inv_fraction1.lookup(index)
        var expected = _np_inv_fraction1[dtype](j)
        var actual = inv_fraction1

        unit_test.assert_all_close(actual, expected, 0.0, rtol)

        # log_fraction1
        var log_fraction1_lead = LogConstants[dtype].log_fraction1_lead.lookup(index)
        var log_fraction1_trail = LogConstants[dtype].log_fraction1_trail.lookup(index)
        expected = _np_log_fraction1[dtype](j)
        actual = log_fraction1_lead + log_fraction1_trail

        unit_test.assert_all_close(actual, expected, 0.0, rtol)


fn main() raises:
    test_exp_table[DType.float64]()
    test_exp_table[DType.float32]()

    test_log_constants[DType.float64]()
    test_log_constants[DType.float32]()
