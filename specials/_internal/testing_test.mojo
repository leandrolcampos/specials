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

"""Tests for testing utilities."""

import math

from specials._internal.testing import UnitTest


fn test_assert_true_success() raises:
    let unit_test = UnitTest("test_assert_true_success")
    unit_test.assert_true(True, "this test should not have failed")


fn test_assert_true_failure() raises:
    let unit_test = UnitTest[True]("test_assert_true_failure")

    try:
        unit_test.assert_true(False, "test")
    except error:
        return

    print("AssertionError: this test should have failed")


fn test_assert_equal_int_success() raises:
    let unit_test = UnitTest("test_assert_equal_int_success")
    unit_test.assert_equal(0, 0)


fn test_assert_equal_int_failure() raises:
    let unit_test = UnitTest[True]("test_assert_equal_int_failure")

    try:
        unit_test.assert_equal(0, 1)
    except error:
        return

    print("AssertionError: this test should have failed")


fn test_assert_equal_success() raises:
    let unit_test = UnitTest("test_assert_equal_success")

    let actual = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    let desired = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)

    unit_test.assert_equal(actual, desired)


fn test_assert_equal_failure() raises:
    let unit_test = UnitTest[True]("test_assert_equal_failure")

    let actual = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    let desired = SIMD[DType.float32, 4](0.0, 2.0, 3.0, 4.0)

    try:
        unit_test.assert_equal(actual, desired)
    except error:
        return

    print("AssertionError: this test should have failed")


fn test_assert_all_close_success() raises:
    let unit_test = UnitTest("test_assert_all_close_success")

    let nan = math.nan[DType.float32]()
    let inf = math.limit.inf[DType.float32]()

    let actual = SIMD[DType.float32, 4](nan, inf, 1.1, 10.2)
    let desired = SIMD[DType.float32, 4](nan, inf, 1.0, 10.0)

    unit_test.assert_all_close(actual, desired, 0.1, 1.0e-2)


fn test_assert_all_close_failure_nan() raises:
    let unit_test = UnitTest[True]("test_assert_all_close_failure_nan")

    let nan = math.nan[DType.float32]()
    let inf = math.limit.inf[DType.float32]()

    let actual = SIMD[DType.float32, 4](nan, inf, 1.1, 10.2)
    let desired = SIMD[DType.float32, 4](0.0, inf, 1.0, 10.0)

    try:
        unit_test.assert_all_close(actual, desired, 0.1, 1.0e-2)
    except error:
        return

    print("AssertionError: this test should have failed")


fn test_assert_all_close_failure_inf() raises:
    let unit_test = UnitTest[True]("test_assert_all_close_failure_inf")

    let nan = math.nan[DType.float32]()
    let inf = math.limit.inf[DType.float32]()

    let actual = SIMD[DType.float32, 4](nan, 0.0, 1.1, 10.2)
    let desired = SIMD[DType.float32, 4](nan, inf, 1.0, 10.0)

    try:
        unit_test.assert_all_close(actual, desired, 0.1, 1.0e-2)
    except error:
        return

    print("AssertionError: this test should have failed")


fn test_assert_all_close_failure_atol() raises:
    let unit_test = UnitTest[True]("test_assert_all_close_failure_atol")

    let nan = math.nan[DType.float32]()
    let inf = math.limit.inf[DType.float32]()

    let actual = SIMD[DType.float32, 4](nan, inf, 1.11, 10.2)
    let desired = SIMD[DType.float32, 4](nan, inf, 1.0, 10.0)

    try:
        unit_test.assert_all_close(actual, desired, 0.1, 1.0e-2)
    except error:
        return

    print("AssertionError: this test should have failed")


fn test_assert_all_close_failure_rtol() raises:
    let unit_test = UnitTest[True]("test_assert_all_close_failure_rtol")

    let nan = math.nan[DType.float32]()
    let inf = math.limit.inf[DType.float32]()

    let actual = SIMD[DType.float32, 4](nan, inf, 1.1, 10.21)
    let desired = SIMD[DType.float32, 4](nan, inf, 1.0, 10.0)

    try:
        unit_test.assert_all_close(actual, desired, 0.1, 1.0e-2)
    except error:
        return

    print("AssertionError: this test should have failed")


fn main() raises:
    test_assert_true_success()
    test_assert_true_failure()

    test_assert_equal_int_success()
    test_assert_equal_int_failure()

    test_assert_equal_success()
    test_assert_equal_failure()

    test_assert_all_close_success()
    test_assert_all_close_failure_nan()
    test_assert_all_close_failure_inf()
    test_assert_all_close_failure_atol()
    test_assert_all_close_failure_rtol()
