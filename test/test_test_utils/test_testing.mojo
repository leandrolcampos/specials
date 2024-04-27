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

"""Tests for testing utilities."""

import math

from testing import assert_raises

from test_utils import UnitTest


fn test_assert_true_success() raises:
    var unit_test = UnitTest("test_assert_true_success")
    unit_test.assert_true(True, msg="this test should not have failed")


fn test_assert_true_failure() raises:
    var unit_test = UnitTest("test_assert_true_failure")
    with assert_raises(contains="this test should have failed"):
        unit_test.assert_true(False, msg="this test should have failed")


fn test_assert_equal_int_success() raises:
    var unit_test = UnitTest("test_assert_equal_int_success")
    unit_test.assert_equal(0, 0, msg="this test should not have failed")


fn test_assert_equal_int_failure() raises:
    var unit_test = UnitTest("test_assert_equal_int_failure")
    with assert_raises(contains="this test should have failed"):
        unit_test.assert_equal(0, 1, msg="this test should have failed")


fn test_assert_equal_simd_success() raises:
    var unit_test = UnitTest("test_assert_equal_simd_success")

    var nan = math.nan[DType.float32]()
    var inf = math.limit.inf[DType.float32]()
    var actual = SIMD[DType.float32, 4](1.0, 2.0, inf, nan)
    var desired = SIMD[DType.float32, 4](1.0, 2.0, inf, nan)

    unit_test.assert_equal(
        actual, desired, msg="this test should not have failed"
    )


fn test_assert_equal_simd_failure() raises:
    var unit_test = UnitTest("test_assert_equal_simd_failure")

    var nan = math.nan[DType.float32]()
    var inf = math.limit.inf[DType.float32]()
    var actual = SIMD[DType.float32, 4](0.0, 2.0, inf, nan)
    var desired = SIMD[DType.float32, 4](0.0, 2.0, inf, 4.0)

    with assert_raises(contains="this test should have failed"):
        unit_test.assert_equal(
            actual, desired, msg="this test should have failed"
        )


fn test_assert_all_close_success() raises:
    var unit_test = UnitTest("test_assert_all_close_success")

    var nan = math.nan[DType.float32]()
    var inf = math.limit.inf[DType.float32]()
    var actual = SIMD[DType.float32, 4](nan, inf, 1.1, 10.2)
    var desired = SIMD[DType.float32, 4](nan, inf, 1.0, 10.0)

    unit_test.assert_all_close(
        actual,
        desired,
        atol=0.1,
        rtol=1.0e-2,
        msg="this test should not have failed",
    )


fn test_assert_all_close_failure_nan() raises:
    var unit_test = UnitTest("test_assert_all_close_failure_nan")

    var nan = math.nan[DType.float32]()
    var inf = math.limit.inf[DType.float32]()
    var actual = SIMD[DType.float32, 4](nan, inf, 1.1, 10.2)
    var desired = SIMD[DType.float32, 4](0.0, inf, 1.0, 10.0)

    with assert_raises(contains="this test should have failed"):
        unit_test.assert_all_close(
            actual,
            desired,
            atol=0.1,
            rtol=1.0e-2,
            msg="this test should have failed",
        )


fn test_assert_all_close_failure_inf() raises:
    var unit_test = UnitTest("test_assert_all_close_failure_inf")

    var nan = math.nan[DType.float32]()
    var inf = math.limit.inf[DType.float32]()
    var actual = SIMD[DType.float32, 4](nan, 0.0, 1.1, 10.2)
    var desired = SIMD[DType.float32, 4](nan, inf, 1.0, 10.0)

    with assert_raises(contains="this test should have failed"):
        unit_test.assert_all_close(
            actual,
            desired,
            atol=0.1,
            rtol=1.0e-2,
            msg="this test should have failed",
        )


fn test_assert_all_close_failure_atol() raises:
    var unit_test = UnitTest("test_assert_all_close_failure_atol")

    var nan = math.nan[DType.float32]()
    var inf = math.limit.inf[DType.float32]()
    var actual = SIMD[DType.float32, 4](nan, inf, 1.11, 10.2)
    var desired = SIMD[DType.float32, 4](nan, inf, 1.0, 10.0)

    with assert_raises(contains="this test should have failed"):
        unit_test.assert_all_close(
            actual,
            desired,
            atol=0.1,
            rtol=1.0e-2,
            msg="this test should have failed",
        )


fn test_assert_all_close_failure_rtol() raises:
    var unit_test = UnitTest("test_assert_all_close_failure_rtol")

    var nan = math.nan[DType.float32]()
    var inf = math.limit.inf[DType.float32]()
    var actual = SIMD[DType.float32, 4](nan, inf, 1.1, 10.21)
    var desired = SIMD[DType.float32, 4](nan, inf, 1.0, 10.0)

    with assert_raises(contains="this test should have failed"):
        unit_test.assert_all_close(
            actual,
            desired,
            atol=0.1,
            rtol=1.0e-2,
            msg="this test should have failed",
        )


fn main() raises:
    test_assert_true_success()
    test_assert_true_failure()

    test_assert_equal_int_success()
    test_assert_equal_int_failure()

    test_assert_equal_simd_success()
    test_assert_equal_simd_failure()

    test_assert_all_close_success()
    test_assert_all_close_failure_nan()
    test_assert_all_close_failure_inf()
    test_assert_all_close_failure_atol()
    test_assert_all_close_failure_rtol()
