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

"""Tests for higher-order functions."""

from collections.list import List

from specials._internal.functional import fori_loop
from test_utils import UnitTest


fn test_fori_loop_increasing_by_1() raises:
    var unit_test = UnitTest("test_fori_loop_increasing_by_1")
    var actual = List[Int](capacity=5)
    var expected = List[Int](capacity=5)

    @parameter
    fn body_func[i: Int]():
        actual.append(i)

    fori_loop[0, 5, 1, body_func]()

    for i in range(5):
        expected.append(i)

    unit_test.assert_equal(len(actual), 5)
    unit_test.assert_equal(len(expected), 5)

    for i in range(5):
        unit_test.assert_equal(actual[i], expected[i])


fn test_fori_loop_decreasing_by_2() raises:
    var unit_test = UnitTest("test_fori_loop_decreasing_by_2")
    var actual = List[Int](capacity=5)
    var expected = List[Int](capacity=5)

    @parameter
    fn body_func[i: Int]():
        actual.append(i)

    fori_loop[5, -1, -2, body_func]()

    for i in range(5, -1, -2):
        expected.append(i)

    unit_test.assert_equal(len(actual), 3)
    unit_test.assert_equal(len(expected), 3)

    for i in range(len(actual)):
        unit_test.assert_equal(actual[i], expected[i])


fn test_fori_loop_with_conditional() raises:
    var unit_test = UnitTest("test_fori_loop_with_conditional")
    var actual = List[Int](capacity=5)
    var expected = List[Int](capacity=5)

    @parameter
    fn body_func[i: Int]() -> Bool:
        actual.append(i)
        if i <= 3:
            return False
        return True

    fori_loop[5, -1, -1, body_func]()

    for i in range(5, -1, -1):
        expected.append(i)
        if i <= 3:
            break

    unit_test.assert_equal(len(actual), 3)
    unit_test.assert_equal(len(expected), 3)

    for i in range(len(actual)):
        unit_test.assert_equal(actual[i], expected[i])


fn main() raises:
    test_fori_loop_increasing_by_1()
    test_fori_loop_decreasing_by_2()
    test_fori_loop_with_conditional()
