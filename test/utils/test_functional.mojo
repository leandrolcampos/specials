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

"""Tests for higher-order functions."""

from collections import List, Optional
from sys.info import simdwidthof
from tensor import Tensor

from specials.utils.functional import elementwise, fori_loop
from test_utils import UnitTest


fn _tensor_mul[
    type: DType
](
    a: Tensor[type],
    b: Tensor[type],
    /,
    *,
    min_simds_per_core: Optional[Int] = None,
) raises -> Tensor[type]:
    alias simd_width = simdwidthof[type]()

    if a.shape() != b.shape():
        raise Error("arguments `a` and `b` must have the same shape")

    if min_simds_per_core.or_else(1) < 1:
        raise Error(
            "argument `min_simds_per_core` must be `None` or a positive"
            " integer value"
        )

    var ret = Tensor[type](a.shape())

    @always_inline
    @parameter
    fn func[simd_width: Int](index: Int):
        var x = a.load[width=simd_width](index)
        var y = b.load[width=simd_width](index)
        ret.store[width=simd_width](index, x * y)

    var num_elements = a.num_elements()

    if min_simds_per_core:
        elementwise[func, simd_width=simd_width](
            num_elements, min_simds_per_core=min_simds_per_core.value()[]
        )
    else:
        elementwise[func, simd_width=simd_width](num_elements)

    return ret^


fn test_elementwise() raises:
    @parameter
    fn test_fn[type: DType]() raises:
        var unit_test = UnitTest("test_elementwise")

        var a = 1.0 + Tensor[type].rand(4096)
        var b = 1.0 + Tensor[type].rand(a.shape())

        var actual = _tensor_mul(a, b)
        var expected = a * b

        unit_test.assert_true(actual == expected, msg=str(type))

    test_fn[DType.float32]()
    test_fn[DType.float64]()


fn test_elementwise_parallel() raises:
    @parameter
    fn test_fn[type: DType](num_elements: Int) raises:
        var unit_test = UnitTest("test_elementwise_parallel")

        var a = 1.0 + Tensor[type].rand(num_elements)
        var b = 1.0 + Tensor[type].rand(a.shape())

        var actual = _tensor_mul(a, b, min_simds_per_core=1)
        var expected = a * b

        unit_test.assert_true(
            actual == expected, msg=str(type) + "_" + str(num_elements)
        )

    for num_elements in List(1, 5, 25, 125, 625):
        test_fn[DType.float32](num_elements[])
        test_fn[DType.float64](num_elements[])


fn test_elementwise_sequential() raises:
    @parameter
    fn test_fn[type: DType](num_elements: Int) raises:
        var unit_test = UnitTest("test_elementwise_sequential")

        var a = 1.0 + Tensor[type].rand(num_elements)
        var b = 1.0 + Tensor[type].rand(a.shape())

        var actual = _tensor_mul(a, b, min_simds_per_core=num_elements + 1)
        var expected = a * b

        unit_test.assert_true(
            actual == expected, msg=str(type) + "_" + str(num_elements)
        )

    for num_elements in List(1, 5, 25, 125, 625):
        test_fn[DType.float32](num_elements[])
        test_fn[DType.float64](num_elements[])


fn test_fori_loop_increasing_by_1() raises:
    var unit_test = UnitTest("test_fori_loop_increasing_by_1")
    var actual = List[Int](capacity=5)
    var expected = List[Int](0, 1, 2, 3, 4)

    @always_inline
    @parameter
    fn func[i: Int]():
        actual.append(i)

    fori_loop[func, 0, 5, 1]()

    unit_test.assert_equal(len(actual), 5)
    unit_test.assert_equal(len(expected), 5)

    for i in range(len(actual)):
        unit_test.assert_equal(actual[i], expected[i])


fn test_fori_loop_decreasing_by_2() raises:
    var unit_test = UnitTest("test_fori_loop_decreasing_by_2")
    var actual = List[Int](capacity=3)
    var expected = List[Int](5, 3, 1)

    @always_inline
    @parameter
    fn func[i: Int]():
        actual.append(i)

    fori_loop[func, 5, -1, -2]()

    unit_test.assert_equal(len(actual), 3)
    unit_test.assert_equal(len(expected), 3)

    for i in range(len(actual)):
        unit_test.assert_equal(actual[i], expected[i])


fn test_fori_loop_with_conditional() raises:
    var unit_test = UnitTest("test_fori_loop_with_conditional")
    var actual = List[Int](capacity=3)
    var expected = List[Int](5, 4, 3)

    @always_inline
    @parameter
    fn func[i: Int]() -> Bool:
        actual.append(i)
        if i <= 3:
            return False
        return True

    fori_loop[func, 5, -1, -1]()

    unit_test.assert_equal(len(actual), 3)
    unit_test.assert_equal(len(expected), 3)

    for i in range(len(actual)):
        unit_test.assert_equal(actual[i], expected[i])


fn main() raises:
    test_elementwise()
    test_elementwise_parallel()
    test_elementwise_sequential()
    test_fori_loop_increasing_by_1()
    test_fori_loop_decreasing_by_2()
    test_fori_loop_with_conditional()
