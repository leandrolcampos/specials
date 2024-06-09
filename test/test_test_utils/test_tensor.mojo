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
# RUN: %mojo %build_dir %debug_level %sanitize_checks %s

"""Tests for tensor utilities."""

import math

from tensor import Tensor, TensorShape

from test_utils import UnitTest
from test_utils.tensor import elementwise, random_uniform


# TODO: Investigate why defining `MOJO_ENABLE_ASSERTIONS` leads to test failures.


fn test_unary_elementwise() raises:
    @parameter
    fn test_fn[type: DType, force_sequential: Bool](*shape: Int) raises:
        var unit_test = UnitTest("test_unary_elementwise")

        var x = random_uniform[type](1.0, 2.0, TensorShape(shape))

        var actual = elementwise[math.reciprocal](x)
        var expected = 1.0 / x

        unit_test.assert_true(
            actual == expected, msg=str(x.spec()) + "_" + str(force_sequential)
        )

    test_fn[DType.float32, force_sequential=False](64, 64)
    test_fn[DType.float32, force_sequential=True](64, 64)

    test_fn[DType.float64, force_sequential=False](64, 64)
    test_fn[DType.float64, force_sequential=True](64, 64)


fn test_binary_scalar_elementwise() raises:
    @parameter
    fn test_fn[type: DType, force_sequential: Bool](*shape: Int) raises:
        var unit_test = UnitTest("test_binary_scalar_elementwise")

        var x = random_uniform[type](1.0, 2.0, TensorShape(shape))

        var actual = elementwise[math.div](x, 2.0)
        var expected = x / 2.0

        unit_test.assert_true(
            actual == expected, msg=str(x.spec()) + "_" + str(force_sequential)
        )

    test_fn[DType.float32, force_sequential=False](64, 64)
    test_fn[DType.float32, force_sequential=True](64, 64)

    test_fn[DType.float64, force_sequential=False](64, 64)
    test_fn[DType.float64, force_sequential=True](64, 64)


fn test_binary_elementwise() raises:
    @parameter
    fn test_fn[type: DType, force_sequential: Bool](*shape: Int) raises:
        var unit_test = UnitTest("test_binary_elementwise")

        var x = random_uniform[type](1.0, 2.0, TensorShape(shape))
        var y = random_uniform[type](1.0, 2.0, x.shape())

        var actual = elementwise[math.div](x, y)
        var expected = x / y

        unit_test.assert_true(
            actual == expected, msg=str(x.spec()) + "_" + str(force_sequential)
        )

    test_fn[DType.float32, force_sequential=False](64, 64)
    test_fn[DType.float32, force_sequential=True](64, 64)

    test_fn[DType.float64, force_sequential=False](64, 64)
    test_fn[DType.float64, force_sequential=True](64, 64)


fn main() raises:
    test_unary_elementwise()
    test_binary_elementwise()
    test_binary_scalar_elementwise()
