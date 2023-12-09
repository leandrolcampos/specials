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

"""Utilities for testing Specials code."""

import testing


@value
struct UnitTest:
    """Provides utilities for testing Specials code."""

    var name: String
    """The name of the unit test."""

    fn __init__(inout self, name: String) -> None:
        self.name = name
        print("# " + name)

    fn assert_true(self, val: Bool, msg: String) raises:
        """Asserts that the input value is `True`. If it is not then an `Error` is raised.

        Raises: An `Error` with the provided message if assert fails and `None` otherwise.

        Args:
            val: The value to assert to be `True`.
            msg: The message to be printed if the assertion fails.
        """
        try:
            testing.assert_true(val, msg)
        except e:
            print(e)

    fn assert_equal(self, lhs: Int, rhs: Int) raises:
        """Asserts that the input values are equal. If it is not then an `Error` is raised.

        Raises: An `Error` if assert fails and `None` otherwise.

        Args:
            lhs: The lhs of the equality.
            rhs: The rhs of the equality.
        """
        try:
            testing.assert_equal(lhs, rhs)
        except e:
            print(e)

    fn assert_equal[
        dtype: DType, simd_width: Int
    ](self, lhs: SIMD[dtype, simd_width], rhs: SIMD[dtype, simd_width]) raises:
        """Asserts that the input values are equal. If it is not then an `Error` is raised.

        Raises: An `Error` if assert fails and `None` otherwise.

        Parameters:
            dtype: The dtype of the left- and right-hand-side SIMD vectors.
            simd_width: The width of the left- and right-hand-side SIMD vectors.

        Args:
            lhs: The lhs of the equality.
            rhs: The rhs of the equality.
        """
        try:
            testing.assert_equal(lhs, rhs)
        except e:
            print(e)

    fn assert_almost_equal[
        dtype: DType, simd_width: Int
    ](
        self,
        lhs: SIMD[dtype, simd_width],
        rhs: SIMD[dtype, simd_width],
        absolute_tolerance: SIMD[dtype, 1],
        relative_tolerance: SIMD[dtype, 1],
    ) raises:
        """Asserts that the input values are equal up to a tolerance. If it is not then
        an `Error` is raised.

        Raises: An `Error` if assert fails and `None` otherwise.

        Parameters:
            dtype: The dtype of the left- and right-hand-side SIMD vectors.
            simd_width: The width of the left- and right-hand-side SIMD vectors.

        Args:
            lhs: The lhs of the equality.
            rhs: The rhs of the equality.
            absolute_tolerance: The absolute tolerance.
            relative_tolerance: The relative tolerance.
        """
        try:
            testing.assert_almost_equal(
                lhs, rhs, absolute_tolerance, relative_tolerance
            )
        except e:
            print(e)
