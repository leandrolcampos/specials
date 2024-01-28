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
#
# Some of the code in this file is adapted from:
#
# numpy/numpy
# Copyright (c) 2005-2023, NumPy Developers.
# Licensed under BSD 3 clause.

"""Utilities for testing Specials code."""

import math
import testing


@value
struct UnitTest[raise_error: Bool = False]:
    """Provides utilities for testing Specials code.

    Parameters:
        raise_error: Whether to raise an `Error` if an assertion fails. If `False` then
            the assertion message is printed instead.
    """

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
        except error:

            @parameter
            if raise_error:
                raise error
            else:
                print(error)

    fn assert_equal(self, actual: Int, desired: Int) raises:
        """Asserts that the input values are equal. If it is not then an `Error` is raised.

        Raises: An `Error` if assert fails and `None` otherwise.

        Args:
            actual: The actual integer.
            desired: The desired integer.
        """
        try:
            testing.assert_equal(actual, desired)
        except error:

            @parameter
            if raise_error:
                raise error
            else:
                print(error)

    fn assert_equal[
        dtype: DType, simd_width: Int
    ](self, actual: SIMD[dtype, simd_width], desired: SIMD[dtype, simd_width]) raises:
        """Asserts that the input values are equal. If it is not then an `Error` is raised.

        In contrast to the standard usage in Mojo, `NaN`s are compared like numbers: no
        assertion is raised if both objects have `NaN`s in the same positions.

        Raises: An `Error` if assert fails and `None` otherwise.

        Parameters:
            dtype: The dtype of the actual and desired SIMD vectors.
            simd_width: The width of the actual and desired SIMD vectors.

        Args:
            actual: The actual SIMD vector.
            desired: The desired SIMD vector.
        """
        var result = (actual == desired)

        @parameter
        if dtype.is_floating_point():
            result |= math.isnan(actual) & math.isnan(desired)

        try:
            testing.assert_true(
                result.reduce_and(), str(actual) + " is not equal to " + str(desired)
            )
        except error:

            @parameter
            if raise_error:
                raise error
            else:
                print(error)

    fn assert_all_close[
        dtype: DType, simd_width: Int
    ](
        self,
        actual: SIMD[dtype, simd_width],
        desired: SIMD[dtype, simd_width],
        absolute_tolerance: SIMD[dtype, 1],
        relative_tolerance: SIMD[dtype, 1],
    ) raises:
        """Asserts that the input values are equal up to a tolerance. If it is not then
        an `Error` is raised.

        The test compares the difference between `actual` and `desired` to the sum of the
        absolute tolerance `atol` and the relative tolerance `rtol * abs(desired)`.

        In contrast to the standard usage in Mojo, `NaN`s are compared like numbers: no
        assertion is raised if both objects have `NaN`s in the same positions.

        Raises: An `Error` if assert fails and `None` otherwise.

        Parameters:
            dtype: The dtype of the actual and desired SIMD vectors.
            simd_width: The width of the actual and desired SIMD vectors.

        Args:
            actual: The actual SIMD vector.
            desired: The desired SIMD vector.
            absolute_tolerance: The absolute tolerance.
            relative_tolerance: The relative tolerance.
        """
        constrained[
            dtype.is_floating_point(),
            "The parameter `dtype` should be a floating-point data type.",
        ]()

        let diff = actual - desired
        var result = (actual == desired)

        result |= math.limit.isfinite(desired) & math.less_equal(
            math.abs(diff),
            absolute_tolerance + relative_tolerance * math.abs(desired),
        )

        @parameter
        if dtype.is_floating_point():
            result |= math.isnan(actual) & math.isnan(desired)

        try:
            testing.assert_true(
                result.reduce_and(),
                str(actual)
                + " is not close to "
                + str(desired)
                + " with a diff of "
                + str(diff),
            )
        except error:

            @parameter
            if raise_error:
                raise error
            else:
                print(error)
