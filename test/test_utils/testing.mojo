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
# modularml/mojo
# Copyright (c) 2024, Modular Inc.
# Licensed under the Apache License v2.0 with LLVM Exceptions.
#
# numpy/numpy
# Copyright (c) 2005-2023, NumPy Developers.
# Licensed under BSD 3 clause.

"""Utilities for testing Specials code."""

import math
import testing


@value
struct UnitTest:
    """Provides utilities for testing Specials code."""

    var name: String
    """The name of the unit test."""

    fn __init__(inout self: Self, name: String):
        """Constructs a unit test with the given name.

        Args:
            name: The name of the unit test.
        """
        self.name = name

    fn _assert_true[T: Boolable](self: Self, val: T, *, msg: String) raises:
        """Base method for asserting that a value is `True`."""
        if not val.__bool__():
            raise Error("[" + self.name + "] AssertionError: " + msg)

    fn assert_true[T: Boolable](self: Self, val: T, *, msg: String = "") raises:
        """Asserts that the input value is `True`. If it is not then an `Error`
        is raised.

        Parameters:
            T: A type that can be converted to a bool.

        Args:
            val: The value to assert to be `True`.
            msg: The message to be printed if the assertion fails. Default is
                an empty string.

        Raises:
            An `Error` with the provided message if assert fails and `None`
            otherwise.
        """
        var err_msg: String = "condition was unexpectedly `False`"

        if msg:
            err_msg += " (" + msg + ")"

        self._assert_true(val, msg=err_msg)

    fn assert_equal(
        self: Self, actual: Int, desired: Int, *, msg: String = ""
    ) raises:
        """Asserts that the input values are equal. If it is not then an `Error`
        is raised.

        Args:
            actual: The actual integer.
            desired: The desired integer.
            msg: The message to be printed if the assertion fails. Default is
                an empty string.

        Raises:
            An `Error` with the provided message if assert fails and `None`
            otherwise.
        """
        var err_msg: String = str(actual) + " is not equal to " + str(desired)
        if msg:
            err_msg += " (" + msg + ")"

        self._assert_true(actual == desired, msg=err_msg)

    fn assert_equal[
        type: DType, width: Int
    ](
        self: Self,
        actual: SIMD[type, width],
        desired: SIMD[type, width],
        *,
        msg: String = "",
    ) raises:
        """Asserts that the input values are equal. If it is not then an `Error`
        is raised.

        In contrast to the standard usage in Mojo, `NaN`s are compared like
        numbers: no assertion is raised if both objects have `NaN`s in the
        same positions.

        Parameters:
            type: The type of the actual and desired SIMD vectors.
            width: The width of the actual and desired SIMD vectors.

        Args:
            actual: The actual SIMD vector.
            desired: The desired SIMD vector.
            msg: The message to be printed if the assertion fails. Default is
                an empty string.

        Raises:
            An `Error` with the provided message if assert fails and `None`
            otherwise.
        """
        var err_msg: String = str(actual) + " is not equal to " + str(desired)
        if msg:
            err_msg += " (" + msg + ")"

        var result = (actual == desired)

        @parameter
        if actual.type.is_floating_point():
            result |= math.isnan(actual) & math.isnan(desired)

        self._assert_true(all(result), msg=err_msg)

    fn assert_all_close[
        type: DType, width: Int
    ](
        self: Self,
        actual: SIMD[type, width],
        desired: SIMD[type, width],
        *,
        atol: Scalar[type] = 0.0,
        rtol: Scalar[type] = 1e-07,
        msg: String = "",
    ) raises:
        """Asserts that the input values are equal up to a tolerance. If it is
        not then an `Error` is raised.

        The test compares the difference between `actual` and `desired` to the
        sum of the absolute tolerance `atol` and the relative tolerance
        `rtol * abs(desired)`.

        In contrast to the standard usage in Mojo, `NaN`s are compared like
        numbers: no assertion is raised if both objects have `NaN`s in the same
        positions.

        Parameters:
            type: The type of the actual and desired SIMD vectors.
            width: The width of the actual and desired SIMD vectors.

        Args:
            actual: The actual SIMD vector.
            desired: The desired SIMD vector.
            atol: The absolute tolerance. Default is 0.0.
            rtol: The relative tolerance. Default is 1e-07.
            msg: The message to be printed if the assertion fails. Default is
                an empty string.

        Raises:
            An `Error` with the provided message if assert fails and `None`
            otherwise.
        """
        constrained[type.is_floating_point(), "type must be a floating-point"]()

        var diff = actual - desired
        var err_msg: String = str(actual) + " is not close to " + str(desired)
        err_msg += " with a diff of " + str(diff)
        if msg:
            err_msg += " (" + msg + ")"

        var result = (actual == desired)

        result |= math.isfinite(desired) & (
            abs(diff) <= (atol + rtol * abs(desired))
        )
        result |= math.isnan(actual) & math.isnan(desired)

        self._assert_true(all(result), msg=err_msg)
