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

"""Utilities for performing assertions within the package."""

import math


@always_inline("nodebug")
fn assert_float_dtype[parameter_name: StringLiteral, parameter_value: DType]() -> None:
    """Asserts that the given parameter is a floating-point data type.

    This package supports only floating-point data types of single (float32) or double
    (float64) precision.
    """
    constrained[
        parameter_value == DType.float32 or parameter_value == DType.float64,
        "The parameter `"
        + parameter_name
        + "` must be a floating-point of single (float32) or double (float64)"
        " precision.",
    ]()


@always_inline("nodebug")
fn assert_in_range[
    parameter_name: StringLiteral,
    parameter_value: Int,
    lower: Int,
    upper: Int,
]() -> None:
    """Asserts that the given parameter is in the given range."""
    constrained[
        parameter_value >= lower and parameter_value < upper,
        "The parameter `" + parameter_name + "` must be in the given range.",
    ]()


@always_inline("nodebug")
fn assert_non_zero[parameter_name: StringLiteral, parameter_value: Int]() -> None:
    """Asserts the condition `parameter_value != 0` holds."""
    constrained[
        parameter_value != 0,
        "The parameter `" + parameter_name + "` must be non-zero.",
    ]()


@always_inline("nodebug")
fn assert_positive[parameter_name: StringLiteral, parameter_value: Int]() -> None:
    """Asserts the condition `parameter_value > 0` holds."""
    constrained[
        parameter_value > 0,
        "The parameter `" + parameter_name + "` must be positive.",
    ]()


@always_inline("nodebug")
fn assert_positive[
    dtype: DType, parameter_name: StringLiteral, parameter_value: SIMD[dtype, 1]
]() -> None:
    """Asserts the condition `parameter_value > 0` holds."""
    constrained[
        parameter_value > 0,
        "The parameter `" + parameter_name + "` must be positive.",
    ]()


@always_inline("nodebug")
fn assert_simd_width[parameter_name: StringLiteral, parameter_value: Int]() -> None:
    """Asserts that the given parameter is a valid SIMD width."""
    constrained[
        parameter_value > 0 and math.is_power_of_2(parameter_value),
        "The parameter `" + parameter_name + "` must be positive and a power of two.",
    ]()
