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

"""Utilities for performing compile-time assertions within the package."""

import math


@always_inline("nodebug")
fn assert_integral_dtype[
    parameter_name: StringLiteral, parameter_value: DType
]() -> None:
    """Asserts that the given parameter is an integral data type."""
    constrained[
        parameter_value.is_integral(),
        "The parameter `" + parameter_name + "` must be an integral data type.",
    ]()


@always_inline("nodebug")
fn assert_float_dtype[parameter_name: StringLiteral, parameter_value: DType]() -> None:
    """Asserts that the given parameter is a floating-point data type.

    This package supports only floating-point data types of single (`float32`) or double
    (`float64`) precision.
    """
    constrained[
        parameter_value == DType.float32 or parameter_value == DType.float64,
        "The parameter `"
        + parameter_name
        + "` must be a floating-point of single (`float32`) or double (`float64`)"
        " precision.",
    ]()


@always_inline("nodebug")
fn assert_in_range[
    parameter_name: StringLiteral,
    parameter_value: Int,
    lower: Int,
    upper: Int,
]() -> None:
    """Asserts that the given parameter is within the specified range."""
    constrained[
        parameter_value >= lower and parameter_value < upper,
        "The parameter `" + parameter_name + "` must be within the specified range.",
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
    dtype: DType, parameter_name: StringLiteral, parameter_value: Scalar[dtype]
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
