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

"""Implements utilities to work with numeric types."""

import math


@always_inline
fn _float_limits_construction_checks[type: DType]():
    """Checks if the type is valid."""
    constrained[
        type == DType.float16 or type == DType.float32 or type == DType.float64,
        (
            "type must be an IEEE 754-2008 floating-point: `DType.float16`,"
            " `DType.float32`, or `DType.float64`."
        ),
    ]()


@always_inline
fn _digits[type: DType]() -> IntLiteral:
    """
    Returns the number of `radix` digits that the given type can represent
    without loss of precision.
    """
    _float_limits_construction_checks[type]()

    @parameter
    if type == DType.float16:
        return 11
    elif type == DType.float32:
        return 24
    else:
        debug_assert(type == DType.float64, "type must be `DType.float64`")
        return 53


@always_inline
fn _max_exponent[type: DType]() -> IntLiteral:
    """
    Returns the maximum positive integer such that `radix` raised to
    `(max_exponent-1)` generates a representable finite floating-point number.
    """

    @parameter
    if type == DType.float16:
        return 16
    elif type == DType.float32:
        return 128
    else:
        debug_assert(type == DType.float64, "type must be `DType.float64`")
        return 1024


@always_inline
fn _min_exponent[type: DType]() -> IntLiteral:
    """
    Returns the minimum negative integer such that `radix` raised to
    `(min_exponent-1)` generates a normalized floating-point number.
    """

    @parameter
    if type == DType.float16:
        return -13
    elif type == DType.float32:
        return -125
    else:
        debug_assert(type == DType.float64, "type must be `DType.float64`")
        return -1021


@register_passable("trivial")
struct FloatLimits[type: DType]:
    """Machine limits for floating-point types.

    Constraints:
        The type must be an IEEE 754-2008 floating-point: `DType.float16`,
        `DType.float32`, or `DType.float64`.

    Parameters:
        type: The floating-point for which `FloatLimits` returns information.
    """

    alias digits: IntLiteral = _digits[type]()
    """
    The number of `radix` digits that the given type can represent without loss
    of precision.
    """

    alias max_exponent: IntLiteral = _max_exponent[type]()
    """
    Maximum positive integer such that `radix` raised to `(max_exponent-1)`
    generates a representable finite floating-point number.
    """

    alias min_exponent: IntLiteral = _min_exponent[type]()
    """
    Minimum negative integer such that `radix` raised to `(min_exponent-1)`
    generates a normalized floating-point number.
    """

    alias radix: IntLiteral = 2
    """The integral base used for the representation of the given type."""

    @staticmethod
    @always_inline
    fn denorm_min() -> Scalar[type]:
        """Returns the minimum positive denormalized value of the given type."""
        return math.ldexp(
            math.ldexp[type, 1](1.0, Self.min_exponent), -Self.digits
        )

    @staticmethod
    @always_inline
    fn epsilon() -> Scalar[type]:
        """
        Returns the difference between 1.0 and the next representable value of
        the given type.
        """
        return math.ldexp[type, 1](1.0, -Self.digits + 1)

    @staticmethod
    @always_inline
    fn epsilon_neg() -> Scalar[type]:
        """
        Returns the difference between 1.0 and the previous representable value
        of the given type.
        """
        return math.ldexp[type, 1](1.0, -Self.digits)

    @staticmethod
    @always_inline
    fn lowest() -> Scalar[type]:
        """Returns the most negative finite value of the given type."""
        return -Self.max()

    @staticmethod
    @always_inline
    fn max() -> Scalar[type]:
        """Returns the maximum finite value of the given type."""
        return math.ldexp[type, 1](
            2.0 * (1.0 - Self.epsilon_neg()), Self.max_exponent - 1
        )

    @staticmethod
    @always_inline
    fn min() -> Scalar[type]:
        """Returns the minimum positive normalized value of the given type."""
        return math.ldexp[type, 1](1.0, Self.min_exponent - 1)
