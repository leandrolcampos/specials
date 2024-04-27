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

"""Defines utilities to work with numeric types."""

from specials._internal.math import ldexp


fn _nmant_impl[dtype: DType]() -> Int:
    """Returns the floating-point arithmetic parameter `nmant`."""
    constrained[
        dtype == DType.float16
        or dtype == DType.float32
        or dtype == DType.float64,
        (
            "The parameter `dtype` must be one of `DType.float16`,"
            " `DType.float32`, or `DType.float64`."
        ),
    ]()

    @parameter
    if dtype == DType.float16:
        return 10
    elif dtype == DType.float32:
        return 23
    else:  # dtype == DType.float64
        return 52


fn _machep_impl[dtype: DType]() -> Int:
    """Returns the floating-point arithmetic parameter `machep`."""

    @parameter
    if dtype == DType.float16:
        return -10
    elif dtype == DType.float32:
        return -23
    else:  # dtype == DType.float64
        return -52


fn _negep_impl[dtype: DType]() -> Int:
    """Returns the floating-point arithmetic parameter `negep`."""

    @parameter
    if dtype == DType.float16:
        return -11
    elif dtype == DType.float32:
        return -24
    else:  # dtype == DType.float64
        return -53


fn _minexp_impl[dtype: DType]() -> Int:
    """Returns the floating-point arithmetic parameter `minexp`."""

    @parameter
    if dtype == DType.float16:
        return -14
    elif dtype == DType.float32:
        return -126
    else:  # dtype == DType.float64
        return -1022


fn _maxexp_impl[dtype: DType]() -> Int:
    """Returns the floating-point arithmetic parameter `maxexp`."""

    @parameter
    if dtype == DType.float16:
        return 16
    elif dtype == DType.float32:
        return 128
    else:  # dtype == DType.float64
        return 1024


fn _eps_impl[dtype: DType]() -> Scalar[dtype]:
    """Computes the floating-point arithmetic parameter `eps`."""
    alias machep = _machep_impl[dtype]()
    return ldexp[dtype, 1](1.0, machep)


fn _epsneg_impl[dtype: DType]() -> Scalar[dtype]:
    """Computes the floating-point arithmetic parameter `epsneg`."""
    alias negep = _negep_impl[dtype]()
    return ldexp[dtype, 1](1.0, negep)


fn _min_impl[dtype: DType]() -> Scalar[dtype]:
    """Returns the floating-point arithmetic parameter `min`."""
    alias minexp = _minexp_impl[dtype]()
    return ldexp[dtype, 1](1.0, minexp)


fn _max_impl[dtype: DType]() -> Scalar[dtype]:
    """Returns the floating-point arithmetic parameter `max`."""
    alias epsneg = _epsneg_impl[dtype]()
    alias maxexp = _maxexp_impl[dtype]()
    return ldexp[dtype, 1](2.0 * (1.0 - epsneg), maxexp - 1)


fn _smallest_subnormal_impl[dtype: DType]() -> Scalar[dtype]:
    """Returns the floating-point arithmetic parameter `smallest_subnormal`."""
    alias minexp = _minexp_impl[dtype]()
    alias nmant = _nmant_impl[dtype]()
    return ldexp[dtype, 1](1.0, minexp - nmant)


@register_passable("trivial")
struct FloatLimits[dtype: DType]:
    """Machine limits for IEEE 754-2008 binary floating-point types.

    Constraints:
        The underlying floating-point data type must be `DType.float16`, `DType.float32`,
        or `DType.float64`.

    Parameters:
        dtype: The floating-point data type for which `FloatLimits` returns information.
    """

    alias nmant: Int = _nmant_impl[dtype]()
    """The number of bits in the mantissa."""

    alias machep: Int = _machep_impl[dtype]()
    """The exponent that yields `eps`."""

    alias negep: Int = _negep_impl[dtype]()
    """The exponent that yields `epsneg`."""

    alias minexp: Int = _minexp_impl[dtype]()
    """
    The most negative power of the base-2 consistent with there being no leading
    zeros in the mantissa.
    """

    alias maxexp: Int = _maxexp_impl[dtype]()
    """The smallest positive power of the base-2 that causes overflow."""

    alias eps: Scalar[dtype] = _eps_impl[dtype]()
    """The smallest positive floating-point number such that `1.0 + eps != 1.0`."""

    alias epsneg: Scalar[dtype] = _epsneg_impl[dtype]()
    """The smallest positive floating-point number such that `1.0 - epsneg != 1.0`."""

    alias min: Scalar[dtype] = _min_impl[dtype]()
    """The smallest positive floating-point number."""

    alias max: Scalar[dtype] = _max_impl[dtype]()
    """The largest positive floating-point number."""

    alias smallest_subnormal: Scalar[dtype] = _smallest_subnormal_impl[dtype]()
    """
    The smallest positive floating point number with 0 as leading bit in the mantissa
    following IEEE-754.
    """
