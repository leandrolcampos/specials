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

"""Mathematical utilities for internal use."""

import math

from specials._internal.numerics import FloatLimits


fn ldexp[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width], exp: SIMD[DType.int32, simd_width]) -> SIMD[
    dtype,
    simd_width,
]:
    """Computes elementwise `ldexp` function.

    Compared to the standard library implementation, this implementation produces
    correct results when `exp` assumes values of greater magnitude.

    # TODO: Report this as a bug in the standard library.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        simd_width: The width of the input and output SIMD vector.

    Args:
        x: SIMD vector of floating point values.
        exp: SIMD vector containing the exponents.

    Returns:
        Vector containing elementwise result of `ldexp` on `x` and `exp`.
    """
    alias min_exponent: SIMD[DType.int32, simd_width] = FloatLimits[
        dtype
    ].minexp
    alias max_exponent: SIMD[DType.int32, simd_width] = FloatLimits[
        dtype
    ].maxexp - 1

    var result: SIMD[dtype, simd_width]

    if ((exp < min_exponent) | (exp > max_exponent)).reduce_or():
        var exponent_clipped = math.clamp(exp, min_exponent, max_exponent)
        var exponent_remainder = exp - exponent_clipped

        result = math.ldexp(math.ldexp(x, exponent_clipped), exponent_remainder)
    else:
        result = math.ldexp(x, exp)

    return result
