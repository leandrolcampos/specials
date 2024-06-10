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
#
# References:
#
# Tang, P. T. P. (1989). Table-driven implementation of the exponential function
#   in IEEE floating-point arithmetic.
# ACM Transactions on Mathematical Software (TOMS), 15(2), 144-157.
# https://doi.org/10.1145/63522.214389
#
# Tang, P. T. P. (1991). Table-lookup algorithms for elementary functions and
#   their error analysis.
# Proceedings 10th IEEE Symposium on Computer Arithmetic, pp. 232-236.
# https://doi.org/10.1109/ARITH.1991.145565

"""Implements the base-2 exponential function."""

import math

from memory.unsafe import bitcast

from specials._internal import math as math_lib
from specials._internal.asserting import assert_float_dtype
from specials._internal.polynomial import Polynomial
from specials.elementary.common_constants import ExpTable


@always_inline
fn _exp2_impl[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width], cond: SIMD[DType.bool, simd_width]) -> SIMD[
    dtype, simd_width
]:
    """Implements the base-2 exponential function."""
    # Reduction: Find the breakpoint c[k] = k/32, k = 0, 1, ..., 31 such that
    #     |x - (m + c[k])| <= 1/64
    #   where m = -1, 0 or 1. Then calculate r by r = x - (m + c[k]). Note that
    #   r is in the range [-1/64, 1/64].
    # Approximation: Approximate 2^r - 1 by a polynomial p(r) whose coefficients
    #   were obtained using Sollya:
    #     > f = (2^x - 1)/x;
    #     > P = fpminimax(f, 2, [|single...|], [-1/64, 1/64]);
    #     or
    #     > P = fpminimax(f, 5, [|D...|], [-1/64, 1/64]);
    #     > P;
    #     > dirtyinfnorm(f-P, [-1/64, 1/64]);
    # Reconstruction: Reconstruct 2^x by the relationship
    #     2^x ~= 2^m * (2^c[k] + 2^c[k] * p(r))
    var safe_x = cond.select(x, 1.0)

    var index: SIMD[DType.int32, simd_width]
    var exponent: SIMD[DType.int32, simd_width]
    var exp2m1_r: SIMD[dtype, simd_width]

    @parameter
    if dtype == DType.float32:
        alias one_over_32: SIMD[dtype, simd_width] = bitcast[
            dtype, DType.uint32
        ](
            0x3D00_0000,
        )
        alias polynomial = Polynomial[
            3, dtype, simd_width
        ].from_hexadecimal_coefficients[
            0x3F31_7218,
            0x3E75_FE66,
            0x3D63_4D8A,
        ]()

        var xn = round(safe_x)
        var xf = safe_x - xn

        var yn = round(xf * 32.0)
        var yn2 = yn % 32.0
        var yn1 = yn - yn2

        var y_reduced = math.fma(-yn, one_over_32, xf)

        index = yn2.cast[DType.int32]()
        exponent = (xn + yn1 / 32).cast[DType.int32]()

        exp2m1_r = y_reduced * polynomial(y_reduced)

    else:  # dtype == DType.float64
        alias one_over_32: SIMD[dtype, simd_width] = bitcast[
            dtype, DType.uint64
        ](
            0x3FA00000_00000000,
        )
        alias polynomial = Polynomial[
            6, dtype, simd_width
        ].from_hexadecimal_coefficients[
            0x3FE62E42_FEFA39EF,
            0x3FCEBFBD_FF82C58E,
            0x3FAC6B08_D70496BC,
            0x3F83B2AB_6FBCFDA6,
            0x3F55D884_2A55CA01,
            0x3F24308B_04A657CB,
        ]()

        var xn = round(safe_x)
        var xf = safe_x - xn

        var yn = round(xf * 32.0)
        var yn2 = yn % 32.0
        var yn1 = yn - yn2

        var y_reduced = math.fma(-yn, one_over_32, xf)

        index = yn2.cast[DType.int32]()
        exponent = (xn + yn1 / 32).cast[DType.int32]()

        exp2m1_r = y_reduced * polynomial(y_reduced)

    var s_lead = ExpTable[dtype].lead.unsafe_lookup(index)
    var s_trail = ExpTable[dtype].trail.unsafe_lookup(index)
    var s = s_lead + s_trail

    var mantissa = s_lead + math.fma(s, exp2m1_r, s_trail)

    return math_lib.ldexp(mantissa, exponent)


fn exp2[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Computes the base-2 exponential of `x`.

    Parameters:
        dtype: The data type of the input and output SIMD vectors.
        simd_width: The width of the input and output SIMD vectors.

    Args:
        x: A SIMD vector of floating-point values.

    Returns:
        A SIMD vector containing the base-2 exponential of `x`.

    Constraints:
        The data type must be a floating-point of single (`float32`) or double
        (`float64`) precision.
    """
    assert_float_dtype["dtype", dtype]()

    alias inf: SIMD[dtype, simd_width] = math.inf[dtype]()

    var result: SIMD[dtype, simd_width] = math.nan[dtype]()
    var x_abs = abs(x)

    # Regions of computation
    var is_in_region1: SIMD[DType.bool, simd_width]  # abs(x) < xeps
    var is_in_region2: SIMD[DType.bool, simd_width]  # x > xmax
    var is_in_region3: SIMD[DType.bool, simd_width]  # x < xmin
    var is_in_region4: SIMD[
        DType.bool, simd_width
    ]  # xmin <= x <= -xeps or xeps <= x <= xmax

    @parameter
    if dtype == DType.float32:
        alias xeps: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint32](
            0x3300_0000,
        )
        # `xmax` is different from what is specified in the reference paper:
        # `alias xmax = math.nextafter(Float32(FloatLimits[dtype].max_exponent), 0.0)`
        alias xmax: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint32](
            0x42FF_FFFF,
        )
        alias xmin: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint32](
            0xC315_0000,
        )

        is_in_region1 = x_abs < xeps
        is_in_region2 = x > xmax
        is_in_region3 = x < xmin
        is_in_region4 = (x_abs >= xeps) & (x >= xmin) & (x <= xmax)

    else:  # dtype == DType.float64
        alias xeps: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint64](
            0x3C900000_00000000,
        )
        # `xmax` is different from what is specified in the reference paper:
        # `alias xmax = math.nextafter(Float64(FloatLimits[dtype].max_exponent), 0.0)`
        alias xmax: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint64](
            0x408FFFFF_FFFFFFFF,
        )
        alias xmin: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint64](
            0xC090C800_00000000,
        )

        is_in_region1 = x_abs < xeps
        is_in_region2 = x > xmax
        is_in_region3 = x < xmin
        is_in_region4 = (x_abs >= xeps) & (x >= xmin) & (x <= xmax)

    result = is_in_region1.select[dtype](1.0, result)
    result = is_in_region2.select(inf, result)
    result = is_in_region3.select[dtype](0.0, result)

    if is_in_region4.reduce_or():
        result = is_in_region4.select(_exp2_impl(x, is_in_region4), result)

    return result
