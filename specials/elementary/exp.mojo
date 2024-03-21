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

"""Exponential function."""

import math

from memory.unsafe import bitcast

from specials._internal import math as math_lib
from specials._internal.asserting import assert_float_dtype
from specials._internal.polynomial import Polynomial
from specials.elementary.common_constants import ExpTable


@always_inline
fn _exp_impl[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width], cond: SIMD[DType.bool, simd_width]) -> SIMD[
    dtype, simd_width
]:
    """Implements the exponential function as specified in the reference paper."""
    var safe_x = cond.select(x, 1.0)

    var index: SIMD[DType.int32, simd_width]
    var exponent: SIMD[DType.int32, simd_width]
    var expm1_r: SIMD[dtype, simd_width]

    @parameter
    if dtype == DType.float32:
        alias inv_ln2_over_32: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint32](
            0x4238_AA3B,
        )
        alias ln2_over_32_lead: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint32](
            0x3CB1_7200,
        )
        alias ln2_over_32_trail: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint32](
            0x333F_BE8E,
        )
        alias polynomial = Polynomial[
            2, dtype, simd_width
        ].from_hexadecimal_coefficients[
            0x3F00_0044,
            0x3E2A_AAEC,
        ]()

        var xn = math.round(safe_x * inv_ln2_over_32)
        var xn2 = math.mod(xn, 32.0)
        var xn1 = xn - xn2

        var xn_is_large = (math.abs(xn) >= 512)
        var x_reduced_lead = math.fma(
            -xn_is_large.select(xn1, xn), ln2_over_32_lead, safe_x
        )
        x_reduced_lead = xn_is_large.select(
            math.fma(-xn2, ln2_over_32_lead, x_reduced_lead), x_reduced_lead
        )
        var x_reduced_trail = -xn * ln2_over_32_trail

        index = xn2.cast[DType.int32]()
        exponent = xn1.cast[DType.int32]() / 32

        var x_reduced = x_reduced_lead + x_reduced_trail

        expm1_r = x_reduced_lead + (
            math.fma(x_reduced * x_reduced, polynomial(x_reduced), x_reduced_trail)
        )

    else:  # dtype == DType.float64
        alias inv_ln2_over_32: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint64](
            0x40471547_652B82FE,
        )
        alias ln2_over_32_lead: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint64](
            0x3F962E42_FEF00000,
        )
        alias ln2_over_32_trail: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint64](
            0x3D8473DE_6AF278ED,
        )
        alias polynomial = Polynomial[
            5, dtype, simd_width
        ].from_hexadecimal_coefficients[
            0x3FE00000_00000000,
            0x3FC55555_55548F7C,
            0x3FA55555_55545D4E,
            0x3F811115_B7AA905E,
            0x3F56C172_8D739765,
        ]()

        var xn = math.round(safe_x * inv_ln2_over_32)
        var xn2 = math.mod(xn, 32.0)
        var xn1 = xn - xn2

        var xn_is_large = (math.abs(xn) >= 512)
        var x_reduced_lead = math.fma(
            -xn_is_large.select(xn1, xn), ln2_over_32_lead, safe_x
        )
        x_reduced_lead = xn_is_large.select(
            math.fma(-xn2, ln2_over_32_lead, x_reduced_lead), x_reduced_lead
        )
        var x_reduced_trail = -xn * ln2_over_32_trail

        index = xn2.cast[DType.int32]()
        exponent = xn1.cast[DType.int32]() / 32

        var x_reduced = x_reduced_lead + x_reduced_trail

        expm1_r = x_reduced_lead + (
            math.fma(x_reduced * x_reduced, polynomial(x_reduced), x_reduced_trail)
        )

    var s_lead = ExpTable[dtype].lead.unsafe_lookup(index)
    var s_trail = ExpTable[dtype].trail.unsafe_lookup(index)
    var s = s_lead + s_trail

    var mantissa = s_lead + math.fma(s, expm1_r, s_trail)

    return math_lib.ldexp(mantissa, exponent)


fn exp[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Computes the exponential of `x`.

    Parameters:
        dtype: The data type of the input and output SIMD vectors.
        simd_width: The width of the input and output SIMD vectors.

    Args:
        x: A SIMD vector of floating-point values.

    Returns:
        A SIMD vector containing the exponential of `x`.

    Constraints:
        The data type must be a floating-point of single (`float32`) or double
        (`float64`) precision.
    """
    assert_float_dtype["dtype", dtype]()

    alias inf: SIMD[dtype, simd_width] = math.limit.inf[dtype]()

    var result: SIMD[dtype, simd_width] = math.nan[dtype]()
    var x_abs = math.abs(x)

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
        # `alias xmax = math.nextafter(log(FloatLimits[dtype].max), 0.0)`
        alias xmax: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint32](
            0x42B1_7217,
        )
        alias xmin: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint32](
            0xC2CE_8ECF,
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
        # `alias xmax = log(FloatLimits[dtype].max)`
        alias xmax: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint64](
            0x40862E42_FEFA39EF,
        )
        alias xmin: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint64](
            0xC0874385_446D71C3,
        )

        is_in_region1 = x_abs < xeps
        is_in_region2 = x > xmax
        is_in_region3 = x < xmin
        is_in_region4 = (x_abs >= xeps) & (x >= xmin) & (x <= xmax)

    result = is_in_region1.select(1.0, result)
    result = is_in_region2.select(inf, result)
    result = is_in_region3.select(0.0, result)

    if is_in_region4.reduce_or():
        result = is_in_region4.select(_exp_impl(x, is_in_region4), result)

    return result
