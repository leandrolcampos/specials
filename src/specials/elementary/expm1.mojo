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
# Tang, P. T. P. (1992). Table-driven implementation of the Expm1 function in
#   IEEE floating-point arithmetic.
# ACM Transactions on Mathematical Software (TOMS), 18(2), 211-222.
# https://doi.org/10.1145/146847.146928

"""Implements the `expm1` function."""

import math

from memory.unsafe import bitcast

from specials._internal import math as math_lib
from specials._internal.asserting import assert_float_dtype
from specials._internal.polynomial import Polynomial
from specials.elementary.common_constants import ExpTable


@always_inline
fn _expm1_procedure_1[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width], cond: SIMD[DType.bool, simd_width]) -> SIMD[
    dtype, simd_width
]:
    """Implements the procedure 1 of `expm1` as specified in the reference paper.
    """
    var safe_x = cond.select(x, 1.0)

    var index: SIMD[DType.int32, simd_width]
    var exponent: SIMD[DType.int32, simd_width]
    var expm1_r: SIMD[dtype, simd_width]
    var precision_minus_1: SIMD[DType.int32, simd_width]

    @parameter
    if dtype == DType.float32:
        alias inv_ln2_over_32: SIMD[dtype, simd_width] = bitcast[
            dtype, DType.uint32
        ](
            0x4238_AA3B,
        )
        alias ln2_over_32_lead: SIMD[dtype, simd_width] = bitcast[
            dtype, DType.uint32
        ](
            0x3CB1_7200,
        )
        alias ln2_over_32_trail: SIMD[dtype, simd_width] = bitcast[
            dtype, DType.uint32
        ](
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
            math.fma(
                x_reduced * x_reduced, polynomial(x_reduced), x_reduced_trail
            )
        )
        precision_minus_1 = 23  # 24 - 1

    else:  # dtype == DType.float64
        alias inv_ln2_over_32: SIMD[dtype, simd_width] = bitcast[
            dtype, DType.uint64
        ](
            0x40471547_652B82FE,
        )
        alias ln2_over_32_lead: SIMD[dtype, simd_width] = bitcast[
            dtype, DType.uint64
        ](
            0x3F962E42_FEF00000,
        )
        alias ln2_over_32_trail: SIMD[dtype, simd_width] = bitcast[
            dtype, DType.uint64
        ](
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
            math.fma(
                x_reduced * x_reduced, polynomial(x_reduced), x_reduced_trail
            )
        )
        precision_minus_1 = 52  # 53 - 1

    var inv_exp2 = math.ldexp[dtype, simd_width](0.25, 2 - exponent)
    var s_lead = ExpTable[dtype].lead.unsafe_lookup(index)
    var s_trail = ExpTable[dtype].trail.unsafe_lookup(index)
    var s = s_lead + s_trail

    var mantissa = (s_lead - inv_exp2) + math.fma(
        s_lead, expm1_r, s_trail * (1.0 + expm1_r)
    )
    mantissa = math.select(
        exponent > precision_minus_1,
        s_lead + math.fma(s, expm1_r, s_trail - inv_exp2),
        mantissa,
    )

    var exponent_is_too_negative = (exponent <= -8.0)
    mantissa = math.select(
        exponent_is_too_negative,
        s_lead + math.fma(s, expm1_r, s_trail),
        mantissa,
    )

    var result = math_lib.ldexp(mantissa, exponent)
    result = math.select(exponent_is_too_negative, result - 1.0, result)

    return result


@always_inline
fn _expm1_procedure_2[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width], cond: SIMD[DType.bool, simd_width]) -> SIMD[
    dtype, simd_width
]:
    """Implements the procedure 2 of `expm1` as specified in the reference paper.
    """
    var safe_x = cond.select(x, 0.1)
    var x_exp2: SIMD[dtype, simd_width]
    var x3_gval: SIMD[dtype, simd_width]

    @parameter
    if dtype == DType.float32:
        alias exp2 = math.ldexp(Scalar[dtype](1.0), 16)
        x_exp2 = safe_x * exp2

        alias g = Polynomial[
            5, dtype, simd_width
        ].from_hexadecimal_coefficients[
            0x3E2A_AAAA,
            0x3D2A_AAA0,
            0x3C08_89FF,
            0x3AB6_4DE5,
            0x394A_B327,
        ]()
        x3_gval = safe_x * safe_x * safe_x * g(safe_x)

    else:  # dtype == DType.float64
        alias exp2 = math.ldexp(Scalar[dtype](1.0), 30)
        x_exp2 = safe_x * exp2

        alias g = Polynomial[
            9, dtype, simd_width
        ].from_hexadecimal_coefficients[
            0x3FC55555_55555549,
            0x3FA55555_555554B6,
            0x3F811111_1111A9F3,
            0x3F56C16C_16CE14C6,
            0x3F2A01A0_1159DD2D,
            0x3EFA019F_635825C4,
            0x3EC71E14_BFE3DB59,
            0x3E928295_484734EA,
            0x3E5A2836_AA646B96,
        ]()

        x3_gval = safe_x * safe_x * safe_x * g(safe_x)

    # x == x_term1 + x_term2
    var x_term1 = (x_exp2 + safe_x) - x_exp2
    var x_term2 = safe_x - x_term1

    # x * x * 0.5 == x2_half_term1 + x2_half_term2
    var x2_half_term1 = x_term1 * x_term1 * 0.5
    var x2_half_term2 = x_term2 * (safe_x + x_term1) * 0.5

    return math.select(
        x2_half_term1 < 0.0078125,
        safe_x + (x2_half_term1 + (x3_gval + x2_half_term2)),
        (x_term1 + x2_half_term1) + (x3_gval + (x_term2 + x2_half_term2)),
    )


fn expm1[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Computes `exp(x) - 1` in a numerically stable way.

    This function is semantically equivalent to `exp(x) - 1`, but it is more
    accurate for `x` close to zero.

    Parameters:
        dtype: The data type of the input and output SIMD vectors.
        simd_width: The width of the input and output SIMD vectors.

    Args:
        x: A SIMD vector of floating-point values.

    Returns:
        A SIMD vector containing the expression `exp(x) - 1` evaluated at `x`.

    Constraints:
        The data type must be a floating-point of single (`float32`) or double
        (`float64`) precision.
    """
    assert_float_dtype["dtype", dtype]()

    alias inf: SIMD[dtype, simd_width] = math.limit.inf[dtype]()

    var result: SIMD[dtype, simd_width] = math.nan[dtype]()
    var x_abs = math.abs(x)

    # Regions of computation
    var is_in_region1: SIMD[DType.bool, simd_width]  # Em1_Tiny | Em1_Zero
    var is_in_region2: SIMD[DType.bool, simd_width]  # Em1_Pos | Em1_+Inf
    var is_in_region3: SIMD[DType.bool, simd_width]  # Em1_Neg | Em1_-Inf
    var is_in_region4: SIMD[DType.bool, simd_width]  # T_1 < x < T_2
    var is_in_region5: SIMD[
        DType.bool, simd_width
    ]  # T- <= x <= T_1 | T_2 <= x <= T+

    @parameter
    if dtype == DType.float32:
        alias xeps: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint32](
            0x3300_0000,
        )
        alias xsml_inf: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint32](
            0xBE93_4B11,
        )
        alias xsml_sup: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint32](
            0x3E64_7FBF,
        )
        alias xmin: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint32](
            0xC18A_A122,
        )
        # `xmax` is different from what is specified in the reference paper:
        # `alias xmax = math.nextafter(log(FloatLimits[dtype].max()), 0.0)`
        alias xmax: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint32](
            0x42B1_7217,
        )

        is_in_region1 = x_abs < xeps
        is_in_region2 = x > xmax
        is_in_region3 = x < xmin
        is_in_region4 = ~is_in_region1 & (x > xsml_inf) & (x < xsml_sup)
        is_in_region5 = ((x >= xmin) & (x <= xsml_inf)) | (
            (x >= xsml_sup) & (x <= xmax)
        )

    else:  # dtype == DType.float64
        alias xeps: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint64](
            0x3C900000_00000000,
        )
        alias xsml_inf: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint64](
            0xBFD26962_1134DB93,
        )
        alias xsml_sup: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint64](
            0x3FCC8FF7_C79A9A22,
        )
        alias xmin: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint64](
            0xC042B708_872320E1,
        )
        # `xmax` is different from what is specified in the reference paper:
        # `alias xmax = log(FloatLimits[dtype].max())`
        alias xmax: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint64](
            0x40862E42_FEFA39EF,
        )

        is_in_region1 = x_abs < xeps
        is_in_region2 = x > xmax
        is_in_region3 = x < xmin
        is_in_region4 = ~is_in_region1 & (x > xsml_inf) & (x < xsml_sup)
        is_in_region5 = ((x >= xmin) & (x <= xsml_inf)) | (
            (x >= xsml_sup) & (x <= xmax)
        )

    result = is_in_region1.select(x, result)
    result = is_in_region2.select(inf, result)
    result = is_in_region3.select(-1.0, result)

    if is_in_region4.reduce_or():
        result = is_in_region4.select(
            _expm1_procedure_2(x, is_in_region4), result
        )

    if is_in_region5.reduce_or():
        result = is_in_region5.select(
            _expm1_procedure_1(x, is_in_region5), result
        )

    return result
