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
# Tang, P. T. P. (1990). Table-driven implementation of the logarithm function
#   in IEEE floating-point arithmetic.
# ACM Transactions on Mathematical Software (TOMS), 16(4), 378-400.
# https://doi.org/10.1145/98267.98294

"""Log1p function."""

import math

from memory.unsafe import bitcast

from specials._internal.asserting import assert_float_dtype
from specials._internal.math import ldexp
from specials._internal.numerics import FloatLimits
from specials._internal.polynomial import Polynomial
from specials.elementary.common_constants import LogConstants


@always_inline
fn _log1p_procedure_1[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width], cond: SIMD[DType.bool, simd_width]) -> SIMD[
    dtype, simd_width
]:
    """Implements the procedure 1 of `log1p` as specified in the reference paper.
    """
    alias max_exponent = FloatLimits[dtype].maxexp - 1
    alias significant_bits = FloatLimits[dtype].nmant + 1
    # There is no risk in using `math.ldexp` directly here.
    alias threshold = math.ldexp[dtype, simd_width](1.0, significant_bits + 2)

    var safe_x = cond.select(x, 1.0)
    var y = math.select(safe_x < threshold, 1.0 + safe_x, safe_x)

    var fraction_and_exponent = math.frexp(y)
    var fraction = 2.0 * fraction_and_exponent[0]
    var exponent = fraction_and_exponent[1] - 1

    var fraction1 = ldexp(math.round(ldexp(fraction, 7)), -7)
    var index = math.round(ldexp(fraction1 - 1.0, 7)).cast[DType.int32]()

    var power_of_two = ldexp[dtype, simd_width](
        1.0, -exponent.cast[DType.int32]()
    )
    var x_times_power_of_two = safe_x * power_of_two
    var fraction2 = math.select(
        (exponent <= -2) | (exponent >= max_exponent),
        fraction - fraction1,
        math.select(
            exponent >= significant_bits,
            (x_times_power_of_two - fraction1) + power_of_two,
            (power_of_two - fraction1) + x_times_power_of_two,
        ),
    )

    var log2_lead = LogConstants[dtype].log_fraction1_lead.get[128]()
    var log2_trail = LogConstants[dtype].log_fraction1_trail.get[128]()

    var result1 = math.fma[dtype, simd_width](
        exponent,
        log2_lead,
        LogConstants[dtype].log_fraction1_lead.unsafe_lookup(index),
    )
    var result2 = math.fma[dtype, simd_width](
        exponent,
        log2_trail,
        LogConstants[dtype].log_fraction1_trail.unsafe_lookup(index),
    )

    var u = fraction2 * LogConstants[dtype].inv_fraction1.unsafe_lookup(index)
    var u_squared_times_pval: SIMD[dtype, simd_width]

    # We use Sollya to find a polynomial p such that p(u) best approximates the
    # function f(u) = (log1p(u) - u)/u**2 on the given interval:
    #   > min_value = -2^(-8);
    #   > max_value = -min_value;
    #   > f = (log1p(u) - u)/u^2;
    #   > P = fpminimax(f, 1, [|single...|], [min_value, max_value]);
    #   or
    #   > P = fpminimax(f, 4, [|D...|], [min_value, max_value]);
    #   > P;
    #   > dirtyinfnorm(f-P, [min_value, max_value]);

    @parameter
    if dtype == DType.float32:
        alias p = Polynomial[
            2, dtype, simd_width
        ].from_hexadecimal_coefficients[
            0xBF00_0020,
            0x3EAA_AAE6,
        ]()
        u_squared_times_pval = u * u * p(u)

    else:  # dtype == DType.float64
        alias p = Polynomial[
            5, dtype, simd_width
        ].from_hexadecimal_coefficients[
            0xBFE00000_00000000,
            0x3FD55555_555279E5,
            0xBFCFFFFF_FFFA0C2B,
            0x3FC999B0_7518C512,
            0xBFC5556A_79F895CB,
        ]()
        u_squared_times_pval = u * u * p(u)

    result2 = u + (u_squared_times_pval + result2)

    return result1 + result2


@always_inline
fn _log1p_procedure_2[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width], cond: SIMD[DType.bool, simd_width]) -> SIMD[
    dtype, simd_width
]:
    """Implements the procedure 2 of `log1p` as specified in the reference paper.
    """
    var safe_x = cond.select(x, 0.0)
    var inv_x_plus_two = math.reciprocal(safe_x + 2.0)
    var u = 2.0 * safe_x * inv_x_plus_two
    var u_squared = u * u

    var u_cubed_times_pval: SIMD[dtype, simd_width]
    var precision_shift: Int

    # We use Sollya to find an even polynomial p such that p(u) best approximates
    # (log1p(2*u/(2 - u)) - u)/u**3 on the given interval:
    #   > min_value = (2 - 2*exp(1/16))/(1 + exp(1/16));
    #   > max_value = -min_value;
    #   > f = (log1p(2*u/(2 - u)) - u)/u^3;
    #   > P = fpminimax(f, [|0,2|], [|single...|], [min_value, max_value]);
    #   or
    #   > P = fpminimax(f, [|0,2,4,6|], [|D...|], [min_value, max_value]);
    #   > P;
    #   > dirtyinfnorm(f-P, [min_value, max_value]);

    @parameter
    if dtype == DType.float32:
        alias p = Polynomial[
            2, dtype, simd_width
        ].from_hexadecimal_coefficients[
            0x3DAA_AAAA,
            0x3C4C_F264,
        ]()
        u_cubed_times_pval = u * u_squared * p(u_squared)

        # precision_shift: 24 - 12 + 1
        precision_shift = 13

    else:  # dtype == DType.float64
        alias p = Polynomial[
            4, dtype, simd_width
        ].from_hexadecimal_coefficients[
            0x3FB55555_5555554A,
            0x3F899999_99A528F3,
            0x3F624923_AA1832F2,
            0x3F3C7D68_D285CA94,
        ]()

        u_cubed_times_pval = u * u_squared * p(u_squared)

        # precision_shift: 53 - 24 + 1
        precision_shift = 30

    # (u == u_term1 + u_term2) and (x == x_term1 + x_term2)
    var u_precision_scale = ldexp(u, precision_shift)
    var u_term1 = (u_precision_scale + u) - u_precision_scale
    var x_precision_scale = ldexp(safe_x, precision_shift)
    var x_term1 = (x_precision_scale + safe_x) - x_precision_scale
    var x_term2 = safe_x - x_term1
    var u_term2 = inv_x_plus_two * (
        math.fma(
            -u_term1,
            x_term2,
            math.fma(-u_term1, x_term1, 2.0 * (safe_x - u_term1)),
        )
    )

    return u_term1 + (u_term2 + u_cubed_times_pval)


@always_inline
fn _log1p_procedure_3[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width], cond: SIMD[DType.bool, simd_width]) -> SIMD[
    dtype, simd_width
]:
    """Implements the procedure of `log1p` for when `x` is tiny."""
    alias smallest_subnormal = FloatLimits[dtype].smallest_subnormal
    var safe_x = cond.select(x, 0.5 * FloatLimits[dtype].epsneg)

    return 0.125 * math.fma[dtype, simd_width](8.0, safe_x, -smallest_subnormal)


fn log1p[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Computes `log(1 + x)` in a numerically stable way.

    This function is semantically equivalent to `log(1 + x)`, but it is more
    accurate for `x` close to zero.

    Parameters:
        dtype: The data type of the input and output SIMD vectors.
        simd_width: The width of the input and output SIMD vectors.

    Args:
        x: A SIMD vector of floating-point values.

    Returns:
        A SIMD vector containing the expression `log(1 + x)` evaluated at `x`.

    Constraints:
        The data type must be a floating-point of single (`float32`) or double
        (`float64`) precision.
    """
    assert_float_dtype["dtype", dtype]()

    alias epsneg = FloatLimits[dtype].epsneg
    alias inf: SIMD[dtype, simd_width] = math.limit.inf[dtype]()

    var result: SIMD[dtype, simd_width] = math.nan[dtype]()
    var x_abs = math.abs(x)

    # Regions of computation
    var is_in_region1 = (x == inf)
    var is_in_region2 = (x_abs == 0.0)
    var is_in_region3 = (x == -1.0)
    var is_in_region4 = (x_abs != 0.0) & (x_abs < epsneg)
    var is_in_region5: SIMD[
        DType.bool, simd_width
    ]  # (x_abs >= epsneg) & (x > xsml_inf) & (x < xsml_sup)
    var is_in_region6: SIMD[
        DType.bool, simd_width
    ]  # ((x > -1.0) & (x <= xsml_inf)) | ((x >= xsml_sup) & (x < inf))

    @parameter
    if dtype == DType.float32:
        alias xsml_inf: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint32](
            0xBD78_2A03,
        )
        alias xsml_sup: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint32](
            0x3D84_15AC,
        )

        is_in_region5 = (x_abs >= epsneg) & (x > xsml_inf) & (x < xsml_sup)
        is_in_region6 = ((x > -1.0) & (x <= xsml_inf)) | (
            (x >= xsml_sup) & (x < inf)
        )

    else:  # dtype == DType.float64
        alias xsml_inf: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint64](
            0xBFAF0540_438FD5C4,
        )
        alias xsml_sup: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint64](
            0x3FB082B5_77D34ED8,
        )

        is_in_region5 = (x_abs >= epsneg) & (x > xsml_inf) & (x < xsml_sup)
        is_in_region6 = ((x > -1.0) & (x <= xsml_inf)) | (
            (x >= xsml_sup) & (x < inf)
        )

    result = (is_in_region1 | is_in_region2).select(x, result)
    result = is_in_region3.select(-inf, result)

    # TODO: Should we avoid creating runtime branches to be accelerator friendly?
    if is_in_region4.reduce_or():
        result = is_in_region4.select(
            _log1p_procedure_3(x, is_in_region4), result
        )

    if is_in_region5.reduce_or():
        result = is_in_region5.select(
            _log1p_procedure_2(x, is_in_region5), result
        )

    if is_in_region6.reduce_or():
        result = is_in_region6.select(
            _log1p_procedure_1(x, is_in_region6), result
        )

    return result
