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
# slatec/fnlib (https://www.netlib.org/slatec/fnlib):
# Public-domain software. No copyright restrictions.
#
# References:
#
# Didonato, A. R., & Morris Jr, A. H. (1992). Algorithm 708: Significant digit
#   computation of the incomplete beta function ratios.
# ACM Transactions on Mathematical Software (TOMS), 18(3), 360-373.
# https://dl.acm.org/doi/abs/10.1145/131766.131776

"""Gamma-related functions."""

import math

from specials._internal import asserting
from specials._internal.numerics import FloatLimits
from specials._internal.polynomial import Chebyshev, Polynomial
from specials.elementary.log import log


fn lbeta[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width], y: SIMD[dtype, simd_width]) -> SIMD[
    dtype, simd_width
]:
    """Computes the natural logarithm of the beta function.

    This function is semantically equivalent to `lgamma(x) + lgamma(y) - lgamma(x + y)`,
    but it is more accurate for arguments greater than or equal to `8.0`.

    Parameters:
        dtype: The data type of the input and output SIMD vectors.
        simd_width: The width of the input and output SIMD vectors.

    Args:
        x: SIMD vector of non-negative floating-point values.
        y: SIMD vector of non-negative floating-point values.

    Returns:
        SIMD vector containing the natural logarithm of the beta function.

    Constraints:
        The data type must be a floating-point of single (`float32`) or double
        (`float64`) precision.
    """
    asserting.assert_float_dtype["dtype", dtype]()

    alias inf: SIMD[dtype, simd_width] = math.limit.inf[dtype]()
    alias nan: SIMD[dtype, simd_width] = math.nan[dtype]()
    alias log_sqrt_2pi: SIMD[
        dtype, simd_width
    ] = 0.91893853320467274178032973640562

    # Ensure that `a` is the smaller of the two arguments and `b` is the larger one.
    # Although the Beta function is mathematically symmetric, this procedure is not.
    var a = math.min(x, y)
    var b = math.max(x, y)

    # The `math.lgamma`` operation is one of the most computationally expensive
    # operations in this procedure. To avoid calling it when possible, we mask out
    # large values of `a` and `b`.
    var a_small = math.select(a < 8.0, a, nan)
    var b_small = math.select(b < 8.0, b, nan)

    var lgamma_a_small = math.lgamma(a_small)
    var apb = a + b
    var a_over_apb = a / apb
    var log1p_neg_a_over_apb = math.log1p(-a_over_apb)

    # `a` and `b` are small: `a <= b < 8.0`.
    var result = lgamma_a_small + math.lgamma(b_small) - math.lgamma(
        a_small + b_small
    )

    # `a` is small, but `b` is large: `a < 8.0 <= b`.
    var correction = lgamma_correction(b) - lgamma_correction(apb)
    var result_for_large_b = (
        lgamma_a_small
        + correction
        + a
        - a * log(apb)
        + (b - 0.5) * log1p_neg_a_over_apb
    )
    result = math.select(b >= 8.0, result_for_large_b, result)

    # `a` and `b` are large: `8.0 <= a <= b`.
    correction += lgamma_correction(a)
    var result_for_large_a = (
        -0.5 * log(b)
        + log_sqrt_2pi
        + correction
        + (a - 0.5) * log(a_over_apb)
        + b * log1p_neg_a_over_apb
    )
    result = math.select(a >= 8.0, result_for_large_a, result)

    # We have already computed the value of the log-beta function for positive arguments.
    # For other cases, this procedure returns the same values as the corresponding one in
    # the R language.
    return math.select(
        (a < 0.0) | math.isnan(x) | math.isnan(y),
        nan,
        math.select(
            a == 0.0,
            inf,
            math.select(
                math.limit.isinf(b),
                -inf,
                result,
            ),
        ),
    )


fn lgamma_correction[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Computes the correction term for the Rocktaeschel's approximation of `lgamma`.

    The correction term is defined as:

    `lgamma_correction(x) = lgamma(x) - (x - 0.5) * log(x) + x - 0.5 * log(2 * pi)`

    for `x >= 8`.

    Parameters:
        dtype: The data type of the input and output SIMD vectors.
        simd_width: The width of the input and output SIMD vectors.

    Args:
        x: SIMD vector of floating-point values greater than or equal to `8.0`.

    Returns:
        SIMD vector containing the correction term for the Rocktaeschel's approximation
        of `lgamma`. If `x` is less than `8.0`, this function returns `NaN`.

    Constraints:
        The data type must be a floating-point of single (`float32`) or double
        (`float64`) precision.
    """
    asserting.assert_float_dtype["dtype", dtype]()

    alias nan: SIMD[dtype, simd_width] = math.nan[dtype]()
    alias zero: SIMD[dtype, simd_width] = 0.0

    alias xmin: SIMD[dtype, simd_width] = 8.0
    alias xbig: SIMD[dtype, simd_width] = math.reciprocal(
        math.exp2[dtype, 1](0.5 * FloatLimits[dtype].negep)
    )
    alias xmax: SIMD[dtype, simd_width] = math.reciprocal(
        12.0 * FloatLimits[dtype].min
    )

    # The coefficients for the Chebyshev approximation of this correction were obtained
    # using the Python library `mpmath`.
    alias p = Chebyshev[20, dtype, simd_width].from_coefficients[
        8.331170390906488010133812318436e-02,
        -2.16055508054460412844538843806e-05,
        2.380513030666125633967836809341e-08,
        -6.79698274141255339847704839014e-11,
        3.598298746801337252645642260689e-13,
        -3.00664186830727299934841316608e-15,
        3.600735976941671612063259733719e-17,
        -5.79169539359268115708965063230e-19,
        1.193723234577627766675645918486e-20,
        -3.04227079051491795027184994902e-22,
        9.322841959636230355753533026206e-24,
        -3.35893572063489139361118640771e-25,
        1.396918849726567542569514379524e-26,
        -6.60413093595761811539762816839e-28,
        3.503829528301945510964075209918e-29,
        -2.06344897851678280985454179987e-30,
        1.336227216567260503733730549409e-31,
        -9.43764302155440072371370822355e-33,
        7.218203478875394218977123232114e-34,
        -5.89320165797572035218853002751e-35,
    ]()
    alias error_tolerance = 0.1 * FloatLimits[dtype].epsneg
    alias num_terms = p.economize[error_tolerance]()
    alias p_truncated = p.truncate[num_terms]()

    return math.select(
        (x < xmin) | math.isnan(x),
        nan,
        math.select(
            x < xbig,
            p_truncated(2.0 * math.pow(xmin / x, 2) - 1.0) / x,
            math.select(
                x < xmax,
                math.reciprocal(12.0 * x),
                zero,
            ),
        ),
    )


fn lgamma1p[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Computes `lgamma(1 + x)` in a numerically stable way.

    This function is semantically equivalent to `lgamma(1 + x)`, but it is more
    accurate for `x` close to zero.

    Parameters:
        dtype: The data type of the input and output SIMD vectors.
        simd_width: The width of the input and output SIMD vectors.

    Args:
        x: SIMD vector of floating-point values.

    Returns:
        SIMD vector containing the expression `lgamma(1 + x)` evaluated at `x`.

    Constraints:
        The data type must be a floating-point of single (`float32`) or double
        (`float64`) precision.
    """
    asserting.assert_float_dtype["dtype", dtype]()

    alias nan: SIMD[dtype, simd_width] = math.nan[dtype]()
    var result: SIMD[dtype, simd_width] = nan

    # Regions of computation.
    var is_in_region1 = (x >= -0.2) & (x < 0.6)
    var is_in_region2 = (x >= 0.6) & (x <= 1.25)
    var is_in_region3 = ~math.isnan(x) & ~is_in_region1 & ~is_in_region2

    # Polynomials for region 1. The coefficients for the Padé approximation were
    # obtained using the Python library `mpmath`.
    alias p = Polynomial[10, dtype, simd_width].from_coefficients[
        5.772156649015328606065120900824e-1,
        1.769751426777699103134469694093e-0,
        1.571904140511368034267480819223e-0,
        -4.57882665358839512689779140447e-1,
        -1.72712505786380004829886606981e-0,
        -1.24373712528022745342232844850e-0,
        -4.17229580597323137925159852465e-1,
        -6.80978370725741258151865551687e-2,
        -4.71020922504118253059534042963e-3,
        -8.87567923452439608685161841459e-5,
    ]()
    alias q = Polynomial[10, dtype, simd_width].from_coefficients[
        1.000000000000000000000000000000e-0,
        4.490901092651424325538968592651e-0,
        8.428109112438682661243930563021e-0,
        8.567162656125254544979174422045e-0,
        5.110442815300870959225621274210e-0,
        1.811088008784189174050238153628e-0,
        3.678104258279395409229240536674e-1,
        3.891333138124453879127500000527e-2,
        1.741014553601874329848935439309e-3,
        1.896441997532694197492403697806e-5,
    ]()

    # Polynomials for region 2. The coefficients for the Padé approximation were
    # obtained using the Python library `mpmath`.
    alias r = Polynomial[8, dtype, simd_width].from_coefficients[
        4.227843350984671393934879099176e-1,
        1.050000850494737509155499279591e-0,
        9.812533673494664828746361809635e-1,
        4.486129361904137782151622525624e-1,
        1.066177232215367809039427008258e-1,
        1.267871740982719010450401143716e-2,
        6.461232819244555998963476071186e-4,
        9.044855054775925733727539415320e-6,
    ]()
    alias s = Polynomial[8, dtype, simd_width].from_coefficients[
        1.000000000000000000000000000000e-0,
        1.720815452874289951756729496983e-0,
        1.167733459492857090468456899665e-0,
        3.958932481495390425060318675588e-1,
        6.995177647341877884678121833272e-2,
        6.082671403258376707307085732028e-3,
        2.160591173994851665996054475890e-4,
        1.832407275230925220950146383070e-6,
    ]()

    if is_in_region1.reduce_or():
        result = is_in_region1.select(-x * (p(x) / q(x)), result)

    if is_in_region2.reduce_or():
        var y = (x - 0.5) - 0.5
        result = is_in_region2.select(y * (r(y) / s(y)), result)

    if is_in_region3.reduce_or():
        var z = is_in_region3.select(1.0 + x, nan)
        result = is_in_region3.select(math.lgamma(z), result)

    return result


fn rgamma1pm1[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Computes `1 / gamma(1 + x) - 1` in a numerically stable way.

    This function is semantically equivalent to `1 / gamma(1 + x) - 1`, but it
    is more accurate for `x` close to zero or one.

    Parameters:
        dtype: The data type of the input and output SIMD vectors.
        simd_width: The width of the input and output SIMD vectors.

    Args:
        x: SIMD vector of floating-point values.

    Returns:
        SIMD vector containing the expression `1 / gamma(1 + x) - 1` evaluated
        at `x`.

    Constraints:
        The data type must be a floating-point of single (`float32`) or double
        (`float64`) precision.
    """
    asserting.assert_float_dtype["dtype", dtype]()

    alias nan: SIMD[dtype, simd_width] = math.nan[dtype]()
    var result: SIMD[dtype, simd_width] = nan

    # Regions of computation.
    var is_in_region1 = (x == 0.0) | (x == 1.0)
    var is_in_region2 = (x >= -0.5) & (x < 0.0)
    var is_in_region3 = (x > 0.0) & (x <= 0.5)
    var is_in_region4 = (x > 0.5) & (x < 1.0)
    var is_in_region5 = (x > 1.0) & (x <= 1.5)
    var is_in_region6 = (x < -0.5) | (x > 1.5)

    # Polynomials for regions 2 and 4. The coefficients for the Padé approximation
    # were obtained using the Python library `mpmath`.
    alias p = Polynomial[10, dtype, simd_width].from_coefficients[
        -4.22784335098467139393487909918e-1,
        -8.76243643973193958120640666347e-1,
        -4.59653437436261810715536535224e-1,
        1.253267646667917761310767750400e-2,
        1.272374059074339062508590520139e-2,
        -5.95722659095617453307017824897e-3,
        3.070451110948726727765078685413e-4,
        2.297364646087461210880646489337e-4,
        -2.92867644133341610115726150281e-5,
        2.184035804013220749396991885951e-6,
    ]()
    alias q = Polynomial[10, dtype, simd_width].from_coefficients[
        1.000000000000000000000000000000e-0,
        5.212245444278738169713276344712e-1,
        1.792664653862777325772960453280e-1,
        3.438203782078653915730104663393e-2,
        4.263280285377850240586205099463e-3,
        -1.05916724442728169202533807166e-4,
        -1.22076484585335162669853056235e-4,
        -3.17838959035037282903233187238e-5,
        -3.65081153946647239430275797323e-6,
        -3.52809435523569771837732680697e-7,
    ]()

    # Polynomials for regions 3 and 5. The coefficients for the Padé approximation
    # were obtained using the Python library `mpmath`.
    alias r = Polynomial[10, dtype, simd_width].from_coefficients[
        5.772156649015328606065120900824e-1,
        -3.55019099545320141149313031876e-1,
        -2.80386972049984078138240489896e-1,
        4.691471428746571677040872413793e-2,
        1.698702087612124086567211030085e-2,
        -6.06314331539890270227271205613e-3,
        1.849686265095375101066548123063e-4,
        1.979525687052423927977413302098e-4,
        -3.29375759528006334058753730013e-5,
        1.831226368489650977559259205254e-6,
    ]()
    alias s = q

    result = is_in_region1.select(0.0, result)

    var t = math.select(is_in_region2 | is_in_region3, x, x - 1.0)

    if (is_in_region2 | is_in_region4).reduce_or():
        var y = p(t) / q(t)
        result = math.select(
            is_in_region2,
            x * (y + 1.0),
            math.select(is_in_region4, (t / x) * y, result),
        )

    if (is_in_region3 | is_in_region5).reduce_or():
        var z = r(t) / s(t)
        result = math.select(
            is_in_region3,
            x * z,
            math.select(is_in_region5, (t / x) * (z - 1.0), result),
        )

    if is_in_region6.reduce_or():
        result = is_in_region6.select(
            math.reciprocal(math.tgamma(1.0 + x)) - 1.0, result
        )

    return result
