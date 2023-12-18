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

"""Gamma-related functions."""

import math

from ._internal import asserting
from ._internal.functional import fori_loop
from ._internal.limits import FloatLimits
from ._internal.math import log
from .polynomial import Chebyshev, Polynomial


fn lgamma_correction[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Computes the correction term for the Rocktaeschel's approximation of `lgamma`.

    The correction term is defined as:

    `lgamma_correction(x) = lgamma(x) - (x - 0.5) * log(x) + x - 0.5 * log(2 * pi)`

    for `x >= 8`.

    Parameters:
        dtype: The data type of the input and output SIMD vectors (`float32` or
            `float64`).
        simd_width: The width of the input and output SIMD vectors.

    Args:
        x: SIMD vector of floating-point values greater than or equal to `8.0`.

    Returns:
        The correction term for the Rocktaeschel's approximation of `lgamma`. If `x` is
        less than `8.0`, this function returns `NaN`.

    Constraints:
        The data type must be a floating-point of single or double precision.
    """
    asserting.assert_float_dtype["dtype", dtype]()

    alias nan: SIMD[dtype, simd_width] = math.nan[dtype]()
    alias zero: SIMD[dtype, simd_width] = 0.0

    alias xmin: SIMD[dtype, simd_width] = 8.0
    alias xbig: SIMD[dtype, simd_width] = math.reciprocal(
        math.exp2(0.5 * math.log2(FloatLimits[dtype].epsneg))
    )
    alias xmax: SIMD[dtype, simd_width] = math.reciprocal(12.0 * FloatLimits[dtype].min)

    # The coefficients for the Chebyshev approximation of this correction were obtained
    # using the Python library `mpmath`.
    alias p = Chebyshev[15, dtype, simd_width].from_coefficients[
        8.331170390906488010133812318436e-2,
        -2.160555080544604128445388438061e-5,
        2.380513030666125633967836809341e-8,
        -6.796982741412553398477048390141e-11,
        3.598298746801337252645642260609e-13,
        -3.006641868307272999348413100017e-15,
        3.600735976941671612063202842825e-17,
        -5.791695393592681157038243797525e-19,
        1.193723234577627761785829083771e-20,
        -3.042270790514913036911869792252e-22,
        9.322841959630994708926417811402e-24,
        -3.358935720040657868013988608955e-25,
        1.396918777539642950417187988646e-26,
        -6.60403655947600591725440741065e-28,
        3.490467256080184812261246362399e-29,
    ]()
    alias error_tolerance = FloatLimits[dtype].epsneg
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
    """Computes the natural logarithm of the gamma function at `1 + x`.

    This function is semantically equivalent to `lgamma(1 + x)`, but it is more
    accurate for arguments close to zero.

    Parameters:
        dtype: The data type of the input and output SIMD vectors (`float32` or
            `float64`).
        simd_width: The width of the input and output SIMD vectors.

    Args:
        x: SIMD vector of floating-point values.

    Returns:
        The natural logarithm of the gamma function at `1 + x`.

    Constraints:
        The data type must be a floating-point of single or double precision.
    """
    asserting.assert_float_dtype["dtype", dtype]()

    alias nan: SIMD[dtype, simd_width] = math.nan[dtype]()
    var result: SIMD[dtype, simd_width] = nan

    # Regions of computation.
    let is_in_region1 = (x >= -0.2) & (x < 0.6)
    let is_in_region2 = (x >= 0.6) & (x <= 1.25)
    let is_in_region3 = ~math.isnan(x) & ~is_in_region1 & ~is_in_region2

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
        let y = (x - 0.5) - 0.5
        result = is_in_region2.select(y * (r(y) / s(y)), result)

    if is_in_region3.reduce_or():
        let z = is_in_region3.select(1.0 + x, nan)
        result = is_in_region3.select(math.lgamma(z), result)

    return result


fn lbeta[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width], y: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Computes the natural logarithm of the beta function.

    This function is semantically equivalent to `lgamma(x) + lgamma(y) - lgamma(x + y)`,
    but it is more accurate for arguments greater than or equal to `8.0`.

    Parameters:
        dtype: The data type of the input and output SIMD vectors (float32 or float64).
        simd_width: The width of the input and output SIMD vectors.

    Args:
        x: SIMD vector of non-negative floating-point values.
        y: SIMD vector of non-negative floating-point values.

    Returns:
        The natural logarithm of the beta function.

    Constraints:
        The data type must be a floating-point of single (float32) or double (float64)
        precision.
    """
    asserting.assert_float_dtype["dtype", dtype]()

    alias inf: SIMD[dtype, simd_width] = math.limit.inf[dtype]()
    alias nan: SIMD[dtype, simd_width] = math.nan[dtype]()
    alias log_sqrt_2pi: SIMD[dtype, simd_width] = 0.91893853320467274178032973640562

    # Ensure that `a` is the smaller of the two arguments and `b` is the larger one.
    # Although the Beta function is mathematically symmetric, this procedure is not.
    let a = math.min(x, y)
    let b = math.max(x, y)

    # The `math.lgamma`` operation is one of the most computationally expensive
    # operations in this procedure. To avoid calling it when possible, we mask out
    # large values of `a` and `b`.
    let a_small = math.select(a < 8.0, a, nan)
    let b_small = math.select(b < 8.0, b, nan)

    let lgamma_a_small = math.lgamma(a_small)
    let apb = a + b
    let a_over_apb = a / apb
    let log1p_neg_a_over_apb = math.log1p(-a_over_apb)

    # `a` and `b` are small: `a <= b < 8.0`.
    var result = lgamma_a_small + math.lgamma(b_small) - math.lgamma(a_small + b_small)

    # `a` is small, but `b` is large: `a < 8.0 <= b`.
    var correction = lgamma_correction(b) - lgamma_correction(apb)
    let result_for_large_b = (
        lgamma_a_small
        + correction
        + a
        - a * log(apb)
        + (b - 0.5) * log1p_neg_a_over_apb
    )
    result = math.select(b >= 8.0, result_for_large_b, result)

    # `a` and `b` are large: `8.0 <= a <= b`.
    correction += lgamma_correction(a)
    let result_for_large_a = (
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
