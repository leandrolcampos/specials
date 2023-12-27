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
# References:
#
# Cody, W. J. (1980). Software Manual for the Elementary Functions.
# Prentice-Hall, Inc.
# https://dl.acm.org/doi/10.5555/1096483

"""Elementary functions."""

import math

from ._internal import asserting
from ._internal.limits import FloatLimits
from .polynomial import Polynomial


fn exp[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Computes the exponential of `x`.

    Parameters:
        dtype: The data type of the input and output SIMD vectors (`float32` or
            `float64`).
        simd_width: The width of the input and output SIMD vectors.

    Args:
        x: SIMD vector of floating-point values.

    Returns:
        SIMD vector containing the exponential of `x`.

    Constraints:
        The data type must be a floating-point of single or double precision.
    """
    asserting.assert_float_dtype["dtype", dtype]()

    alias log2_e: SIMD[dtype, simd_width] = 1.44269504088896340735992468100189

    alias xmin: SIMD[dtype, simd_width] = log(FloatLimits[dtype].min)
    alias xmax: SIMD[dtype, simd_width] = log(FloatLimits[dtype].max)
    alias xeps: SIMD[dtype, simd_width] = 0.5 * FloatLimits[dtype].eps
    alias max_exponent: SIMD[DType.int32, simd_width] = FloatLimits[dtype].maxexp - 1

    var result: SIMD[dtype, simd_width] = math.nan[dtype]()

    # Regions of computation.
    let is_in_region1 = (x < xmin)
    let is_in_region2: SIMD[DType.bool, simd_width]

    @parameter
    if dtype == DType.float32:
        is_in_region2 = x >= xmax
    else:  # dtype == DType.float64
        is_in_region2 = x > xmax

    let is_in_region3 = (math.abs(x) < xeps)
    let is_in_region4 = (
        ~(is_in_region1 | is_in_region2 | is_in_region3) & math.limit.isfinite(x)
    )

    result = is_in_region1.select(0.0, result)
    result = is_in_region2.select(math.limit.inf[dtype](), result)
    result = is_in_region3.select(1.0, result)

    if is_in_region4.reduce_or():
        alias c1: SIMD[dtype, simd_width] = 0.693359375
        alias c2: SIMD[dtype, simd_width] = -2.1219444005469058277e-4

        let xn = math.floor(x * log2_e + 0.5)
        let g = (x - xn * c1) - xn * c2
        let g_squared = g * g
        let g_times_pval: SIMD[dtype, simd_width]
        let qval: SIMD[dtype, simd_width]

        @parameter
        if dtype == DType.float32:
            alias p = Polynomial[2, dtype, simd_width].from_coefficients[
                0.24999999950e00,
                0.41602886268e-2,
            ]()
            alias q = Polynomial[2, dtype, simd_width].from_coefficients[
                0.50000000000e00,
                0.49987178778e-1,
            ]()

            g_times_pval = g * p(g_squared)
            qval = q(g_squared)

        else:  # dtype == DType.float64
            alias p = Polynomial[3, dtype, simd_width].from_coefficients[
                0.249999999999999993e00,
                0.694360001511792852e-2,
                0.165203300268279130e-4,
            ]()
            alias q = Polynomial[3, dtype, simd_width].from_coefficients[
                0.500000000000000000e00,
                0.555538666969001188e-1,
                0.495862884905441294e-3,
            ]()

            g_times_pval = g * p(g_squared)
            qval = q(g_squared)

        let exp_g = 0.5 + g_times_pval / (qval - g_times_pval)
        let exponent = xn.cast[DType.int32]() + 1

        if (exponent > max_exponent).reduce_or():
            let exponent_clipped = math.min(exponent, max_exponent)
            let exponent_remainder = exponent - exponent_clipped

            result = is_in_region4.select(
                math.ldexp(math.ldexp(exp_g, exponent_clipped), exponent_remainder),
                result,
            )
        else:
            result = is_in_region4.select(math.ldexp(exp_g, exponent), result)

    return result


fn log[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Computes the natural logarithm of `x`.

    Parameters:
        dtype: The data type of the input and output SIMD vectors (`float32` or
            `float64`).
        simd_width: The width of the input and output SIMD vectors.

    Args:
        x: SIMD vector of floating-point values.

    Returns:
        SIMD vector containing the natural logarithm of `x`.

    Constraints:
        The data type must be a floating-point of single or double precision.
    """
    asserting.assert_float_dtype["dtype", dtype]()

    alias sqrt_half: SIMD[dtype, simd_width] = 0.70710678118654752440084436210485

    alias inf: SIMD[dtype, simd_width] = math.limit.inf[dtype]()
    alias xmin: SIMD[dtype, simd_width] = FloatLimits[dtype].min

    var result: SIMD[dtype, simd_width] = math.nan[dtype]()

    # Regions of computation.
    let is_in_region1 = (x == 0.0)
    let is_in_region2 = (x == inf)
    let is_in_region3 = (x > 0.0) & ~is_in_region2

    result = is_in_region1.select(-inf, result)
    result = is_in_region2.select(inf, result)

    if is_in_region3.reduce_or():
        let fraction_and_exponent = math.frexp(x)
        let fraction = fraction_and_exponent[0]
        let fraction_gt_sqrt_half = (fraction > sqrt_half)

        var exponent = fraction_and_exponent[1]
        exponent = fraction_gt_sqrt_half.select(exponent, exponent - 1)

        var znum = fraction - 0.5
        znum = fraction_gt_sqrt_half.select(znum - 0.5, znum)
        let zden = fraction_gt_sqrt_half.select(fraction, znum) * 0.5 + 0.5
        let z = znum / zden
        let z_squared = z * z

        let r: SIMD[dtype, simd_width]

        @parameter
        if dtype == DType.float32:
            alias a = Polynomial[1, dtype, simd_width].from_coefficients[
                -0.5527074855e00,
            ]()
            alias b = Polynomial[2, dtype, simd_width].from_coefficients[
                -0.6632718214e01,
                0.10000000000e01,
            ]()

            r = z + z * (z_squared * a(z_squared) / b(z_squared))

        else:  # dtype == DType.float64
            alias a = Polynomial[3, dtype, simd_width].from_coefficients[
                -0.64124943423745581147e02,
                0.163839435630215342220e02,
                -0.78956112887491257267e00,
            ]()
            alias b = Polynomial[4, dtype, simd_width].from_coefficients[
                -0.76949932108494879777e03,
                0.312032220919245328440e03,
                -0.35667977739034646171e02,
                0.100000000000000000000e01,
            ]()

            r = z + z * (z_squared * a(z_squared) / b(z_squared))

        alias c1: SIMD[dtype, simd_width] = 0.693359375
        alias c2: SIMD[dtype, simd_width] = -2.121944400546905827679e-4

        result = is_in_region3.select((exponent * c2 + r) + exponent * c1, result)

    return result
