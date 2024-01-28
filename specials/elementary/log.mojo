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

"""Natural logarithmic function."""

import math

from specials._internal import asserting
from specials._internal.limits import FloatLimits
from specials._internal.polynomial import Polynomial


fn log[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Computes the natural logarithm of `x`.

    Parameters:
        dtype: The data type of the input and output SIMD vectors.
        simd_width: The width of the input and output SIMD vectors.

    Args:
        x: SIMD vector of floating-point values.

    Returns:
        SIMD vector containing the natural logarithm of `x`.

    Constraints:
        The data type must be a floating-point of single (`float32`) or double
        (`float64`) precision.
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
