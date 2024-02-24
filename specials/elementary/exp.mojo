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

from memory.unsafe import bitcast

from specials._internal.asserting import assert_float_dtype
from specials._internal.limits import FloatLimits
from specials._internal.polynomial import Polynomial
from specials._internal.table import FloatTable


@always_inline
fn _get_s_lead_table[dtype: DType]() -> FloatTable[32, dtype]:
    """Returns the table entries of `s_lead` for single or double precision."""

    @parameter
    if dtype == DType.float32:
        return FloatTable[32, dtype].from_hexadecimal_values[
            0x3F80_0000,
            0x3F82_CD80,
            0x3F85_AAC0,
            0x3F88_9800,
            0x3F8B_95C0,
            0x3F8E_A400,
            0x3F91_C3C0,
            0x3F94_F4C0,
            0x3F98_37C0,
            0x3F9B_8D00,
            0x3F9E_F500,
            0x3FA2_7040,
            0x3FA5_FEC0,
            0x3FA9_A140,
            0x3FAD_5800,
            0x3FB1_23C0,
            0x3FB5_04C0,
            0x3FB8_FB80,
            0x3FBD_0880,
            0x3FC1_2C40,
            0x3FC5_6700,
            0x3FC9_B980,
            0x3FCE_2480,
            0x3FD2_A800,
            0x3FD7_44C0,
            0x3FDB_FB80,
            0x3FE0_CCC0,
            0x3FE5_B900,
            0x3FEA_C0C0,
            0x3FEF_E480,
            0x3FF5_2540,
            0x3FFA_8380,
        ]()
    else:  # dtype == DType.float64
        return FloatTable[32, dtype].from_hexadecimal_values[
            0x3FF00000_00000000,
            0x3FF059B0_D3158540,
            0x3FF0B558_6CF98900,
            0x3FF11301_D0125B40,
            0x3FF172B8_3C7D5140,
            0x3FF1D487_3168B980,
            0x3FF2387A_6E756200,
            0x3FF29E9D_F51FDEC0,
            0x3FF306FE_0A31B700,
            0x3FF371A7_373AA9C0,
            0x3FF3DEA6_4C123400,
            0x3FF44E08_60618900,
            0x3FF4BFDA_D5362A00,
            0x3FF5342B_569D4F80,
            0x3FF5AB07_DD485400,
            0x3FF6247E_B03A5580,
            0x3FF6A09E_667F3BC0,
            0x3FF71F75_E8EC5F40,
            0x3FF7A114_73EB0180,
            0x3FF82589_994CCE00,
            0x3FF8ACE5_422AA0C0,
            0x3FF93737_B0CDC5C0,
            0x3FF9C491_82A3F080,
            0x3FFA5503_B23E2540,
            0x3FFAE89F_995AD380,
            0x3FFB7F76_F2FB5E40,
            0x3FFC199B_DD855280,
            0x3FFCB720_DCEF9040,
            0x3FFD5818_DCFBA480,
            0x3FFDFC97_337B9B40,
            0x3FFEA4AF_A2A490C0,
            0x3FFF5076_5B6E4540,
        ]()


@always_inline
fn _get_s_trail_table[dtype: DType]() -> FloatTable[32, dtype]:
    """Returns the table entries of `s_trail` for single or double precision."""

    @parameter
    if dtype == DType.float32:
        return FloatTable[32, dtype].from_hexadecimal_values[
            0x0000_0000,
            0x3553_1585,
            0x34D9_F312,
            0x35E8_092E,
            0x3471_F546,
            0x36E6_2D17,
            0x361B_9D59,
            0x36BE_A3FC,
            0x36C1_4637,
            0x36E6_E755,
            0x36C9_8247,
            0x34C0_C312,
            0x3635_4D8B,
            0x3655_A754,
            0x36FB_A90B,
            0x36D6_074B,
            0x36CC_CFE7,
            0x36BD_1D8C,
            0x368E_7D60,
            0x35CC_A667,
            0x36A8_4554,
            0x36F6_19B9,
            0x35C1_51F8,
            0x366C_8F89,
            0x36F3_2B5A,
            0x36DE_5F6C,
            0x3677_6155,
            0x355C_EF90,
            0x355C_FBA5,
            0x36E6_6F73,
            0x36F4_5492,
            0x36CB_6DC9,
        ]()
    else:  # dtype == DType.float64
        return FloatTable[32, dtype].from_hexadecimal_values[
            0x00000000_00000000,
            0x3D0A1D73_E2A475B4,
            0x3CEEC531_7256E308,
            0x3CF0A4EB_BF1AED93,
            0x3D0D6E6F_BE462876,
            0x3D053C02_DC0144C8,
            0x3D0C3360_FD6D8E0B,
            0x3D009612_E8AFAD12,
            0x3CF52DE8_D5A46306,
            0x3CE54E28_AA05E8A9,
            0x3D011ADA_0911F09F,
            0x3D068189_B7A04EF8,
            0x3D038EA1_CBD7F621,
            0x3CBDF0A8_3C49D86A,
            0x3D04AC64_980A8C8F,
            0x3CD2C7C3_E81BF4B7,
            0x3CE92116_5F626CDD,
            0x3D09EE91_B8797785,
            0x3CDB5F54_408FDB37,
            0x3CF28ACF_88AFAB35,
            0x3CFB5BA7_C55A192D,
            0x3D027A28_0E1F92A0,
            0x3CF01C7C_46B071F3,
            0x3CFC8B42_4491CAF8,
            0x3D06AF43_9A68BB99,
            0x3CDBAA9E_C206AD4F,
            0x3CFC2220_CB12A092,
            0x3D048A81_E5E8F4A5,
            0x3CDC9768_16BAD9B8,
            0x3CFEB968_CAC39ED3,
            0x3CF9858F_73A18F5E,
            0x3C99D3E1_2DD8A18B,
        ]()


@always_inline
fn _exp_impl[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width], cond: SIMD[DType.bool, simd_width]) -> SIMD[
    dtype, simd_width
]:
    """Implements the exponential function as specified in the reference paper."""
    alias max_exponent: SIMD[DType.int32, simd_width] = FloatLimits[dtype].maxexp - 1
    alias s_lead_table = _get_s_lead_table[dtype]()
    alias s_trail_table = _get_s_trail_table[dtype]()

    let safe_x = cond.select(x, 1.0)

    let index: SIMD[DType.int32, simd_width]
    let exponent: SIMD[DType.int32, simd_width]
    let expm1_r: SIMD[dtype, simd_width]

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

        let xn = math.round(safe_x * inv_ln2_over_32)
        let xn2 = math.mod(xn, 32.0)
        let xn1 = xn - xn2

        let xn_is_large = (math.abs(xn) >= 512)
        var x_reduced_lead = math.fma(
            -xn_is_large.select(xn1, xn), ln2_over_32_lead, safe_x
        )
        x_reduced_lead = xn_is_large.select(
            math.fma(-xn2, ln2_over_32_lead, x_reduced_lead), x_reduced_lead
        )
        let x_reduced_trail = -xn * ln2_over_32_trail

        index = xn2.cast[DType.int32]()
        exponent = xn1.cast[DType.int32]() / 32

        let x_reduced = x_reduced_lead + x_reduced_trail

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

        let xn = math.round(safe_x * inv_ln2_over_32)
        let xn2 = math.mod(xn, 32.0)
        let xn1 = xn - xn2

        let xn_is_large = (math.abs(xn) >= 512)
        var x_reduced_lead = math.fma(
            -xn_is_large.select(xn1, xn), ln2_over_32_lead, safe_x
        )
        x_reduced_lead = xn_is_large.select(
            math.fma(-xn2, ln2_over_32_lead, x_reduced_lead), x_reduced_lead
        )
        let x_reduced_trail = -xn * ln2_over_32_trail

        index = xn2.cast[DType.int32]()
        exponent = xn1.cast[DType.int32]() / 32

        let x_reduced = x_reduced_lead + x_reduced_trail

        expm1_r = x_reduced_lead + (
            math.fma(x_reduced * x_reduced, polynomial(x_reduced), x_reduced_trail)
        )

    let s_lead = s_lead_table.unsafe_lookup(index)
    let s_trail = s_trail_table.unsafe_lookup(index)
    let s = s_lead + s_trail

    let mantissa = s_lead + math.fma(s, expm1_r, s_trail)

    let result: SIMD[dtype, simd_width]

    if (exponent > max_exponent).reduce_or():
        let exponent_clipped = math.min(exponent, max_exponent)
        let exponent_remainder = exponent - exponent_clipped

        result = math.ldexp(math.ldexp(mantissa, exponent_clipped), exponent_remainder)
    else:
        result = math.ldexp(mantissa, exponent)

    return result


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
    let x_abs = math.abs(x)

    # Regions of computation
    let is_in_region1: SIMD[DType.bool, simd_width]  # abs(x) < xeps
    let is_in_region2: SIMD[DType.bool, simd_width]  # x > xmax
    let is_in_region3: SIMD[DType.bool, simd_width]  # x < -xmax
    let is_in_region4: SIMD[DType.bool, simd_width]  # xeps <= abs(x) <= xmax

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

        is_in_region1 = x_abs < xeps
        is_in_region2 = x > xmax
        is_in_region3 = x < -xmax
        is_in_region4 = (x_abs >= xeps) & (x_abs <= xmax)

    else:  # dtype == DType.float64
        alias xeps: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint64](
            0x3C900000_00000000,
        )
        # `xmax` is different from what is specified in the reference paper:
        # `alias xmax = log(FloatLimits[dtype].max)`
        alias xmax: SIMD[dtype, simd_width] = bitcast[dtype, DType.uint64](
            0x40862E42_FEFA39EF,
        )

        is_in_region1 = x_abs < xeps
        is_in_region2 = x > xmax
        is_in_region3 = x < -xmax
        is_in_region4 = (x_abs >= xeps) & (x_abs <= xmax)

    result = is_in_region1.select(1.0, result)
    result = is_in_region2.select(inf, result)
    result = is_in_region3.select(0.0, result)

    if is_in_region4.reduce_or():
        result = is_in_region4.select(_exp_impl(x, is_in_region4), result)

    return result
