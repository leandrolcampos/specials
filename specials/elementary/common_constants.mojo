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

"""Common constants for elementary function implementations."""

from specials._internal.table import FloatTable


# ===----------------------- Exponential Functions ------------------------=== #
#
# References:
#
# Tang, P. T. P. (1989). Table-driven implementation of the exponential function
#   in IEEE floating-point arithmetic.
# ACM Transactions on Mathematical Software (TOMS), 15(2), 144-157.
# https://doi.org/10.1145/63522.214389


@always_inline
fn _get_exp_lead_table[dtype: DType]() -> FloatTable[32, dtype]:
    """Returns the table entries of `exp_lead` for single or double precision."""

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
fn _get_exp_trail_table[dtype: DType]() -> FloatTable[32, dtype]:
    """Returns the table entries of `exp_trail` for single or double precision."""

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


@register_passable("trivial")
struct ExpTable[dtype: DType]:
    """Table entries of `exp_lead` and `exp_trail` for single or double precision."""

    alias lead = _get_exp_lead_table[dtype]()
    alias trail = _get_exp_trail_table[dtype]()
