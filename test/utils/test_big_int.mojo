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
# RUN: %mojo %build_dir %assertion_flag %debug_level %sanitize_checks %s

"""Tests for the `BigInt` struct."""

from builtin._location import __call_location
from testing.testing import _assert_equal_error, assert_equal

from specials.utils.big_int import BigInt, BigUInt
from test_utils import UnitTest


alias DEST_TYPE = DType.int64
alias DEFAULT_WIDTH = 1


@always_inline
fn _assert_equal[
    width: Int = DEFAULT_WIDTH
](
    lhs: BigInt[_, width=width, signed=_, word_type=_],
    rhs: SIMD[DEST_TYPE, width],
) raises:
    var casted_lhs = lhs.cast[DEST_TYPE]()

    if any(casted_lhs != rhs):
        raise _assert_equal_error(
            str(casted_lhs), str(rhs), "", __call_location()
        )


fn test_init() raises:
    _assert_equal(BigInt[8, width=1](), 0)
    _assert_equal(BigInt[24, width=1](), 0)
    _assert_equal(BigInt[8, width=4](), 0)
    _assert_equal(BigInt[24, width=4](), 0)

    _assert_equal(BigUInt[8, width=1](), 0)
    _assert_equal(BigUInt[24, width=1](), 0)
    _assert_equal(BigUInt[8, width=4](), 0)
    _assert_equal(BigUInt[24, width=4](), 0)


fn test_init_unsafe() raises:
    _ = BigInt[8, width=1](unsafe_uninitialized=True)
    _ = BigInt[24, width=1](unsafe_uninitialized=True)

    _ = BigUInt[8, width=1](unsafe_uninitialized=True)
    _ = BigUInt[24, width=1](unsafe_uninitialized=True)


fn test_init_from_int() raises:
    _assert_equal(BigInt[8, width=1](1), 1)
    _assert_equal(BigInt[24, width=1](1), 1)

    _assert_equal(BigUInt[8, width=1](-1), 255)
    _assert_equal(BigUInt[24, width=1](-1), 16_777_215)


fn test_init_from_signed_simd() raises:
    alias WIDTH = 4
    alias VALUE = SIMD[DEST_TYPE, WIDTH](-2, -1, 1, 2)

    _assert_equal(BigInt[8](VALUE), VALUE)
    _assert_equal(BigInt[24](VALUE), VALUE)

    _assert_equal(
        BigUInt[8](VALUE),
        SIMD[DEST_TYPE, WIDTH](254, 255, 1, 2),
    )
    _assert_equal(
        BigUInt[24](VALUE),
        SIMD[DEST_TYPE, WIDTH](16_777_214, 16_777_215, 1, 2),
    )


fn test_init_from_unsigned_simd() raises:
    alias WIDTH = 4
    alias VALUE = SIMD[DType.uint64, WIDTH](1, 2, 255, 16_777_215)

    _assert_equal(
        BigInt[8](VALUE),
        SIMD[DEST_TYPE, WIDTH](1, 2, -1, -1),
    )
    _assert_equal(
        BigInt[24](VALUE),
        SIMD[DEST_TYPE, WIDTH](1, 2, 255, -1),
    )

    _assert_equal(
        BigUInt[8](VALUE),
        SIMD[DEST_TYPE, WIDTH](1, 2, 255, 255),
    )
    _assert_equal(
        BigUInt[24](VALUE),
        SIMD[DEST_TYPE, WIDTH](1, 2, 255, 16_777_215),
    )


fn test_explicit_copy() raises:
    alias BITS = List[Int](8, 24)
    alias SIGNED = List[Bool](True, False)

    @parameter
    for i in range(len(BITS)):
        alias bits = BITS[i]

        @parameter
        for j in range(len(SIGNED)):
            alias signed = SIGNED[j]

            var existing = BigInt[bits, width=2, signed=signed](1)
            var copy = BigInt(existing)

            assert_equal(existing.bits, copy.bits)
            assert_equal(existing.width, copy.width)
            assert_equal(existing.signed, copy.signed)
            assert_equal(existing.word_type, existing.word_type)

            _assert_equal(existing, 1)
            _assert_equal(copy, 1)

            copy += 1
            _assert_equal(existing, 1)
            _assert_equal(copy, 2)


fn test_all_ones() raises:
    _assert_equal(BigInt[8, width=1].all_ones(), -1)
    _assert_equal(BigInt[24, width=1].all_ones(), -1)

    _assert_equal(BigUInt[8, width=1].all_ones(), 255)
    _assert_equal(BigUInt[24, width=1].all_ones(), 16_777_215)


fn test_min() raises:
    _assert_equal(BigInt[8, width=1].min(), -128)
    _assert_equal(BigInt[24, width=1].min(), -8_388_608)

    _assert_equal(BigUInt[8, width=1].min(), 0)
    _assert_equal(BigUInt[24, width=1].min(), 0)


fn test_max() raises:
    _assert_equal(BigInt[8, width=1].max(), 127)
    _assert_equal(BigInt[24, width=1].max(), 8_388_607)

    _assert_equal(BigUInt[8, width=1].max(), 255)
    _assert_equal(BigUInt[24, width=1].max(), 16_777_215)


fn test_one() raises:
    _assert_equal(BigInt[8, width=1].one(), 1)
    _assert_equal(BigInt[24, width=1].one(), 1)

    _assert_equal(BigUInt[8, width=1].one(), 1)
    _assert_equal(BigUInt[24, width=1].one(), 1)


fn test_zero() raises:
    _assert_equal(BigInt[8, width=1].zero(), 0)
    _assert_equal(BigInt[24, width=1].zero(), 0)

    _assert_equal(BigUInt[8, width=1].zero(), 0)
    _assert_equal(BigUInt[24, width=1].zero(), 0)


fn test_add() raises:
    _assert_equal(BigInt[8, width=1](1) + BigInt[8, width=1](-2), -1)
    _assert_equal(BigInt[24, width=1](1) + BigInt[24, width=1](-2), -1)

    _assert_equal(BigUInt[8, width=1](15) + BigUInt[8, width=1](16), 31)
    _assert_equal(BigUInt[24, width=1](255) + BigUInt[24, width=1](1), 256)

    _assert_equal(
        BigUInt[24](SIMD[DEST_TYPE, 2](1, 2)) + 2,
        SIMD[DEST_TYPE, 2](3, 4),
    )


fn test_iadd() raises:
    var sval = BigInt[8, width=1](1)
    var uval = BigUInt[24](SIMD[DEST_TYPE, 2](0, 255))

    sval += BigInt[8, width=1](-2)
    _assert_equal(sval, -1)

    uval += 1
    _assert_equal(uval, SIMD[DEST_TYPE, 2](1, 256))


fn test_add_with_overflow() raises:
    var sval8 = BigInt[8](SIMD[DEST_TYPE, 2](-128, 127))
    var uval8 = BigUInt[8](SIMD[DEST_TYPE, 2](0, 255))

    assert_equal(sval8.add_with_overflow(1), SIMD[DType.bool, 2](False, True))
    _assert_equal(sval8, SIMD[DEST_TYPE, 2](-127, -128))

    assert_equal(sval8.add_with_overflow(-1), SIMD[DType.bool, 2](False, True))
    _assert_equal(sval8, SIMD[DEST_TYPE, 2](-128, 127))

    assert_equal(uval8.add_with_overflow(1), SIMD[DType.bool, 2](False, True))
    _assert_equal(uval8, SIMD[DEST_TYPE, 2](1, 0))

    var sval24 = BigInt[24](SIMD[DEST_TYPE, 2](-8_388_608, 8_388_607))
    var uval24 = BigUInt[24](SIMD[DEST_TYPE, 2](0, 16_777_215))

    assert_equal(sval24.add_with_overflow(1), SIMD[DType.bool, 2](False, True))
    _assert_equal(sval24, SIMD[DEST_TYPE, 2](-8_388_607, -8_388_608))

    assert_equal(sval24.add_with_overflow(-1), SIMD[DType.bool, 2](False, True))
    _assert_equal(sval24, SIMD[DEST_TYPE, 2](-8_388_608, 8_388_607))

    assert_equal(uval24.add_with_overflow(1), SIMD[DType.bool, 2](False, True))
    _assert_equal(uval24, SIMD[DEST_TYPE, 2](1, 0))


fn test_sub() raises:
    _assert_equal(BigInt[8, width=1](1) - BigInt[8, width=1](-2), 3)
    _assert_equal(BigInt[24](Int32(383)) - BigInt[24](Int32(-511)), 894)

    _assert_equal(BigUInt[8, width=1](16) - BigUInt[8, width=1](15), 1)
    _assert_equal(BigUInt[24, width=1](383) - BigUInt[24, width=1](255), 128)

    _assert_equal(
        BigInt[24, width=4](SIMD[DEST_TYPE, 4](1, 2, 3, 4)) - 2,
        SIMD[DEST_TYPE, 4](-1, 0, 1, 2),
    )


fn test_isub() raises:
    var sval = BigInt[8, width=1](1)
    var uval = BigUInt[24](SIMD[DEST_TYPE, 2](1, 256))

    sval -= BigInt[8, width=1](-2)
    _assert_equal(sval, 3)

    uval -= 1
    _assert_equal(uval, SIMD[DEST_TYPE, 2](0, 255))


fn test_sub_with_overflow() raises:
    var sval8 = BigInt[8](SIMD[DEST_TYPE, 2](-128, 127))
    var uval8 = BigUInt[8](SIMD[DEST_TYPE, 2](0, 255))

    assert_equal(sval8.sub_with_overflow(1), SIMD[DType.bool, 2](True, False))
    _assert_equal(sval8, SIMD[DEST_TYPE, 2](127, 126))

    assert_equal(sval8.sub_with_overflow(-1), SIMD[DType.bool, 2](True, False))
    _assert_equal(sval8, SIMD[DEST_TYPE, 2](-128, 127))

    assert_equal(uval8.sub_with_overflow(1), SIMD[DType.bool, 2](True, False))
    _assert_equal(uval8, SIMD[DEST_TYPE, 2](255, 254))

    var sval24 = BigInt[24](SIMD[DEST_TYPE, 2](-8_388_608, 8_388_607))
    var uval24 = BigUInt[24](SIMD[DEST_TYPE, 2](0, 16_777_215))

    assert_equal(sval24.sub_with_overflow(1), SIMD[DType.bool, 2](True, False))
    _assert_equal(sval24, SIMD[DEST_TYPE, 2](8_388_607, 8_388_606))

    assert_equal(sval24.sub_with_overflow(-1), SIMD[DType.bool, 2](True, False))
    _assert_equal(sval24, SIMD[DEST_TYPE, 2](-8_388_608, 8_388_607))

    assert_equal(uval24.sub_with_overflow(1), SIMD[DType.bool, 2](True, False))
    _assert_equal(uval24, SIMD[DEST_TYPE, 2](16_777_215, 16_777_214))


fn test_comparison() raises:
    @always_inline
    fn _test_cmp[bits: Int, signed: Bool]() raises:
        alias WIDTH = 4

        var val1: BigInt[bits, width=WIDTH, signed=signed]
        var val2: __type_of(val1)

        @parameter
        if signed:
            val1 = SIMD[DEST_TYPE, WIDTH](-1, 0, 1, 2)
            val2 = SIMD[DEST_TYPE, WIDTH](-2, 0, 2, -1)
        else:
            val1 = SIMD[DEST_TYPE, WIDTH](1, 0, 1, 2)
            val2 = SIMD[DEST_TYPE, WIDTH](0, 0, 2, 1)

        assert_equal(
            val1 == val2, SIMD[DType.bool, WIDTH](False, True, False, False)
        )
        assert_equal(
            val1 != val2, SIMD[DType.bool, WIDTH](True, False, True, True)
        )
        assert_equal(
            val1 < val2, SIMD[DType.bool, WIDTH](False, False, True, False)
        )
        assert_equal(
            val1 <= val2, SIMD[DType.bool, WIDTH](False, True, True, False)
        )
        assert_equal(
            val1 > val2, SIMD[DType.bool, WIDTH](True, False, False, True)
        )
        assert_equal(
            val1 >= val2, SIMD[DType.bool, WIDTH](True, True, False, True)
        )

    _test_cmp[8, signed=True]()
    _test_cmp[8, signed=False]()

    _test_cmp[24, signed=True]()
    _test_cmp[24, signed=False]()


fn test_invert() raises:
    _assert_equal(~BigInt[8, width=1](1), -2)
    _assert_equal(~BigInt[24, width=1](1), -2)

    _assert_equal(~BigUInt[8, width=1](1), 254)
    _assert_equal(~BigUInt[24, width=1](1), 16_777_214)

    _assert_equal(~BigInt[8, width=1](127), -128)
    _assert_equal(~BigInt[24, width=1](8_388_607), -8_388_608)

    _assert_equal(~BigUInt[8, width=1](127), 128)
    _assert_equal(~BigUInt[24, width=1](8_388_607), 8_388_608)


fn test_lshift() raises:
    var sval8 = BigInt[8, width=4, word_type = DType.uint8](-99)
    var uval8 = BigUInt[8, width=4, word_type = DType.uint8](157)
    var offset = SIMD[DType.uint8, 4](0, 1, 4, 7)

    _assert_equal(sval8 << 0, -99)
    _assert_equal(sval8 << offset, SIMD[DEST_TYPE, 4](-99, 58, -48, -128))

    _assert_equal(uval8 << 0, 157)
    _assert_equal(uval8 << offset, SIMD[DEST_TYPE, 4](157, 58, 208, 128))

    var sval24 = BigInt[24, width=4, word_type = DType.uint8](-3_962_546)
    var uval24 = BigUInt[24, width=4, word_type = DType.uint8](12_814_670)
    offset = SIMD[uval24.word_type, 4](0, 1, 12, 23)

    _assert_equal(sval24 << 0, -3_962_546)
    _assert_equal(
        sval24 << offset,
        SIMD[DEST_TYPE, 4](-3_962_546, -7_925_092, -7_020_544, 0),
    )

    _assert_equal(uval24 << 0, 12_814_670)
    _assert_equal(
        uval24 << offset,
        SIMD[DEST_TYPE, 4](12_814_670, 8_852_124, 9_756_672, 0),
    )


fn test_ilshift() raises:
    var val = BigInt[24, width=1](1)
    val <<= 8
    _assert_equal(val, 256)


fn test_rshift() raises:
    var sval8 = BigInt[8, word_type = DType.uint8](
        SIMD[DType.int8, 4](1, 99, -99, -1)
    )
    var uval8 = BigUInt[8, width=4, word_type = DType.uint8](157)
    var offset = SIMD[DType.uint8, 4](0, 1, 4, 7)

    _assert_equal(sval8 >> 0, SIMD[DEST_TYPE, 4](1, 99, -99, -1))
    _assert_equal(sval8 >> offset, SIMD[DEST_TYPE, 4](1, 49, -7, -1))

    _assert_equal(uval8 >> 0, 157)
    _assert_equal(uval8 >> offset, SIMD[DEST_TYPE, 4](157, 78, 9, 1))

    var sval24 = BigInt[24, width=4, word_type = DType.uint8](-3_962_546)
    var uval24 = BigUInt[24, width=4, word_type = DType.uint8](12_814_670)
    offset = SIMD[uval24.word_type, 4](0, 1, 12, 23)

    _assert_equal(sval24 >> 0, -3_962_546)
    _assert_equal(
        sval24 >> offset,
        SIMD[DEST_TYPE, 4](-3_962_546, -1_981_273, -968, -1),
    )

    _assert_equal(uval24 >> 0, 12_814_670)
    _assert_equal(
        uval24 >> offset,
        SIMD[DEST_TYPE, 4](12_814_670, 6_407_335, 3_128, 1),
    )


fn test_irshift() raises:
    var val = BigInt[24, width=1].min()
    val >>= 16
    _assert_equal(val, -128)


fn test_cast_to_signed_big_int() raises:
    var sval8 = BigInt[8, width=1](-1)
    var uval8 = BigUInt[8, width=1](255)

    _assert_equal(sval8.cast[8, signed=True](), -1)
    _assert_equal(sval8.cast[24, signed=True](), -1)
    _assert_equal(uval8.cast[8, signed=True](), -1)
    _assert_equal(uval8.cast[24, signed=True](), 255)

    var sval24 = BigInt[24, width=1](-1)
    var uval24 = BigUInt[24, width=1](16_777_215)

    _assert_equal(sval24.cast[8, signed=True](), -1)
    _assert_equal(sval24.cast[24, signed=True](), -1)
    _assert_equal(uval24.cast[8, signed=True](), -1)
    _assert_equal(uval24.cast[24, signed=True](), -1)


fn test_cast_to_unsigned_big_int() raises:
    var sval8 = BigInt[8, width=1](-1)
    var uval8 = BigUInt[8, width=1](255)

    _assert_equal(sval8.cast[8, signed=False](), 255)
    _assert_equal(sval8.cast[24, signed=False](), 255)
    _assert_equal(uval8.cast[8, signed=False](), 255)
    _assert_equal(uval8.cast[24, signed=False](), 255)

    var sval24 = BigInt[24, width=1](-1)
    var uval24 = BigUInt[24, width=1](16_777_215)

    _assert_equal(sval24.cast[8, signed=False](), 255)
    _assert_equal(sval24.cast[24, signed=False](), 16_777_215)
    _assert_equal(uval24.cast[8, signed=False](), 255)
    _assert_equal(uval24.cast[24, signed=False](), 16_777_215)


fn test_cast_to_simd() raises:
    alias DTYPES = List[DType](
        DType.int8,
        DType.uint8,
        DType.int16,
        DType.uint16,
        DType.int32,
        DType.uint32,
        DType.int64,
        DType.uint64,
    )
    var pos_num = Int64(876925160929511475)
    var neg_num = Int64(-876925160929511475)

    var pos_sval = BigInt[64](pos_num)
    var neg_sval = BigInt[64](neg_num)
    var uval = BigUInt[64](pos_num)

    @parameter
    for i in range(len(DTYPES)):
        alias DTYPE = DTYPES[i]

        var casted_pos_num = pos_num.cast[DTYPE]()
        var casted_neg_num = neg_num.cast[DTYPE]()

        assert_equal(pos_sval.cast[DTYPE](), casted_pos_num)
        assert_equal(neg_sval.cast[DTYPE](), casted_neg_num)
        assert_equal(uval.cast[DTYPE](), casted_pos_num)


fn test_most_significant_digit() raises:
    var sval = BigInt[24, width=1].max()
    var uval = BigUInt[24, width=1].max()

    _assert_equal(sval, 8_388_607)
    _assert_equal(uval, 16_777_215)

    assert_equal(sval.get_most_significant_bit(), False)
    assert_equal(uval.get_most_significant_bit(), True)

    sval.set_most_significant_bit()
    uval.clear_most_significant_bit()

    assert_equal(sval.get_most_significant_bit(), True)
    assert_equal(uval.get_most_significant_bit(), False)

    _assert_equal(sval, -1)
    _assert_equal(uval, 8_388_607)

    sval.clear_most_significant_bit()
    uval.set_most_significant_bit()

    _assert_equal(sval, 8_388_607)
    _assert_equal(uval, 16_777_215)

    assert_equal(sval.get_most_significant_bit(), False)
    assert_equal(uval.get_most_significant_bit(), True)


fn test_count_leading_zeros() raises:
    var uval8 = BigUInt[8, width=1](0)

    assert_equal(uval8.count_leading_zeros(), 8)

    uval8.set_most_significant_bit()
    assert_equal(uval8.count_leading_zeros(), 0)

    uval8 >>= 1
    assert_equal(uval8.count_leading_zeros(), 1)

    var uval24 = BigUInt[24, width=1](0)

    assert_equal(uval24.count_leading_zeros(), 24)

    uval24.set_most_significant_bit()
    assert_equal(uval24.count_leading_zeros(), 0)

    uval24 >>= 12
    assert_equal(uval24.count_leading_zeros(), 12)


fn test_is_negative() raises:
    var val = SIMD[DType.int8, 2](-1, 0)

    assert_equal(BigInt[8](val).is_negative(), val < 0)
    assert_equal(BigUInt[8](val).is_negative(), False)

    assert_equal(BigInt[24](val).is_negative(), val < 0)
    assert_equal(BigUInt[24](val).is_negative(), False)


fn test_is_zero() raises:
    var val = SIMD[DType.int8, 4](-1, 0, 1, 2)
    var expected = val == 0

    assert_equal(BigInt[8](val).is_zero(), expected)
    assert_equal(BigUInt[8](val).is_zero(), expected)

    assert_equal(BigInt[24](val).is_zero(), expected)
    assert_equal(BigUInt[24](val).is_zero(), expected)


fn main() raises:
    test_init()
    test_init_unsafe()
    test_init_from_int()
    test_init_from_signed_simd()
    test_init_from_unsigned_simd()
    test_explicit_copy()

    test_all_ones()
    test_min()
    test_max()
    test_one()
    test_zero()

    test_add()
    test_iadd()
    test_add_with_overflow()

    test_sub()
    test_isub()
    test_sub_with_overflow()

    test_comparison()

    test_invert()

    test_lshift()
    test_ilshift()

    test_rshift()
    test_irshift()

    test_cast_to_signed_big_int()
    test_cast_to_unsigned_big_int()
    test_cast_to_simd()

    test_most_significant_digit()

    test_count_leading_zeros()

    test_is_negative()

    test_is_zero()
