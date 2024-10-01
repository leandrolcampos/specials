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
from testing.testing import _assert_equal_error, assert_equal, assert_true

from specials.utils.big_int import BigInt, BigUInt
from test_utils import UnitTest


alias DEST_TYPE = DType.int64
alias DEFAULT_WIDTH = 1


@always_inline
fn _assert_equal[
    size: Int = DEFAULT_WIDTH
](
    lhs: BigInt[_, size=size, signed=_, word_type=_],
    rhs: SIMD[DEST_TYPE, size],
) raises:
    var casted_lhs = lhs.cast[DEST_TYPE]()

    if any(casted_lhs != rhs):
        raise _assert_equal_error(
            str(casted_lhs), str(rhs), "", __call_location()
        )


fn test_init() raises:
    _assert_equal(BigInt[8, size=1](), 0)
    _assert_equal(BigInt[24, size=1](), 0)
    _assert_equal(BigInt[8, size=4](), 0)
    _assert_equal(BigInt[24, size=4](), 0)

    _assert_equal(BigUInt[8, size=1](), 0)
    _assert_equal(BigUInt[24, size=1](), 0)
    _assert_equal(BigUInt[8, size=4](), 0)
    _assert_equal(BigUInt[24, size=4](), 0)


fn test_init_unsafe() raises:
    _ = BigInt[8, size=1](unsafe_uninitialized=True)
    _ = BigInt[24, size=1](unsafe_uninitialized=True)

    _ = BigUInt[8, size=1](unsafe_uninitialized=True)
    _ = BigUInt[24, size=1](unsafe_uninitialized=True)


fn test_init_from_int_literal() raises:
    _assert_equal(BigInt[8, size=1](0), 0)
    _assert_equal(BigInt[8, size=1](127), 127)
    _assert_equal(BigInt[8, size=1](-1), -1)
    _assert_equal(BigInt[8, size=1](-128), -128)
    _assert_equal(BigInt[24, size=1](0), 0)
    _assert_equal(BigInt[24, size=1](8_388_607), 8_388_607)
    _assert_equal(BigInt[24, size=1](-1), -1)
    _assert_equal(BigInt[24, size=1](-8_388_608), -8_388_608)

    _assert_equal(BigUInt[8, size=1](0), 0)
    _assert_equal(BigUInt[8, size=1](255), 255)
    _assert_equal(BigUInt[24, size=1](0), 0)
    _assert_equal(BigUInt[24, size=1](16_777_215), 16_777_215)


fn test_init_from_int() raises:
    _assert_equal(BigInt[8, size=1](Int(1)), 1)
    _assert_equal(BigInt[24, size=1](Int(1)), 1)

    _assert_equal(BigUInt[8, size=1](Int(255)), 255)
    _assert_equal(BigUInt[24, size=1](Int(16_777_215)), 16_777_215)


fn test_init_from_signed_simd() raises:
    alias VALUE = SIMD[DEST_TYPE, 4](-128, -1, 1, 127)
    alias POS_VALUE = SIMD[DEST_TYPE, 2](1, 127)

    _assert_equal(BigInt[8](VALUE), VALUE)
    _assert_equal(BigInt[24](VALUE), VALUE)

    _assert_equal(BigUInt[8](POS_VALUE), POS_VALUE)
    _assert_equal(BigUInt[24](POS_VALUE), POS_VALUE)


fn test_init_from_unsigned_simd() raises:
    alias VALUE = SIMD[DEST_TYPE, 4](1, 127, 128, 255)
    alias SMALL_VALUE = SIMD[DEST_TYPE, 2](1, 127)

    _assert_equal(BigInt[8](SMALL_VALUE), SMALL_VALUE)
    _assert_equal(BigInt[24](VALUE), VALUE)

    _assert_equal(BigUInt[8](VALUE), VALUE)
    _assert_equal(BigUInt[24](VALUE), VALUE)


fn test_explicit_copy() raises:
    alias BITS = List[Int](8, 24)
    alias SIGNED = List[Bool](True, False)

    @parameter
    for i in range(len(BITS)):
        alias bits = BITS[i]

        @parameter
        for j in range(len(SIGNED)):
            alias signed = SIGNED[j]

            var existing = BigInt[bits, size=2, signed=signed](1)
            var copy = BigInt(existing)

            assert_equal(existing.bits, copy.bits)
            assert_equal(existing.size, copy.size)
            assert_equal(existing.signed, copy.signed)
            assert_equal(existing.word_type, existing.word_type)

            _assert_equal(existing, 1)
            _assert_equal(copy, 1)

            copy += 1
            _assert_equal(existing, 1)
            _assert_equal(copy, 2)


fn test_from_signed_big_int() raises:
    alias _BigInt = BigInt[_, size=1, signed=_, word_type = DType.uint8]
    alias _BigUInt = BigUInt[_, size=1, word_type = DType.uint8]

    var sval8 = _BigInt[8](-1)
    var sval24 = _BigInt[24](127)

    _assert_equal(_BigInt[8](other=sval24), 127)
    _assert_equal(_BigUInt[8](other=sval24), 127)

    _assert_equal(_BigInt[24](other=sval8), -1)
    _assert_equal(_BigUInt[24](other=sval24), 127)


fn test_from_unsigned_big_int() raises:
    alias _BigInt = BigInt[_, size=1, signed=_, word_type = DType.uint8]
    alias _BigUInt = BigUInt[_, size=1, word_type = DType.uint8]

    var uval8 = _BigUInt[8](255)
    var uval24 = _BigUInt[24](127)

    _assert_equal(_BigInt[8](other=uval24), 127)
    _assert_equal(_BigUInt[8](other=uval24), 127)

    _assert_equal(_BigInt[24](other=uval8), 255)
    _assert_equal(_BigUInt[24](other=uval24), 127)


fn test_max() raises:
    _assert_equal(BigInt[8, size=1].max(), 127)
    _assert_equal(BigInt[24, size=1].max(), 8_388_607)

    _assert_equal(BigUInt[8, size=1].max(), 255)
    _assert_equal(BigUInt[24, size=1].max(), 16_777_215)


fn test_min() raises:
    _assert_equal(BigInt[8, size=1].min(), -128)
    _assert_equal(BigInt[24, size=1].min(), -8_388_608)

    _assert_equal(BigUInt[8, size=1].min(), 0)
    _assert_equal(BigUInt[24, size=1].min(), 0)


fn test_select() raises:
    var condition = SIMD[DType.bool, 4](True, False, True, False)

    _assert_equal(
        BigUInt[8, size=4].select(condition, 1, 0),
        SIMD[DEST_TYPE, 4](1, 0, 1, 0),
    )

    _assert_equal(
        BigInt[24, size=4].select(condition, 1, -1),
        SIMD[DEST_TYPE, 4](1, -1, 1, -1),
    )


fn test_abs() raises:
    _assert_equal(
        abs(BigUInt[8](SIMD[DEST_TYPE, 4](0, 1, 2, 3))),
        SIMD[DEST_TYPE, 4](0, 1, 2, 3),
    )

    _assert_equal(
        abs(BigInt[24](SIMD[DEST_TYPE, 4](0, -1, 2, -3))),
        SIMD[DEST_TYPE, 4](0, 1, 2, 3),
    )


fn test_negation() raises:
    var val = SIMD[DEST_TYPE, 4](-2, -1, 0, 1)

    _assert_equal(-BigInt[8](val), BigInt[8](-val).cast[DEST_TYPE]())
    _assert_equal(-BigInt[24](val), BigInt[24](-val).cast[DEST_TYPE]())


fn test_unary_plus() raises:
    var val = SIMD[DEST_TYPE, 2](0, 1)

    @always_inline
    @parameter
    fn _test_plus[bits: Int, signed: Bool]() raises:
        var original = BigInt[bits, signed=signed](val)
        var updated = +original

        _assert_equal(updated, original.cast[DEST_TYPE]())

        updated += 1
        _assert_equal(updated, (original + 1).cast[DEST_TYPE]())

    _test_plus[8, signed=True]()
    _test_plus[8, signed=False]()

    _test_plus[24, signed=True]()
    _test_plus[24, signed=False]()


fn test_add() raises:
    _assert_equal(BigInt[8, size=1](1) + BigInt[8, size=1](-2), -1)
    _assert_equal(BigInt[24, size=1](1) + BigInt[24, size=1](-2), -1)

    _assert_equal(BigUInt[8, size=1](15) + BigUInt[8, size=1](16), 31)
    _assert_equal(BigUInt[24, size=1](255) + BigUInt[24, size=1](1), 256)

    _assert_equal(
        BigUInt[24](SIMD[DEST_TYPE, 2](1, 2)) + 2,
        SIMD[DEST_TYPE, 2](3, 4),
    )


fn test_iadd() raises:
    var sval = BigInt[8, size=1](1)
    var uval = BigUInt[24](SIMD[DEST_TYPE, 2](0, 255))

    sval += BigInt[8, size=1](-2)
    _assert_equal(sval, -1)

    uval += 1
    _assert_equal(uval, SIMD[DEST_TYPE, 2](1, 256))


fn test_sub() raises:
    _assert_equal(BigInt[8, size=1](1) - BigInt[8, size=1](-2), 3)
    _assert_equal(BigInt[24](Int32(383)) - BigInt[24](Int32(-511)), 894)

    _assert_equal(BigUInt[8, size=1](16) - BigUInt[8, size=1](15), 1)
    _assert_equal(BigUInt[24, size=1](383) - BigUInt[24, size=1](255), 128)

    _assert_equal(
        BigInt[24, size=4](SIMD[DEST_TYPE, 4](1, 2, 3, 4)) - 2,
        SIMD[DEST_TYPE, 4](-1, 0, 1, 2),
    )


fn test_isub() raises:
    var sval = BigInt[8, size=1](1)
    var uval = BigUInt[24](SIMD[DEST_TYPE, 2](1, 256))

    sval -= BigInt[8, size=1](-2)
    _assert_equal(sval, 3)

    uval -= 1
    _assert_equal(uval, SIMD[DEST_TYPE, 2](0, 255))


fn test_mul() raises:
    var ulhs = BigUInt[24](SIMD[DEST_TYPE, 4](51, 351, 2_500, 16_777_215))
    var urhs = BigUInt[24](SIMD[DEST_TYPE, 4](5, 128, 4_500, 1))

    _assert_equal(
        ulhs * urhs,
        SIMD[DEST_TYPE, 4](255, 44_928, 11_250_000, 16_777_215),
    )

    var slhs = BigInt[24](SIMD[DEST_TYPE, 4](51, -351, 1_051, -8_388_608))
    var srhs = BigInt[24](SIMD[DEST_TYPE, 4](5, -128, -4_500, 1))

    _assert_equal(
        slhs * srhs,
        SIMD[DEST_TYPE, 4](255, 44_928, -4_729_500, -8_388_608),
    )


fn test_imul() raises:
    var sval = BigInt[8, size=1](1)
    var uval = BigUInt[24](SIMD[DEST_TYPE, 2](1, 256))

    sval *= BigInt[8, size=1](-2)
    _assert_equal(sval, -2)

    uval *= 2
    _assert_equal(uval, SIMD[DEST_TYPE, 2](2, 512))


fn test_signed_bitwise_op() raises:
    var a = BigInt[24, size=1](-8_355_613)
    var b = BigInt[24, size=1](32_933)

    _assert_equal(a & b, 32_929)
    _assert_equal(a | b, -8_355_609)
    _assert_equal(a ^ b, -8_388_538)

    var c = BigInt[24, size=1](a)
    c &= b
    _assert_equal(c, 32_929)

    c = BigInt[24, size=1](a)
    c |= b
    _assert_equal(c, -8_355_609)

    c = BigInt[24, size=1](a)
    c ^= b
    _assert_equal(c, -8_388_538)


fn test_unsigned_bitwise_op() raises:
    var a = BigUInt[24, size=1](8_421_603)
    var b = BigUInt[24, size=1](32_933)

    _assert_equal(a & b, 32_929)
    _assert_equal(a | b, 8_421_607)
    _assert_equal(a ^ b, 8_388_678)

    var c = BigUInt[24, size=1](a)
    c &= b
    _assert_equal(c, 32_929)

    c = BigUInt[24, size=1](a)
    c |= b
    _assert_equal(c, 8_421_607)

    c = BigUInt[24, size=1](a)
    c ^= b
    _assert_equal(c, 8_388_678)


fn test_invert() raises:
    _assert_equal(~BigInt[8, size=1](1), -2)
    _assert_equal(~BigInt[24, size=1](1), -2)

    _assert_equal(~BigUInt[8, size=1](1), 254)
    _assert_equal(~BigUInt[24, size=1](1), 16_777_214)

    _assert_equal(~BigInt[8, size=1](127), -128)
    _assert_equal(~BigInt[24, size=1](8_388_607), -8_388_608)

    _assert_equal(~BigUInt[8, size=1](127), 128)
    _assert_equal(~BigUInt[24, size=1](8_388_607), 8_388_608)


fn test_lshift() raises:
    var sval8 = BigInt[8, size=4, word_type = DType.uint8](-99)
    var uval8 = BigUInt[8, size=4, word_type = DType.uint8](157)
    var offset = SIMD[DType.index, 4](0, 1, 4, 7)

    _assert_equal(sval8 << 0, -99)
    _assert_equal(sval8 << offset, SIMD[DEST_TYPE, 4](-99, 58, -48, -128))

    _assert_equal(uval8 << 0, 157)
    _assert_equal(uval8 << offset, SIMD[DEST_TYPE, 4](157, 58, 208, 128))

    var sval24 = BigInt[24, size=4, word_type = DType.uint8](-3_962_546)
    var uval24 = BigUInt[24, size=4, word_type = DType.uint8](12_814_670)
    offset = SIMD[DType.index, 4](0, 1, 12, 23)

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


fn test_lshift_edge_cases() raises:
    _assert_equal(BigInt[8, size=1](-128) << 1, 0)
    _assert_equal(BigInt[8, size=1](-1) << 8, 0)
    _assert_equal(BigInt[8, size=1](64) << 2, 0)
    _assert_equal(BigInt[8, size=1](1) << 8, 0)
    _assert_equal(BigUInt[8, size=1](128) << 1, 0)
    _assert_equal(BigUInt[8, size=1](1) << 8, 0)

    _assert_equal(BigInt[24, size=1](-8388608) << 1, 0)
    _assert_equal(BigInt[24, size=1](-1) << 24, 0)
    _assert_equal(BigInt[24, size=1](4194304) << 2, 0)
    _assert_equal(BigInt[24, size=1](1) << 24, 0)
    _assert_equal(BigUInt[24, size=1](8388608) << 1, 0)
    _assert_equal(BigUInt[24, size=1](1) << 24, 0)


fn test_ilshift() raises:
    var val = BigInt[24, size=1](1)
    val <<= 8
    _assert_equal(val, 256)


fn test_rshift() raises:
    var sval8 = BigInt[8, word_type = DType.uint8](
        SIMD[DType.int8, 4](1, 99, -99, -1)
    )
    var uval8 = BigUInt[8, size=4, word_type = DType.uint8](157)
    var offset = SIMD[DType.index, 4](0, 1, 4, 7)

    _assert_equal(sval8 >> 0, SIMD[DEST_TYPE, 4](1, 99, -99, -1))
    _assert_equal(sval8 >> offset, SIMD[DEST_TYPE, 4](1, 49, -7, -1))

    _assert_equal(uval8 >> 0, 157)
    _assert_equal(uval8 >> offset, SIMD[DEST_TYPE, 4](157, 78, 9, 1))

    var sval24 = BigInt[24, size=4, word_type = DType.uint8](-3_962_546)
    var uval24 = BigUInt[24, size=4, word_type = DType.uint8](12_814_670)
    offset = SIMD[DType.index, 4](0, 1, 12, 23)

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


fn test_rshift_edge_cases() raises:
    _assert_equal(BigInt[8, size=1](-128) >> 7, -1)
    _assert_equal(BigInt[8, size=1](-128) >> 8, -1)
    _assert_equal(BigInt[8, size=1](64) >> 7, 0)
    _assert_equal(BigInt[8, size=1](64) >> 8, 0)
    _assert_equal(BigUInt[8, size=1](128) >> 7, 1)
    _assert_equal(BigUInt[8, size=1](128) >> 8, 0)

    _assert_equal(BigInt[24, size=1](-8388608) >> 23, -1)
    _assert_equal(BigInt[24, size=1](-8388608) >> 24, -1)
    _assert_equal(BigInt[24, size=1](4194304) >> 23, 0)
    _assert_equal(BigInt[24, size=1](4194304) >> 24, 0)
    _assert_equal(BigUInt[24, size=1](8388608) >> 23, 1)
    _assert_equal(BigUInt[24, size=1](8388608) >> 24, 0)


fn test_irshift() raises:
    var val = BigInt[24, size=1].min()
    val >>= 16
    _assert_equal(val, -128)


fn test_comparison() raises:
    @always_inline
    @parameter
    fn _test_cmp[bits: Int, signed: Bool]() raises:
        alias WIDTH = 4

        var val1: BigInt[bits, size=WIDTH, signed=signed]
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


fn test_iadd_with_overflow() raises:
    var sval8 = BigInt[8](SIMD[DEST_TYPE, 2](-128, 127))
    var uval8 = BigUInt[8](SIMD[DEST_TYPE, 2](0, 255))

    assert_equal(sval8.iadd_with_overflow(1), SIMD[DType.bool, 2](False, True))
    _assert_equal(sval8, SIMD[DEST_TYPE, 2](-127, -128))

    assert_equal(sval8.iadd_with_overflow(-1), SIMD[DType.bool, 2](False, True))
    _assert_equal(sval8, SIMD[DEST_TYPE, 2](-128, 127))

    assert_equal(uval8.iadd_with_overflow(1), SIMD[DType.bool, 2](False, True))
    _assert_equal(uval8, SIMD[DEST_TYPE, 2](1, 0))

    var sval24 = BigInt[24](SIMD[DEST_TYPE, 2](-8_388_608, 8_388_607))
    var uval24 = BigUInt[24](SIMD[DEST_TYPE, 2](0, 16_777_215))

    assert_equal(sval24.iadd_with_overflow(1), SIMD[DType.bool, 2](False, True))
    _assert_equal(sval24, SIMD[DEST_TYPE, 2](-8_388_607, -8_388_608))

    assert_equal(
        sval24.iadd_with_overflow(-1), SIMD[DType.bool, 2](False, True)
    )
    _assert_equal(sval24, SIMD[DEST_TYPE, 2](-8_388_608, 8_388_607))

    assert_equal(uval24.iadd_with_overflow(1), SIMD[DType.bool, 2](False, True))
    _assert_equal(uval24, SIMD[DEST_TYPE, 2](1, 0))


fn test_isub_with_overflow() raises:
    var sval8 = BigInt[8](SIMD[DEST_TYPE, 2](-128, 127))
    var uval8 = BigUInt[8](SIMD[DEST_TYPE, 2](0, 255))

    assert_equal(sval8.isub_with_overflow(1), SIMD[DType.bool, 2](True, False))
    _assert_equal(sval8, SIMD[DEST_TYPE, 2](127, 126))

    assert_equal(sval8.isub_with_overflow(-1), SIMD[DType.bool, 2](True, False))
    _assert_equal(sval8, SIMD[DEST_TYPE, 2](-128, 127))

    assert_equal(uval8.isub_with_overflow(1), SIMD[DType.bool, 2](True, False))
    _assert_equal(uval8, SIMD[DEST_TYPE, 2](255, 254))

    var sval24 = BigInt[24](SIMD[DEST_TYPE, 2](-8_388_608, 8_388_607))
    var uval24 = BigUInt[24](SIMD[DEST_TYPE, 2](0, 16_777_215))

    assert_equal(sval24.isub_with_overflow(1), SIMD[DType.bool, 2](True, False))
    _assert_equal(sval24, SIMD[DEST_TYPE, 2](8_388_607, 8_388_606))

    assert_equal(
        sval24.isub_with_overflow(-1), SIMD[DType.bool, 2](True, False)
    )
    _assert_equal(sval24, SIMD[DEST_TYPE, 2](-8_388_608, 8_388_607))

    assert_equal(uval24.isub_with_overflow(1), SIMD[DType.bool, 2](True, False))
    _assert_equal(uval24, SIMD[DEST_TYPE, 2](16_777_215, 16_777_214))


fn test_full_mul() raises:
    var num = 8_388_607
    var slhs = BigInt[24](SIMD[DEST_TYPE, 4](num, -num, num, -num))
    var srhs = BigInt[24](SIMD[DEST_TYPE, 4](num, num, -num, -num))

    _assert_equal(
        slhs.full_mul(srhs),
        SIMD[DEST_TYPE, 4](num * num, -(num * num), -(num * num), num * num),
    )

    num = 16_777_215
    var big_num = BigUInt[24, size=1](num)

    _assert_equal(
        big_num.full_mul(big_num),
        SIMD[DEST_TYPE, 1](num * num),
    )


fn test_approx_mul_high() raises:
    alias BITS = 24

    var a = BigUInt[BITS](
        SIMD[DEST_TYPE, 4](2_097_151, 4_194_303, 8_388_607, 16_777_215)
    )
    var b = BigUInt[BITS, size=4](16_777_215)

    var full_prod = a.full_mul(b)

    _assert_equal(
        full_prod,
        SIMD[DEST_TYPE, 4](
            35_184_353_214_465,
            70_368_723_206_145,
            140_737_463_189_505,
            281_474_943_156_225,
        ),
    )

    var error = (
        (a.full_mul(b) >> BITS).cast[DEST_TYPE]()
        - a.approx_mul_high(b).cast[DEST_TYPE]()
    )

    assert_true(all(error >= 0))
    assert_true(all(error < a.WORD_COUNT))


fn test_cast() raises:
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


fn test_is_negative() raises:
    var val = SIMD[DType.int8, 2](-1, 0)

    assert_equal(BigInt[8](val).is_negative(), val < 0)
    assert_equal(BigUInt[8, size=1](0).is_negative(), False)

    assert_equal(BigInt[24](val).is_negative(), val < 0)
    assert_equal(BigUInt[24, size=1](0).is_negative(), False)


fn test_is_zero() raises:
    var sval = SIMD[DType.int8, 4](-1, 0, 1, 2)
    var uval = SIMD[DType.uint8, 4](0, 1, 2, 3)

    assert_equal(BigInt[8](sval).is_zero(), sval == 0)
    assert_equal(BigUInt[8](uval).is_zero(), uval == 0)

    assert_equal(BigInt[24](sval).is_zero(), sval == 0)
    assert_equal(BigUInt[24](uval).is_zero(), uval == 0)


fn test_most_significant_digit() raises:
    var sval = BigInt[24, size=1].max()
    var uval = BigUInt[24, size=1].max()

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
    var uval8 = BigUInt[8, size=1](0)

    assert_equal(uval8.count_leading_zeros(), 8)

    uval8.set_most_significant_bit()
    assert_equal(uval8.count_leading_zeros(), 0)

    uval8 >>= 1
    assert_equal(uval8.count_leading_zeros(), 1)

    var uval24 = BigUInt[24, size=1](0)

    assert_equal(uval24.count_leading_zeros(), 24)

    uval24.set_most_significant_bit()
    assert_equal(uval24.count_leading_zeros(), 0)

    uval24 >>= 12
    assert_equal(uval24.count_leading_zeros(), 12)


fn main() raises:
    test_init()
    test_init_unsafe()
    test_init_from_int_literal()
    test_init_from_int()
    test_init_from_signed_simd()
    test_init_from_unsigned_simd()
    test_explicit_copy()
    test_from_signed_big_int()
    test_from_unsigned_big_int()

    test_max()
    test_min()
    test_select()

    test_abs()
    test_negation()
    test_unary_plus()

    test_add()
    test_iadd()

    test_sub()
    test_isub()

    test_mul()
    test_imul()

    test_signed_bitwise_op()
    test_unsigned_bitwise_op()
    test_invert()

    test_lshift()
    test_lshift_edge_cases()
    test_ilshift()

    test_rshift()
    test_rshift_edge_cases()
    test_irshift()

    test_comparison()

    test_iadd_with_overflow()
    test_isub_with_overflow()

    test_full_mul()
    test_approx_mul_high()

    test_cast()

    test_is_negative()
    test_is_zero()

    test_most_significant_digit()

    test_count_leading_zeros()
