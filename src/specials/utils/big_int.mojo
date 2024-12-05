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
# Some of the code in this file is adapted from:
#
# llvm/llvm-project:
# Licensed under the Apache License v2.0 with LLVM Exceptions.

"""Implements the `BigInt` struct."""

from bit import countl_zero
from memory import DTypePointer, memset_zero
from sys.info import is_64bit, sizeof
from sys.intrinsics import _RegisterPackType
from utils import InlineArray
from utils.numerics import max_finite


# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


@always_inline
fn _big_int_construction_checks[
    bits: Int,
    word_type: DType,
]():
    """Performs checks on the parameters of a `BigInt` constructor."""
    constrained[bits > 0, "number of bits must be positive"]()
    constrained[
        word_type.is_unsigned(), "word type must be an unsigned, integral type"
    ]()
    constrained[
        bits % word_type.bitwidth() == 0,
        "number of bits must be a multiple of the word type's bitwidth",
    ]()


@always_inline
fn _bits_as_int_literal[bits: Int]() -> IntLiteral:
    """Converts the number of bits to an integer literal."""
    constrained[bits >= 0, "number of bits must be non-negative"]()
    constrained[bits % 8 == 0, "number of bits must be a multiple of 8"]()

    var result: IntLiteral = 0

    @parameter
    for _ in range(0, bits, 8):
        result += 8

    return result


@always_inline
fn _exp2[n: IntLiteral]() -> IntLiteral:
    """Computes 2 raised to the power of `n`."""
    constrained[n >= 0, "exponent must be non-negative"]()

    return 1 << n


@always_inline
fn _conditional[T: AnyType, //, pred: Bool, true_case: T, false_case: T]() -> T:
    """Returns the true or false case based on the value of `pred`."""

    @parameter
    if pred:
        return true_case
    else:
        return false_case


@always_inline
fn _default_word_type[bits: Int]() -> DType:
    """Returns the default word type for a `BigInt` based on `bits`."""
    constrained[bits % 8 == 0, "number of bits must be a multiple of 8"]()

    @parameter
    if bits % 64 == 0 and is_64bit():
        return DType.uint64
    elif bits % 32 == 0:
        return DType.uint32
    elif bits % 16 == 0:
        return DType.uint16
    else:
        return DType.uint8


@always_inline
fn _half_bitwidth_type_of[type: DType]() -> DType:
    """Returns the unsigned type with half the bitwidth of the input type."""
    constrained[type.is_unsigned(), "type must be an unsigned, integral type"]()

    @parameter
    if type == DType.uint16:
        return DType.uint8
    elif type == DType.uint32:
        return DType.uint16
    elif type == DType.uint64:
        return DType.uint32
    else:
        return DType.invalid


# ===----------------------------------------------------------------------=== #
# Operations with carry propagation
# ===----------------------------------------------------------------------=== #


@always_inline
fn _uadd_with_overflow(
    lhs: SIMD, rhs: __type_of(lhs)
) -> (__type_of(lhs), __type_of(lhs)):
    """Performs unsigned addition with overflow detection."""
    constrained[
        lhs.type.is_unsigned(),
        "input type must be an unsigned, integral type",
    ]()

    var result = llvm_intrinsic[
        "llvm.uadd.with.overflow",
        _RegisterPackType[__type_of(lhs), SIMD[DType.bool, lhs.size]],
        __type_of(lhs),
        __type_of(lhs),
    ](lhs, rhs)

    return result[0], result[1].cast[lhs.type]()


@always_inline
fn _uadd_with_carry(
    lhs: SIMD, rhs: __type_of(lhs), carry_in: __type_of(lhs)
) -> (__type_of(lhs), __type_of(lhs)):
    """Performs unsigned addition with carry propagation."""
    constrained[
        lhs.type.is_unsigned(),
        "input type must be an unsigned, integral type",
    ]()

    var sum_and_carry0 = _uadd_with_overflow(lhs, rhs)
    var sum_and_carry1 = _uadd_with_overflow(sum_and_carry0[0], carry_in)
    var carry_out = sum_and_carry0[1] | sum_and_carry1[1]

    return sum_and_carry1[0], carry_out


@always_inline
fn _usub_with_overflow(
    lhs: SIMD, rhs: __type_of(lhs)
) -> (__type_of(lhs), __type_of(lhs)):
    """Performs unsigned subtraction with overflow detection."""
    constrained[
        lhs.type.is_unsigned(),
        "input type must be an unsigned, integral type",
    ]()

    var result = llvm_intrinsic[
        "llvm.usub.with.overflow",
        _RegisterPackType[__type_of(lhs), SIMD[DType.bool, lhs.size]],
        __type_of(lhs),
        __type_of(lhs),
    ](lhs, rhs)

    return result[0], result[1].cast[lhs.type]()


@always_inline
fn _usub_with_carry(
    lhs: SIMD, rhs: __type_of(lhs), carry_in: __type_of(lhs)
) -> (__type_of(lhs), __type_of(lhs)):
    """Performs subtraction with carry propagation."""
    constrained[
        lhs.type.is_unsigned(),
        "input type must be an unsigned, integral type",
    ]()

    var diff_and_carry0 = _usub_with_overflow(lhs, rhs)
    var diff_and_carry1 = _usub_with_overflow(diff_and_carry0[0], carry_in)
    var carry_out = diff_and_carry0[1] | diff_and_carry1[1]

    return diff_and_carry1[0], carry_out


@always_inline
fn _apply_binop_with_carry[
    binop_with_carry: fn[type: DType, size: Int] (
        SIMD[type, size], SIMD[type, size], SIMD[type, size]
    ) -> (SIMD[type, size], SIMD[type, size]),
    type: DType,
    size: Int,
](
    inout dst: InlineArray[SIMD[type, size], _],
    rhs: InlineArray[SIMD[type, size], _],
) -> SIMD[type, size]:
    """Performs an in-place binary operation with carry propagation."""
    constrained[
        dst.size >= rhs.size,
        "`dst` must have at least as many elements as `rhs`",
    ]()

    var rhs_value: SIMD[type, size]
    var carry_out = SIMD[type, size](0)

    @parameter
    for i in range(dst.size):
        alias HAS_RHS_VALUE = i < rhs.size

        @parameter
        if HAS_RHS_VALUE:
            rhs_value = rhs[i]
        else:
            rhs_value = 0

        var carry_in = carry_out
        var result_and_carry = binop_with_carry(dst[i], rhs_value, carry_in)

        dst[i] = result_and_carry[0]
        carry_out = result_and_carry[1]

        # Stop early when rhs is over and there is no carry out to propagate.
        @parameter
        if not HAS_RHS_VALUE:
            if all(carry_out == 0):
                break

    return carry_out


@always_inline
fn _get_lower_half[new_type: DType](value: SIMD) -> SIMD[new_type, value.size]:
    """Extracts the lower half of each element in a SIMD vector."""
    constrained[
        new_type.is_unsigned(), "new type must be an unsigned, integral type"
    ]()
    constrained[
        value.type.is_unsigned(),
        "input type must be an unsigned, integral type",
    ]()
    constrained[
        value.type.bitwidth() == 2 * new_type.bitwidth(),
        "new type must be half the bitwidth of the input type",
    ]()

    return value.cast[new_type]()


@always_inline
fn _get_higher_half[new_type: DType](value: SIMD) -> SIMD[new_type, value.size]:
    """Extracts the higher half of each element in a SIMD vector."""
    constrained[
        new_type.is_unsigned(), "new type must be an unsigned, integral type"
    ]()
    constrained[
        value.type.is_unsigned(),
        "input type must be an unsigned, integral type",
    ]()
    constrained[
        value.type.bitwidth() == 2 * new_type.bitwidth(),
        "new type must be half the bitwidth of the input type",
    ]()

    return (value >> new_type.bitwidth()).cast[new_type]()


@always_inline
fn _split_into_halves[
    new_type: DType
](value: SIMD) -> InlineArray[SIMD[new_type, value.size], 2]:
    """Splits each element in a SIMD vector into its lower and higher halves."""
    return InlineArray[SIMD[new_type, value.size], 2](
        _get_lower_half[new_type](value),
        _get_higher_half[new_type](value),
    )


@always_inline
fn _umul_double_wide(
    lhs: SIMD, rhs: __type_of(lhs)
) -> InlineArray[__type_of(lhs), 2]:
    """Performs unsigned multiplication producing a double-width result."""
    constrained[
        lhs.type.is_unsigned(),
        "input type must be an unsigned, integral type",
    ]()

    @parameter
    if lhs.type == DType.uint8:
        return _split_into_halves[lhs.type](
            lhs.cast[DType.uint16]() * rhs.cast[DType.uint16]()
        )
    elif lhs.type == DType.uint16:
        return _split_into_halves[lhs.type](
            lhs.cast[DType.uint32]() * rhs.cast[DType.uint32]()
        )
    elif lhs.type == DType.uint32 and is_64bit():
        return _split_into_halves[lhs.type](
            lhs.cast[DType.uint64]() * rhs.cast[DType.uint64]()
        )
    else:
        alias HALF_BITWIDTH = lhs.type.bitwidth() // 2
        alias HALF_BITWIDTH_TYPE = _half_bitwidth_type_of[lhs.type]()

        var lhs_lo = _get_lower_half[HALF_BITWIDTH_TYPE](lhs).cast[lhs.type]()
        var rhs_lo = _get_lower_half[HALF_BITWIDTH_TYPE](rhs).cast[lhs.type]()
        var lhs_hi = _get_higher_half[HALF_BITWIDTH_TYPE](lhs).cast[lhs.type]()
        var rhs_hi = _get_higher_half[HALF_BITWIDTH_TYPE](rhs).cast[lhs.type]()

        var prod_lo_lo = rhs_lo * lhs_lo
        var prod_lo_hi = rhs_lo * lhs_hi
        var prod_hi_lo = rhs_hi * lhs_lo
        var prod_hi_hi = rhs_hi * lhs_hi

        var sum_and_carry = _uadd_with_carry(
            prod_lo_lo, prod_lo_hi << HALF_BITWIDTH, 0
        )
        var lower_half = sum_and_carry[0]
        var carry = sum_and_carry[1]

        sum_and_carry = _uadd_with_carry(
            prod_hi_hi, prod_lo_hi >> HALF_BITWIDTH, carry
        )
        var higher_half = sum_and_carry[0]

        sum_and_carry = _uadd_with_carry(
            lower_half, prod_hi_lo << HALF_BITWIDTH, 0
        )
        lower_half = sum_and_carry[0]
        carry = sum_and_carry[1]

        sum_and_carry = _uadd_with_carry(
            higher_half, prod_hi_lo >> HALF_BITWIDTH, carry
        )
        higher_half = sum_and_carry[0]

        return InlineArray[__type_of(lhs), 2](lower_half, higher_half)


@value
struct _Accumulator[type: DType, size: Int]:
    """Represents an accumulator for SIMD operations with carry propagation."""

    var _storage: InlineArray[SIMD[type, size], 2]
    """The internal storage of the accumulator."""

    @always_inline
    fn __init__(inout self):
        """Initializes the accumulator with all elements set to zero."""
        self._storage = InlineArray[SIMD[type, size], 2](0, 0)

    @always_inline
    fn propagate(inout self, carry_in: SIMD[type, size]) -> SIMD[type, size]:
        """Propagates the accumulator elements forward and incorporates a
        carry-in value.
        """
        var result = self._storage[0]
        self._storage[0] = self._storage[1]
        self._storage[1] = carry_in

        return result

    @always_inline
    fn get_sum(self) -> SIMD[type, size]:
        """Returns the sum of the accumulator."""
        return self._storage[0]

    @always_inline
    fn get_carry(self) -> SIMD[type, size]:
        """Returns the carry of the accumulator."""
        return self._storage[1]


@always_inline
fn _umul_uadd_with_carry[
    type: DType,
    size: Int,
](
    inout dst: _Accumulator[type, size],
    lhs: SIMD[type, size],
    rhs: SIMD[type, size],
) -> SIMD[type, size]:
    """Performs in-place unsigned multiply-add operation with carry propagation.
    """
    return _apply_binop_with_carry[_uadd_with_carry](
        dst._storage, _umul_double_wide(lhs, rhs)
    )


@always_inline
fn _umul_with_carry[
    type: DType, size: Int
](
    inout dst: InlineArray[SIMD[type, size], _],
    lhs: InlineArray[SIMD[type, size], _],
    rhs: InlineArray[SIMD[type, size], _],
) -> SIMD[type, size]:
    """Performs in-place unsigned multiplication with carry propagation."""
    constrained[
        dst.size >= lhs.size + rhs.size,
        "`dst.size` must be greater than or equal to `lhs.size + rhs.size`",
    ]()

    var acc = _Accumulator[type, size]()

    @parameter
    for i in range(dst.size):
        alias LOWER_IDX = _conditional[i < rhs.size, 0, i - rhs.size + 1]()
        alias UPPER_IDX = _conditional[i < lhs.size, i, lhs.size - 1]()

        var carry = SIMD[type, size](0)

        @parameter
        for j in range(LOWER_IDX, UPPER_IDX + 1):
            carry += _umul_uadd_with_carry(acc, lhs[j], rhs[i - j])

        dst[i] = acc.propagate(carry)

    return acc.get_carry()


@always_inline
fn _approx_umul_high[
    type: DType, size: Int
](
    inout dst: InlineArray[SIMD[type, size], _],
    lhs: __type_of(dst),
    rhs: __type_of(dst),
):
    """Approximates the higher part of the unsigned multiplication result."""
    var acc = _Accumulator[type, size]()
    var carry = SIMD[type, size](0)

    @parameter
    for i in range(dst.size):
        carry += _umul_uadd_with_carry(acc, lhs[i], rhs[dst.size - 1 - i])

    @parameter
    for i in range(dst.size, 2 * dst.size - 1):
        _ = acc.propagate(carry)
        carry = 0

        @parameter
        for j in range(i - dst.size + 1, dst.size):
            carry += _umul_uadd_with_carry(acc, lhs[j], rhs[i - j])

        dst[i - dst.size] = acc.get_sum()

    dst[dst.size - 1] = acc.get_carry()


# ===----------------------------------------------------------------------=== #
# Operations for bit manipulation
# ===----------------------------------------------------------------------=== #


@always_inline
fn _count_leading_zeros[
    type: DType, size: Int
](val: InlineArray[SIMD[type, size], _]) -> SIMD[type, size]:
    """Counts the leading zeros in the internal representation of a `BigInt`."""
    constrained[type.is_integral(), "type must be an integral type"]()

    var result = SIMD[type, size](0)
    var should_stop = SIMD[DType.bool, size](False)

    @parameter
    for i in reversed(range(val.size)):
        var bit_count = countl_zero(val[i])
        result += should_stop.select(0, bit_count)
        should_stop |= bit_count < type.bitwidth()

        if all(should_stop):
            break

    return result


@always_inline
fn _mask_leading_ones[type: DType, count: Int]() -> Scalar[type]:
    """Creates a mask with the specified number of leading ones."""
    return ~_mask_trailing_ones[type, type.bitwidth() - count]()


@always_inline
fn _mask_trailing_ones[type: DType, count: Int]() -> Scalar[type]:
    """Creates a mask with the specified number of trailing ones."""
    constrained[type.is_unsigned(), "type must be an unsigned, integral type"]()
    constrained[count >= 0, "count must be non-negative"]()
    constrained[
        count <= type.bitwidth(),
        "count must be less than or equal to the type's bitwidth",
    ]()

    @parameter
    if count == 0:
        return Scalar[type](0)
    else:
        return ~Scalar[type](0) >> (type.bitwidth() - count)


@always_inline
fn _iota[type: DType, size: Int]() -> SIMD[type, size]:
    """Creates a SIMD vector containing an increasing sequence starting from 0.
    """
    # Unlike `math.iota`, this function can be executed at compile-time.
    var result = SIMD[type, size]()

    @parameter
    for i in range(size):
        result[i] = i

    return result


@always_inline
fn _shift[
    *, is_left_shift: Bool, treat_offset_as_int: Bool = False
](val: BigInt, offset: SIMD[DType.index, val.size]) -> __type_of(val):
    """Performs a bitwise shift on a `BigInt` vector, element-wise."""
    alias SHIFT = _iota[DType.index, val.size]()
    alias TYPE_BITWIDTH = val.word_type.bitwidth()

    var dst = __type_of(val)(unsafe_uninitialized=True)
    var val_is_negative = val.is_negative()
    var val_ptr = DTypePointer(
        val._storage.unsafe_ptr().bitcast[Scalar[val.word_type]]()
    )

    @parameter
    @always_inline
    fn at[index: Int]() -> Int:
        return _conditional[is_left_shift, val.WORD_COUNT - index - 1, index]()

    @parameter
    @always_inline
    fn at(index: __type_of(offset)) -> __type_of(offset):
        @parameter
        if is_left_shift:
            alias WORD_COUNT_MINUS_ONE = val.WORD_COUNT - 1
            return WORD_COUNT_MINUS_ONE - index
        else:
            return index

    @parameter
    @always_inline
    fn safe_get(index: __type_of(offset)) -> SIMD[val.word_type, val.size]:
        var is_index_below_size = index < val.WORD_COUNT
        var mask = (index >= 0) & is_index_below_size
        var default = (val_is_negative & ~is_index_below_size).select(
            SIMD[val.word_type, val.size](-1), 0
        )

        @parameter
        if treat_offset_as_int:
            var safe_index = int(mask.select(index, 0)[0])
            return mask.select(
                val_ptr.load[width = val.size](safe_index * val.size), default
            )
        else:
            return val_ptr.gather(index.fma(val.size, SHIFT), mask, default)

    var index_offset = offset // TYPE_BITWIDTH
    var index_offset_plus_one = index_offset + 1

    var bit_offset = (offset % TYPE_BITWIDTH).cast[val.word_type]()
    var bit_offset_is_zero = bit_offset == 0
    var bit_offset_complement = TYPE_BITWIDTH - bit_offset

    var part2 = safe_get(at(index_offset))

    @parameter
    for i in range(val.WORD_COUNT):
        var part1 = part2
        part2 = safe_get(at(i + index_offset_plus_one))

        @parameter
        if is_left_shift:
            dst._storage[at[i]()] = bit_offset_is_zero.select(
                part1,
                part1 << bit_offset | part2 >> bit_offset_complement,
            )
        else:
            dst._storage[at[i]()] = bit_offset_is_zero.select(
                part1,
                part1 >> bit_offset | part2 << bit_offset_complement,
            )

    return dst


@always_inline
fn _extend[
    type: DType, size: Int, *, start: Int
](
    inout val: InlineArray[SIMD[type, size], _],
    is_negative: SIMD[DType.bool, size],
):
    """Extends the internal representation of a `BigInt` with a sign extension.
    """
    constrained[type.is_unsigned(), "type must be an unsigned, integral type"]()

    var extension = is_negative.select(max_finite[type](), 0)

    @parameter
    for i in range(start, val.size):
        val[i] = extension


@always_inline
fn _apply_binop[
    binop: fn[type: DType, size: Int] (
        SIMD[type, size], SIMD[type, size]
    ) -> SIMD[type, size],
    type: DType,
    size: Int,
](
    lhs: InlineArray[SIMD[type, size], _],
    rhs: __type_of(lhs),
) -> __type_of(
    lhs
):
    """Performs a binary operation element-wise."""
    var dst = __type_of(lhs)(unsafe_uninitialized=True)

    @parameter
    for i in range(lhs.size):
        dst[i] = binop(lhs[i], rhs[i])

    return dst


@always_inline
fn _apply_inplace_binop[
    binop: fn[type: DType, size: Int] (
        SIMD[type, size], SIMD[type, size]
    ) -> SIMD[type, size],
    type: DType,
    size: Int,
](inout dst: InlineArray[SIMD[type, size], _], rhs: __type_of(dst)):
    """Performs an in-place binary operation element-wise."""

    @parameter
    for i in range(dst.size):
        dst[i] = binop(dst[i], rhs[i])


# ===----------------------------------------------------------------------=== #
# Operations for comparison
# ===----------------------------------------------------------------------=== #


@always_inline
fn _compare(lhs: SIMD, rhs: __type_of(lhs)) -> SIMD[DType.int8, lhs.size]:
    """Compares two SIMD vectors element-wise."""
    return (lhs == rhs).select(
        0, (lhs < rhs).select(-1, SIMD[DType.int8, lhs.size](1))
    )


@always_inline
fn _compare(lhs: BigInt, rhs: __type_of(lhs)) -> SIMD[DType.int8, lhs.size]:
    """Compares two `BigInt` vectors element-wise."""
    var result = SIMD[DType.int8, lhs.size](0)

    @parameter
    if lhs.signed:
        var lhs_is_negative = lhs.is_negative()
        var rhs_is_negative = rhs.is_negative()
        result = (lhs_is_negative != rhs_is_negative).select(
            lhs_is_negative.select(-1, SIMD[DType.int8, lhs.size](1)), result
        )

    @parameter
    for i in reversed(range(lhs.WORD_COUNT)):
        result = (result == 0).select(
            _compare(lhs._storage[i], rhs._storage[i]), result
        )

        if all(result != 0):
            break

    return result


# ===----------------------------------------------------------------------=== #
# Casting safety checks
# ===----------------------------------------------------------------------=== #


@always_inline
fn _is_casting_safe[bits: Int, signed: Bool](value: IntLiteral) -> Bool:
    """Checks if `value` fits in an integer with the specified number of bits
    and signedness.
    """

    @parameter
    if signed:
        return bits >= value._bit_width()
    else:
        return value >= 0 and bits >= (value._bit_width() - 1)


@always_inline
fn _is_casting_safe[bits: Int, signed: Bool](value: SIMD) -> Bool:
    """Checks if `value` fits in an integer with the specified number of bits
    and signedness.
    """
    constrained[
        value.type.is_integral(),
        "value type must be an integral type",
    ]()

    @parameter
    if bits > value.type.bitwidth():
        return signed or value.type.is_unsigned() or all(value >= 0)
    else:
        alias TWO = Scalar[value.type](2)

        @parameter
        if value.type.is_unsigned():
            alias MAX_VALUE = _conditional[
                signed, TWO ** (bits - 1) - 1, TWO**bits - 1
            ]()

            return all(value <= MAX_VALUE)
        else:
            alias MIN_VALUE = _conditional[signed, -(TWO ** (bits - 1)), 0]()

            @parameter
            if bits == value.type.bitwidth():
                return all(value >= MIN_VALUE)
            else:
                alias MAX_VALUE = _conditional[
                    signed, TWO ** (bits - 1) - 1, TWO**bits - 1
                ]()

                return all(value >= MIN_VALUE) and all(value <= MAX_VALUE)


@always_inline
fn _is_casting_safe[bits: Int, signed: Bool](value: BigInt) -> Bool:
    """Checks if `value` fits in an integer with the specified number of bits
    and signedness.
    """

    @parameter
    if bits > value.bits:
        return signed or (not value.signed) or all(value >= 0)
    else:
        alias BITS: IntLiteral = _bits_as_int_literal[bits]()

        @parameter
        if not value.signed:
            alias MAX_VALUE: IntLiteral = (
                _exp2[BITS - 1]() - 1 if signed else _exp2[BITS]() - 1
            )

            return all(value <= MAX_VALUE)
        else:
            alias MIN_VALUE = -_exp2[BITS - 1]() if signed else 0

            @parameter
            if bits == value.bits:
                return all(value >= MIN_VALUE)
            else:
                alias MAX_VALUE: IntLiteral = (
                    _exp2[BITS - 1]() - 1 if signed else _exp2[BITS]() - 1
                )

                return all(value >= MIN_VALUE) and all(value <= MAX_VALUE)


# ===----------------------------------------------------------------------=== #
# BigInt
# ===----------------------------------------------------------------------=== #


alias BigUInt = BigInt[_, size=_, signed=False, word_type=_]
"""Represents a small vector of arbitrary, fixed bit-size unsigned integers."""


@value
struct BigInt[
    bits: Int,
    /,
    *,
    size: Int,
    signed: Bool = True,
    word_type: DType = _default_word_type[bits](),
](Absable, Copyable, Defaultable, ExplicitlyCopyable, Movable):
    """Represents a small vector of arbitrary, fixed bit-size integers.

    It can represent both signed and unsigned integers with a fixed number of
    bits each and leverages SIMD operations for enhanced performance.

    Constraints:
        The number of bits must be a multiple of the word type's bitwidth.

    Parameters:
        bits: The number of bits for each element of the `BigInt` vector.
            Constraints: Must be positive.
        size: The size of the `BigInt` vector. Constraints: Must be positive
            and a power of two.
        signed: A boolean indicating whether the integers are signed (`True`)
            or unsigned (`False`). Defaults to `True`.
        word_type: The type of each word used in the internal representation
            of the `BigInt` vector. For performance reasons, defaults to the
            largest unsigned integer type whose bitwidth divides `bits` evenly.
    """

    # ===------------------------------------------------------------------=== #
    # Aliases
    # ===------------------------------------------------------------------=== #

    alias WORD_TYPE_BITWIDTH = 8 * sizeof[word_type]()
    """The size, in bits, of the `word_type` parameter."""

    alias WORD_COUNT = bits // Self.WORD_TYPE_BITWIDTH
    """The number of words used to represent the integers."""

    alias StorageType = InlineArray[SIMD[word_type, size], Self.WORD_COUNT]
    """The type used for internal storage, represented as an array of words."""

    # ===------------------------------------------------------------------=== #
    # Fields
    # ===------------------------------------------------------------------=== #

    var _storage: Self.StorageType
    """The internal array of words representing the `BigInt` vector."""

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __init__(inout self):
        """Initializes the `BigInt` vector with all elements set to zero."""
        _big_int_construction_checks[bits, word_type]()

        alias BLOCK_SIZE = size * Self.WORD_COUNT

        self._storage = Self.StorageType(unsafe_uninitialized=True)

        memset_zero[word_type](
            self._storage.unsafe_ptr().bitcast[Scalar[word_type]](),
            count=BLOCK_SIZE,
        )

    @doc_private
    @always_inline
    fn __init__(inout self, *, unsafe_uninitialized: Bool):
        """Initializes the `BigInt` vector with an uninitialized storage.

        Args:
            unsafe_uninitialized: Marker argument indicating this initializer
                is unsafe. Its value is not used.
        """
        _big_int_construction_checks[bits, word_type]()

        _ = unsafe_uninitialized
        self._storage = Self.StorageType(unsafe_uninitialized=True)

    @always_inline
    fn __init__(inout self, value: IntLiteral):
        """Initializes the `BigInt` vector with a signed integer literal.

        Args:
            value: The signed integer literal to be splatted across all the
                elements of the `BigInt` vector. Should be within the bounds of
                the `BigInt`.
        """
        _big_int_construction_checks[bits, word_type]()

        debug_assert(
            _is_casting_safe[bits, signed](value),
            "value should be within the bounds of the `BigInt`",
        )

        self._storage = Self.StorageType(unsafe_uninitialized=True)
        self._storage[0] = SIMD[word_type, size](value)

        if (
            Self.WORD_TYPE_BITWIDTH >= value._bit_width()
            and self.WORD_COUNT > 1
        ):
            _extend[start=1](self._storage, is_negative=value < 0 and signed)
            return

        var tmp: IntLiteral = value

        @parameter
        for i in range(1, Self.WORD_COUNT):
            tmp >>= Self.WORD_TYPE_BITWIDTH
            self._storage[i] = SIMD[word_type, size](tmp)

    @always_inline
    fn __init__(inout self, value: Int):
        """Initializes the `BigInt` vector with a signed integer value.

        Args:
            value: The signed integer value to be splatted across all the
                elements of the `BigInt` vector. Should be within the bounds of
                the `BigInt`.
        """
        self.__init__(SIMD[DType.index, size](value))

    @always_inline
    fn __init__(inout self, value: SIMD[_, size]):
        """Initializes the `BigInt` vector with a SIMD vector.

        Constraints:
            The value type must be an integral type.

        Args:
            value: The SIMD vector to initialize the `BigInt` vector with. Each
                of its elements should be within the bounds of the `BigInt`.
        """
        constrained[
            value.type.is_integral(),
            "value type must be an integral type",
        ]()
        _big_int_construction_checks[bits, word_type]()

        debug_assert(
            _is_casting_safe[bits, signed](value),
            (
                "each element in `value` should be within the bounds of the"
                " `BigInt`"
            ),
        )

        self._storage = Self.StorageType(unsafe_uninitialized=True)
        self._storage[0] = value.cast[word_type]()

        @parameter
        if (
            Self.WORD_TYPE_BITWIDTH >= value.type.bitwidth()
            and self.WORD_COUNT > 1
        ):
            _extend[start=1](self._storage, is_negative=value < 0 & signed)
            return

        var tmp = value

        @parameter
        for i in range(1, Self.WORD_COUNT):
            tmp >>= Self.WORD_TYPE_BITWIDTH
            self._storage[i] = tmp.cast[word_type]()

    @always_inline
    fn __init__(inout self, other: Self):
        """Initializes the `BigInt` vector by copying an existing one.

        Args:
            other: The `BigInt` vector to copy from.
        """
        self.__copyinit__(other)

    @always_inline
    fn __init__(
        inout self,
        *,
        other: BigInt[_, size=size, signed=_, word_type=word_type],
        unsafe_unchecked: Bool = False,
    ):
        """Initializes the `BigInt` vector by casting an existing one.

        Args:
            other: The `BigInt` vector to cast from. Must have the same size and
                word type as the new `BigInt`, and each element should be within
                the bounds of the new vector.
            unsafe_unchecked: A boolean indicating whether to skip the bounds
                checks that are performed when debug assertions are enabled. If
                set to `True`, the constructor will always bypass these checks,
                which may lead to data truncation if elements in `other` exceed
                the capacity of the new `BigInt`. Defaults to `False`.
        """
        _big_int_construction_checks[bits, word_type]()

        debug_assert(
            unsafe_unchecked or _is_casting_safe[bits, signed](other),
            (
                "each element in `other` should be within the bounds of the"
                " new `BigInt`"
            ),
        )

        self._storage = Self.StorageType(unsafe_uninitialized=True)

        @parameter
        if bits <= other.bits:

            @parameter
            for i in range(Self.WORD_COUNT):
                self._storage[i] = other._storage[i]

        else:
            var extension = (other.is_negative() & other.signed).select(
                max_finite[word_type](), 0
            )

            @parameter
            for i in range(Self.WORD_COUNT):

                @parameter
                if i < other.WORD_COUNT:
                    self._storage[i] = other._storage[i]
                else:
                    self._storage[i] = extension

    # ===------------------------------------------------------------------=== #
    # Factory methods
    # ===------------------------------------------------------------------=== #

    @staticmethod
    @always_inline
    fn max() -> Self:
        """Creates a `BigInt` vector with all elements set to the maximum
        representable value.

        Returns:
            A new `BigInt` vector with all elements set to the maximum
            representable value.
        """
        var result = ~Self()

        @parameter
        if signed:
            result.clear_most_significant_bit()

        return result

    @staticmethod
    @always_inline
    fn min() -> Self:
        """Creates a `BigInt` vector with all elements set to the minimum
        representable value.

        Returns:
            A new `BigInt` vector with all elements set to the minimum
            representable value.
        """
        var result = Self()

        @parameter
        if signed:
            result.set_most_significant_bit()

        return result

    @staticmethod
    @always_inline
    fn select(
        condition: SIMD[DType.bool, size], true_case: Self, false_case: Self
    ) -> Self:
        """Selects elements from two `BigInt` vectors element-wise based on a
        condition.

        Args:
            condition: A boolean SIMD vector used to select elements. If `True`,
                selects from `true_case`; otherwise, from `false_case`.
            true_case: The `BigInt` vector to select elements from when the
                condition is `True`.
            false_case: The `BigInt` vector to select elements from when the
                condition is `False`.

        Returns:
            A new `BigInt` vector with elements selected from `true_case` or
            `false_case` based on the corresponding elements in `condition`.
        """
        var result = Self(unsafe_uninitialized=True)

        @parameter
        for i in range(Self.WORD_COUNT):
            result._storage[i] = condition.select(
                true_case._storage[i], false_case._storage[i]
            )

        return result

    # ===------------------------------------------------------------------=== #
    # Operator dunders
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __abs__(self) -> Self:
        """Performs the absolute value operation on a `BigInt` vector,
        element-wise.

        Returns:
            A new `BigInt` vector containing the absolute values of the
            original elements.
        """

        @parameter
        if signed:
            return Self.select(self.is_negative(), -self, self)
        else:
            return Self(self)

    @always_inline
    fn __neg__(self) -> Self:
        """Performs arithmetic negation on a `BigInt` vector, element-wise.

        It is only applicable to signed `BigInt` vectors.

        Returns:
            A new `BigInt` vector representing the result of the negation.
        """
        constrained[signed, "argument must be a signed `BigInt` vector"]()

        var result = ~self
        result += 1
        return result

    @always_inline
    fn __pos__(self) -> Self:
        """Performs the unary plus operation on a `BigInt` vector, element-wise.

        Returns:
            A new `BigInt` vector that is identical to the original.
        """
        return Self(self)

    @always_inline
    fn __add__(self, rhs: Self) -> Self:
        """Performs addition between two `BigInt` vectors, element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.

        Returns:
            A new `BigInt` vector containing the result of the addition.
        """
        var result = Self(self)

        _ = _apply_binop_with_carry[_uadd_with_carry](
            result._storage, rhs._storage
        )

        return result

    @always_inline
    fn __iadd__(inout self, rhs: Self):
        """Performs in-place addition between two `BigInt` vectors, element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.
        """
        _ = _apply_binop_with_carry[_uadd_with_carry](
            self._storage, rhs._storage
        )

    @always_inline
    fn __sub__(self, rhs: Self) -> Self:
        """Performs subtraction between two `BigInt` vectors, element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.

        Returns:
            A new `BigInt` vector containing the result of the subtraction.
        """
        var result = Self(self)

        _ = _apply_binop_with_carry[_usub_with_carry](
            result._storage, rhs._storage
        )

        return result

    @always_inline
    fn __isub__(inout self, rhs: Self):
        """Performs in-place subtraction between two `BigInt` vectors,
        element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.
        """
        _ = _apply_binop_with_carry[_usub_with_carry](
            self._storage, rhs._storage
        )

    @always_inline
    fn __mul__(self, rhs: Self) -> Self:
        """Performs multiplication between two `BigInt` vectors, element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.

        Returns:
            A new `BigInt` vector containing the result of the multiplication.
        """
        var result = BigInt[
            2 * bits, size=size, signed=signed, word_type=word_type
        ](unsafe_uninitialized=True)

        _ = _umul_with_carry[word_type, size](
            result._storage, self._storage, rhs._storage
        )

        return Self(other=result, unsafe_unchecked=True)

    @always_inline
    fn __imul__(inout self, rhs: Self):
        """Performs in-place multiplication between two `BigInt` vectors,
        element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.
        """
        self = self * rhs

    @always_inline
    fn __and__(self, rhs: Self) -> Self:
        """Performs a bitwise AND operation between two `BigInt` vectors,
        element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.

        Returns:
            A new `BigInt` vector containing the result of the bitwise AND
            operation.
        """
        return _apply_binop[SIMD.__and__](self._storage, rhs._storage)

    @always_inline
    fn __iand__(inout self, rhs: Self):
        """Performs an in-place bitwise AND operation between two `BigInt`
        vectors, element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.
        """
        _apply_inplace_binop[SIMD.__and__](self._storage, rhs._storage)

    @always_inline
    fn __or__(self, rhs: Self) -> Self:
        """Performs a bitwise OR operation between two `BigInt` vectors,
        element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.

        Returns:
            A new `BigInt` vector containing the result of the bitwise OR
            operation.
        """
        return _apply_binop[SIMD.__or__](self._storage, rhs._storage)

    @always_inline
    fn __ior__(inout self, rhs: Self):
        """Performs an in-place bitwise OR operation between two `BigInt`
        vectors, element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.
        """
        _apply_inplace_binop[SIMD.__or__](self._storage, rhs._storage)

    @always_inline
    fn __xor__(self, rhs: Self) -> Self:
        """Performs a bitwise XOR operation between two `BigInt` vectors,
        element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.

        Returns:
            A new `BigInt` vector containing the result of the bitwise XOR
            operation.
        """
        return _apply_binop[SIMD.__xor__](self._storage, rhs._storage)

    @always_inline
    fn __ixor__(inout self, rhs: Self):
        """Performs an in-place bitwise XOR operation between two `BigInt`
        vectors, element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.
        """
        _apply_inplace_binop[SIMD.__xor__](self._storage, rhs._storage)

    @always_inline
    fn __invert__(self) -> Self:
        """Performs a bitwise NOT operation on a `BigInt` vector, element-wise.

        Returns:
            A new `BigInt` vector containing the result of the bitwise NOT
            operation.
        """
        var result = Self(unsafe_uninitialized=True)

        @parameter
        for i in range(Self.WORD_COUNT):
            result._storage[i] = ~self._storage[i]

        return result

    @always_inline
    fn __lshift__(self, offset: Int) -> Self:
        """Performs a bitwise left shift on a `BigInt` vector, element-wise.

        The bits that are shifted off the left end are discarded, including the
        sign bit, if present. And the bit positions vacated at the right end are
        filled with zeros.

        Args:
            offset: The number of bits to shift the vector by. Must be
                non-negative.

        Returns:
            A new `BigInt` vector containing the result of the bitwise left
            shift.
        """
        debug_assert(all(offset >= 0), "offset must be non-negative")

        return _shift[is_left_shift=True, treat_offset_as_int=True](
            self, offset
        )

    @always_inline
    fn __lshift__(self, offset: SIMD[DType.index, size]) -> Self:
        """Performs a bitwise left shift on a `BigInt` vector, element-wise.

        The bits that are shifted off the left end are discarded, including the
        sign bit, if present. And the bit positions vacated at the right end are
        filled with zeros.

        Args:
            offset: The number of bits to shift the vector by. Must be
                non-negative.

        Returns:
            A new `BigInt` vector containing the result of the bitwise left
            shift.
        """
        debug_assert(all(offset >= 0), "offset must be non-negative")

        return _shift[is_left_shift=True](self, offset)

    @always_inline
    fn __ilshift__(inout self, offset: Int):
        """Performs an in-place bitwise left shift on a `BigInt` vector,
        element-wise.

        The bits that are shifted off the left end are discarded, including the
        sign bit, if present. And the bit positions vacated at the right end are
        filled with zeros.

        Args:
            offset: The number of bits to shift the vector by. Must be
                non-negative.
        """
        self = self << offset

    @always_inline
    fn __ilshift__(inout self, offset: SIMD[DType.index, size]):
        """Performs an in-place bitwise left shift on a `BigInt` vector,
        element-wise.

        The bits that are shifted off the left end are discarded, including the
        sign bit, if present. And the bit positions vacated at the right end are
        filled with zeros.

        Args:
            offset: The number of bits to shift the vector by. Must be
                non-negative.
        """
        self = self << offset

    @always_inline
    fn __rshift__(self, offset: Int) -> Self:
        """Performs a bitwise right shift on a `BigInt` vector, element-wise.

        The bits that are shifted off the right end are discarded, except for the
        sign bit, if present. And the bit positions vacated at the left end are
        filled with zeros, if the `BigInt` is unsigned, or with the sign bit, if
        the `BigInt` is signed.

        Args:
            offset: The number of bits to shift the vector by. Must be
                non-negative.

        Returns:
            A new `BigInt` vector containing the result of the bitwise right
            shift.
        """
        debug_assert(all(offset >= 0), "offset must be non-negative")

        return _shift[is_left_shift=False, treat_offset_as_int=True](
            self, offset
        )

    @always_inline
    fn __rshift__(self, offset: SIMD[DType.index, size]) -> Self:
        """Performs a bitwise right shift on a `BigInt` vector, element-wise.

        The bits that are shifted off the right end are discarded, except for the
        sign bit, if present. And the bit positions vacated at the left end are
        filled with zeros, if the `BigInt` is unsigned, or with the sign bit, if
        the `BigInt` is signed.

        Args:
            offset: The number of bits to shift the vector by. Must be
                non-negative.

        Returns:
            A new `BigInt` vector containing the result of the bitwise right
            shift.
        """
        debug_assert(all(offset >= 0), "offset must be non-negative")

        return _shift[is_left_shift=False](self, offset)

    @always_inline
    fn __irshift__(inout self, offset: Int):
        """Performs an in-place bitwise right shift on a `BigInt` vector,
        element-wise.

        The bits that are shifted off the right end are discarded, except for the
        sign bit, if present. And the bit positions vacated at the left end are
        filled with zeros, if the `BigInt` is unsigned, or with the sign bit, if
        the `BigInt` is signed.

        Args:
            offset: The number of bits to shift the vector by. Must be
                non-negative.
        """
        self = self >> offset

    @always_inline
    fn __irshift__(inout self, offset: SIMD[DType.index, size]):
        """Performs an in-place bitwise right shift on a `BigInt` vector,
        element-wise.

        The bits that are shifted off the right end are discarded, except for the
        sign bit, if present. And the bit positions vacated at the left end are
        filled with zeros, if the `BigInt` is unsigned, or with the sign bit, if
        the `BigInt` is signed.

        Args:
            offset: The number of bits to shift the vector by. Must be
                non-negative.
        """
        self = self >> offset

    @always_inline
    fn __eq__(self, rhs: Self) -> SIMD[DType.bool, size]:
        """Compares two `BigInt` vectors for equality, element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.

        Returns:
            A SIMD vector of booleans indicating whether the corresponding
            elements are equal.
        """
        return _compare(self, rhs) == 0

    @always_inline
    fn __ne__(self, rhs: Self) -> SIMD[DType.bool, size]:
        """Compares two `BigInt` vectors for inequality, element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.

        Returns:
            A SIMD vector of booleans indicating whether the corresponding
            elements are not equal.
        """
        return _compare(self, rhs) != 0

    @always_inline
    fn __lt__(self, rhs: Self) -> SIMD[DType.bool, size]:
        """Compares two `BigInt` vectors for less-than, element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.

        Returns:
            A SIMD vector of booleans indicating whether the corresponding
            elements are less than those of the right-hand side.
        """
        return _compare(self, rhs) == -1

    @always_inline
    fn __le__(self, rhs: Self) -> SIMD[DType.bool, size]:
        """Compares two `BigInt` vectors for less-than-or-equal, element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.

        Returns:
            A SIMD vector of booleans indicating whether the corresponding
            elements are less than or equal to those of the right-hand side.
        """
        return _compare(self, rhs) != 1

    @always_inline
    fn __gt__(self, rhs: Self) -> SIMD[DType.bool, size]:
        """Compares two `BigInt` vectors for greater-than, element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.

        Returns:
            A SIMD vector of booleans indicating whether the corresponding
            elements are greater than those of the right-hand side.
        """
        return _compare(self, rhs) == 1

    @always_inline
    fn __ge__(self, rhs: Self) -> SIMD[DType.bool, size]:
        """Compares two `BigInt` vectors for greater-than-or-equal, element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.

        Returns:
            A SIMD vector of booleans indicating whether the corresponding
            elements are greater than or equal to those of the right-hand side.
        """
        return _compare(self, rhs) != -1

    # ===------------------------------------------------------------------=== #
    # Methods
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn iadd_with_overflow(inout self, rhs: Self) -> SIMD[DType.bool, size]:
        """Performs in-place addition with overflow detection, element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.

        Returns:
            A SIMD vector of booleans indicating whether an overflow occurred
            during the addition.
        """

        @parameter
        if signed:
            var lhs_msb = self.get_most_significant_bit()
            var rhs_msb = rhs.get_most_significant_bit()

            _ = _apply_binop_with_carry[_uadd_with_carry](
                self._storage, rhs._storage
            )

            return (lhs_msb == rhs_msb) & (
                lhs_msb != self.get_most_significant_bit()
            )
        else:
            var carry_out = _apply_binop_with_carry[_uadd_with_carry](
                self._storage, rhs._storage
            )
            return carry_out.cast[DType.bool]()

    @always_inline
    fn isub_with_overflow(inout self, rhs: Self) -> SIMD[DType.bool, size]:
        """Performs in-place subtraction with overflow detection, element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.

        Returns:
            A SIMD vector of booleans indicating whether an overflow occurred
            during the subtraction.
        """

        @parameter
        if signed:
            var lhs_msb = self.get_most_significant_bit()
            var rhs_msb = rhs.get_most_significant_bit()

            _ = _apply_binop_with_carry[_usub_with_carry](
                self._storage, rhs._storage
            )

            return (lhs_msb != rhs_msb) & (
                lhs_msb != self.get_most_significant_bit()
            )
        else:
            var carry_out = _apply_binop_with_carry[_usub_with_carry](
                self._storage, rhs._storage
            )
            return carry_out.cast[DType.bool]()

    @always_inline
    fn full_mul(
        self, rhs: Self
    ) -> BigInt[2 * bits, size=size, signed=signed, word_type=word_type]:
        """Performs full multiplication between two `BigInt` vectors,
        element-wise.

        By construction, this operation is safe from overflow.

        Args:
            rhs: The right-hand side `BigInt` vector.

        Returns:
            A new `BigInt` vector containing the result of the full
            multiplication.
        """
        alias ResultType = BigInt[
            2 * bits, size=size, signed=signed, word_type=word_type
        ]

        @parameter
        if signed:
            alias ExtendedInputType = BigInt[
                2 * bits,
                size=size,
                signed=True,
                word_type=word_type,
            ]
            alias ExtendedResultType = BigInt[
                4 * bits,
                size=size,
                signed=True,
                word_type=word_type,
            ]

            var extended_lhs = ExtendedInputType(other=self)
            var extended_rhs = ExtendedInputType(other=rhs)
            var extended_result = ExtendedResultType(unsafe_uninitialized=True)

            _ = _umul_with_carry[word_type, size](
                extended_result._storage,
                extended_lhs._storage,
                extended_rhs._storage,
            )

            return ResultType(other=extended_result, unsafe_unchecked=True)
        else:
            var result = ResultType(unsafe_uninitialized=True)

            _ = _umul_with_carry[word_type, size](
                result._storage, self._storage, rhs._storage
            )

            return result

    @always_inline
    fn approx_mul_high(self, rhs: Self) -> Self:
        """Approximates the higher part of the result of the full multiplication
        between two unsigned `BigInt` vectors, element-wise.

        The normal multiplication returns the `bits` least significant bits of
        the result of the full multiplication. This method approximates the most
        significant `bits` of the same result, with an error bounded by:

        `0 <= (a.full_mul(b) >> bits) - a.approx_mul_high(b) <= WORD_COUNT - 1`

        An example usage of this is to quickly, but less accurately, compute the
        product of normalized mantissas of two floating-point numbers.

        Args:
            rhs: The right-hand side `BigInt` vector.

        Returns:
            A new `BigInt` vector containing the approximated higher part of the
            result of the full multiplication.
        """
        constrained[not signed, "arguments must be unsigned `BigInt` vectors"]()

        var result = Self(unsafe_uninitialized=True)

        _approx_umul_high(result._storage, self._storage, rhs._storage)

        return result

    @always_inline
    fn cast[type: DType](self) -> SIMD[type, size]:
        """Casts the `BigInt` vector to a SIMD vector with the same size.

        Parameters:
            type: The type of the SIMD vector to cast to. Constraints: Must be
                an integral type.

        Returns:
            A SIMD vector containing the values from the `BigInt` vector, with
            the specified type.
        """
        constrained[type.is_integral(), "type must be an integral type"]()

        alias TYPE_BITWIDTH = type.bitwidth()
        alias MAX_COUNT = _conditional[
            TYPE_BITWIDTH > bits,
            Self.WORD_COUNT,
            TYPE_BITWIDTH // Self.WORD_TYPE_BITWIDTH,
        ]()

        var result = self._storage[0].cast[type]()

        @parameter
        if TYPE_BITWIDTH <= Self.WORD_TYPE_BITWIDTH:
            return result
        else:

            @parameter
            for i in range(1, MAX_COUNT):
                result += self._storage[i].cast[type]() << (
                    Self.WORD_TYPE_BITWIDTH * i
                )

            @parameter
            if signed and TYPE_BITWIDTH > bits:
                alias MASK = ~Scalar[type](0) << bits
                return self.is_negative().select(result | MASK, result)
            else:
                return result

    @always_inline
    fn is_negative(self) -> SIMD[DType.bool, size]:
        """Checks if the `BigInt` vector is negative, element-wise.

        Returns:
            A SIMD vector of booleans indicating whether each element in the
            `BigInt` vector is negative.
        """
        return self.get_most_significant_bit() & signed

    @always_inline
    fn is_zero(self) -> SIMD[DType.bool, size]:
        """Checks if the `BigInt` vector is zero, element-wise.

        Returns:
            A SIMD vector of booleans indicating whether each element in the
            `BigInt` vector is zero.
        """
        var result = SIMD[DType.bool, size](True)

        @parameter
        for i in range(Self.WORD_COUNT):
            result &= self._storage[i] == 0

            if not any(result):
                break

        return result

    @always_inline
    fn count_leading_zeros(self) -> SIMD[word_type, size]:
        """Counts the number of leading zeros in each element of the `BigInt`
        vector.

        Returns:
            A SIMD vector containing the count of leading zeros for each element
            in the `BigInt` vector.
        """
        return _count_leading_zeros(self._storage)

    @always_inline
    fn get_most_significant_bit(self) -> SIMD[DType.bool, size]:
        """Gets the most significant bit in each element of the `BigInt` vector.

        Returns:
            A SIMD vector containing the most significant bit for each element
            in the `BigInt` vector.
        """
        var msb = self._storage[Self.WORD_COUNT - 1] >> (
            Self.WORD_TYPE_BITWIDTH - 1
        )
        return msb.cast[DType.bool]()

    @always_inline
    fn set_most_significant_bit(inout self):
        """Sets the most significant bit of the `BigInt` vector, element-wise.
        """
        self._storage[Self.WORD_COUNT - 1] |= _mask_leading_ones[word_type, 1]()

    @always_inline
    fn clear_most_significant_bit(inout self):
        """Clears the most significant bit of the `BigInt` vector, element-wise.
        """
        self._storage[Self.WORD_COUNT - 1] &= _mask_trailing_ones[
            word_type, Self.WORD_TYPE_BITWIDTH - 1
        ]()
