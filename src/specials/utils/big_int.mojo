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

"""Implements the `BigInt` struct.

Provides the implementation of the `BigInt` struct, enabling arbitrary, fixed
bit-size integer arithmetic with SIMD support.
"""

from bit import countl_zero
from memory import DTypePointer, memset_zero
from sys.info import is_64bit
from sys.intrinsics import _RegisterPackType
from utils import InlineArray
from utils.numerics import max_finite


# ===----------------------------------------------------------------------=== #
# Operations with carry propagation
# ===----------------------------------------------------------------------=== #


@always_inline
fn _uadd_with_overflow(
    lhs: SIMD, rhs: __type_of(lhs)
) -> (__type_of(lhs), __type_of(lhs)):
    """Performs unsigned addition with overflow detection."""
    constrained[
        lhs.type.is_unsigned(), "argument type must be an unsigned integer"
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
        lhs.type.is_unsigned(), "argument type must be an unsigned integer"
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
        lhs.type.is_unsigned(), "argument type must be an unsigned integer"
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
        lhs.type.is_unsigned(), "argument type must be an unsigned integer"
    ]()

    var diff_and_carry0 = _usub_with_overflow(lhs, rhs)
    var diff_and_carry1 = _usub_with_overflow(diff_and_carry0[0], carry_in)
    var carry_out = diff_and_carry0[1] | diff_and_carry1[1]

    return diff_and_carry1[0], carry_out


@always_inline
fn _inplace_binop[
    binop_with_carry: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width], SIMD[type, width]
    ) -> (SIMD[type, width], SIMD[type, width]),
    type: DType,
    width: Int,
](
    inout dst: InlineArray[SIMD[type, width], _],
    rhs: InlineArray[SIMD[type, width], _],
) -> SIMD[type, width]:
    """Performs an in-place binary operation with carry propagation."""
    constrained[
        dst.size >= rhs.size, "dst must have at least as many elements as rhs"
    ]()

    var carry_out = SIMD[type, width](0)

    @parameter
    for i in range(dst.size):
        var has_rhs_value = i < rhs.size
        var rhs_value = rhs[i] if has_rhs_value else 0
        var carry_in = carry_out
        var result_and_carry = binop_with_carry(dst[i], rhs_value, carry_in)

        dst[i] = result_and_carry[0]
        carry_out = result_and_carry[1]

        # Stop early when rhs is over and there is no carry out to propagate.
        if all(carry_out == 0) and not has_rhs_value:
            break

    return carry_out


# ===----------------------------------------------------------------------=== #
# Operations for bit manipulation
# ===----------------------------------------------------------------------=== #


@always_inline
fn _count_leading_zeros[
    type: DType, width: Int
](val: InlineArray[SIMD[type, width], _]) -> SIMD[type, width]:
    """Counts the leading zeros in the internal representation of a `BigInt`."""
    constrained[type.is_integral(), "type must be an integer"]()

    alias TYPE_SIZE = type.bitwidth()

    var result = SIMD[type, width](0)
    var should_stop = SIMD[DType.bool, width](False)

    @parameter
    for i in reversed(range(val.size)):
        var bit_count = countl_zero(val[i])
        result += should_stop.select(0, bit_count)
        should_stop |= bit_count < TYPE_SIZE

        if all(should_stop):
            break

    return result


@always_inline
fn _mask_leading_ones[word_type: DType, count: Int]() -> Scalar[word_type]:
    """Creates a mask with the specified number of leading ones."""
    return ~_mask_trailing_ones[word_type, word_type.bitwidth() - count]()


@always_inline
fn _mask_trailing_ones[word_type: DType, count: Int]() -> Scalar[word_type]:
    """Creates a mask with the specified number of trailing ones."""
    constrained[
        word_type.is_unsigned(), "word type must be an unsigned integer"
    ]()
    constrained[count >= 0, "count must be non-negative"]()
    constrained[
        count <= word_type.bitwidth(),
        "count must be less than or equal to word size",
    ]()

    @parameter
    if count == 0:
        return Scalar[word_type](0)
    else:
        return ~Scalar[word_type](0) >> (word_type.bitwidth() - count)


@always_inline
fn _iota[type: DType, width: Int]() -> SIMD[type, width]:
    """Creates a SIMD vector containing an increasing sequence starting from 0.
    """
    # Unlike `math.iota`, this function can be executed at compile-time.
    var result = SIMD[type, width]()

    @parameter
    for i in range(width):
        result[i] = i

    return result


@always_inline
fn _gather[
    type: DType,
    width: Int,
](
    ptr: DTypePointer[type, *_],
    offset: SIMD[_, width],
    mask: SIMD[DType.bool, width] = True,
    default: SIMD[type, width] = 0,
) -> SIMD[type, width]:
    """Gathers values from memory using a SIMD vector of offsets."""
    constrained[
        offset.type.is_integral(), "offset type must be an integral type"
    ]()

    alias SHIFT = _iota[offset.type, width]()

    return ptr.gather(offset.fma(width, SHIFT), mask, default)


@always_inline
fn _shift[
    type: DType, width: Int, *, is_left_shift: Bool
](
    *,
    val: InlineArray[SIMD[type, width], _],
    offset: SIMD[type, width],
    is_negative: SIMD[DType.bool, width],
    inout dst: __type_of(val),
):
    """Performs a bitwise shift on the internal representation of a `BigInt`."""
    constrained[type.is_unsigned(), "type must be an unsigned integer"]()

    alias TYPE_SIZE = type.bitwidth()

    var val_ptr = DTypePointer(val.unsafe_ptr().bitcast[Scalar[type]]())

    if all(offset == 0):

        @parameter
        for i in range(dst.size):
            dst[i] = val[i]

        return

    @parameter
    fn at(index: SIMD[DType.index, _]) -> __type_of(index):
        @parameter
        if is_left_shift:
            return dst.size - index - 1
        else:
            return index

    @parameter
    fn safe_get(index: SIMD[DType.index, width]) -> SIMD[type, width]:
        var is_index_below_size = index < dst.size
        var mask = (index >= 0) & is_index_below_size
        var default = (is_negative & ~is_index_below_size).select(
            SIMD[type, width](-1), 0
        )

        return _gather(val_ptr, index, mask, default)

    var index_offset = offset.cast[DType.index]() // TYPE_SIZE
    var bit_offset = offset % TYPE_SIZE

    @parameter
    for i in range(dst.size):
        var index = Scalar[DType.index](i)
        var part1 = safe_get(at(index + index_offset))
        var part2 = safe_get(at(index + index_offset + 1))

        @parameter
        if is_left_shift:
            dst[int(at(index))] = (bit_offset == 0).select(
                part1, part1 << bit_offset | part2 >> (TYPE_SIZE - bit_offset)
            )
        else:
            dst[int(at(index))] = (bit_offset == 0).select(
                part1, part1 >> bit_offset | part2 << (TYPE_SIZE - bit_offset)
            )


# ===----------------------------------------------------------------------=== #
# Operations for comparison
# ===----------------------------------------------------------------------=== #


@always_inline
fn _compare(lhs: SIMD, rhs: __type_of(lhs)) -> SIMD[DType.int8, lhs.size]:
    """Compares two SIMD vectors element-wise."""
    alias ONE = SIMD[DType.int8, lhs.size](1)

    return (lhs == rhs).select(0, (lhs < rhs).select(-1, ONE))


@always_inline
fn _compare(lhs: BigInt, rhs: __type_of(lhs)) -> SIMD[DType.int8, lhs.width]:
    """Compares two `BigInt` values element-wise."""
    alias ONE = SIMD[DType.int8, lhs.width](1)

    var result = SIMD[DType.int8, lhs.width](0)

    @parameter
    if lhs.signed:
        var lhs_is_negative = lhs.is_negative()
        var rhs_is_negative = rhs.is_negative()
        result = (lhs_is_negative != rhs_is_negative).select(
            lhs_is_negative.select(-1, ONE), result
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
# BigInt
# ===----------------------------------------------------------------------=== #


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
    constrained[bits > 0, "number of bits must be positive"]()
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
fn _big_int_construction_checks[
    bits: Int,
    word_type: DType,
]():
    """Performs checks on the parameters of a `BigInt` constructor."""
    constrained[bits > 0, "number of bits must be positive"]()
    constrained[
        word_type.is_unsigned(), "word type must be an unsigned integer"
    ]()
    constrained[
        bits % word_type.bitwidth() == 0,
        "number of bits must be a multiple of word size",
    ]()


alias BigUInt = BigInt[_, width=_, signed=False, word_type=_]
"""Represents an arbitrary, fixed bit-size unsigned integer."""


@value
struct BigInt[
    bits: Int,
    /,
    *,
    width: Int,
    signed: Bool = True,
    word_type: DType = _default_word_type[bits](),
](Copyable, ExplicitlyCopyable, Movable):
    """Represents an arbitrary, fixed bit-size integer.

    It can represent both signed and unsigned integers with a fixed number of
    bits and leverages SIMD operations for enhanced performance.

    Constraints:
        The number of bits must be a multiple of the word size in bits, which
        is determined by the `word_type`.

    Parameters:
        bits: The number of bits that the `BigInt` can represent. Constraints:
            Must be positive.
        width: The SIMD width, representing the number of lanes for parallel
            processing. Constraints: Must be positive and a power of two.
        signed: A boolean indicating whether the integer is signed (`True`) or
            unsigned (`False`). Defaults to `True`.
        word_type: The type of the words used to represent the integer. For
            performance reasons, defaults to the largest unsigned integer type
            whose size in bits divides `bits` evenly.
    """

    # ===------------------------------------------------------------------=== #
    # Aliases
    # ===------------------------------------------------------------------=== #

    alias WORD_SIZE = word_type.bitwidth()
    """The word size in bits."""

    alias WORD_COUNT = bits // Self.WORD_SIZE
    """The number of words used to represent the integer."""

    alias StorageType = InlineArray[SIMD[word_type, width], Self.WORD_COUNT]
    """The type used to represent the integer internally."""

    # ===------------------------------------------------------------------=== #
    # Fields
    # ===------------------------------------------------------------------=== #

    var _storage: Self.StorageType
    """The internal storage of the integer."""

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __init__(inout self):
        """Initializes the `BigInt` with zeros."""
        _big_int_construction_checks[bits, word_type]()

        alias BLOCK_SIZE = width * Self.WORD_COUNT

        self._storage = Self.StorageType(unsafe_uninitialized=True)

        memset_zero[word_type](
            self._storage.unsafe_ptr().bitcast[Scalar[word_type]](), BLOCK_SIZE
        )

    @always_inline
    fn __init__(inout self, *, unsafe_uninitialized: Bool):
        """Initializes the `BigInt` with uninitialized internal storage.

        Args:
            unsafe_uninitialized: A boolean indicating whether the internal
                storage should be left uninitialized. Always set to `True`
                (it's not actually used inside the constructor).
        """
        _big_int_construction_checks[bits, word_type]()

        _ = unsafe_uninitialized
        self._storage = Self.StorageType(unsafe_uninitialized=True)

    @always_inline
    fn __init__(inout self, value: Int):
        """Initializes the `BigInt` with the provided integer value.

        Args:
            value: The integer value to initialize the `BigInt` with.
        """
        self.__init__(SIMD[DType.index, width](value))

    @always_inline
    fn __init__(inout self, value: SIMD[_, width]):
        """Initializes the `BigInt` with the provided SIMD vector.

        Args:
            value: The SIMD vector to initialize the `BigInt` with.
        """
        constrained[
            value.type.is_integral(),
            "value type must be an integral type",
        ]()
        _big_int_construction_checks[bits, word_type]()

        alias TYPE_SIZE = value.type.bitwidth()

        var extension = ((value < 0) & signed).select(
            max_finite[word_type](), 0
        )
        var tmp = value

        self._storage = Self.StorageType(unsafe_uninitialized=True)

        @parameter
        for i in range(Self.WORD_COUNT):
            self._storage[i] = (tmp == 0).select(
                extension, tmp.cast[word_type]()
            )

            @parameter
            if TYPE_SIZE > Self.WORD_SIZE:
                tmp >>= Self.WORD_SIZE
            else:
                tmp = 0

    @always_inline
    fn __init__(inout self, other: Self):
        """Initializes a new `BigInt` by copying the provided `BigInt` value.

        Args:
            other: The `BigInt` value to copy from.
        """
        self.__copyinit__(other)

    # ===------------------------------------------------------------------=== #
    # Factory methods
    # ===------------------------------------------------------------------=== #

    @staticmethod
    @always_inline
    fn all_ones() -> Self:
        """Creates a `BigInt` with all bits set to one.

        Returns:
            A new `BigInt` value with all bits set to one.
        """
        return ~Self()

    @staticmethod
    @always_inline
    fn max() -> Self:
        """Creates a `BigInt` with the maximum representable value.

        Returns:
            A new `BigInt` value containing the maximum representable value.
        """
        var result = Self.all_ones()

        @parameter
        if signed:
            result.clear_most_significant_bit()

        return result

    @staticmethod
    @always_inline
    fn min() -> Self:
        """Creates a `BigInt` with the minimum representable value.

        Returns:
            A new `BigInt` value containing the minimum representable value.
        """
        var result = Self()

        @parameter
        if signed:
            result.set_most_significant_bit()

        return result

    @staticmethod
    @always_inline
    fn one() -> Self:
        """Creates a `BigInt` with the value one.

        Returns:
            A new `BigInt` value containing the value one.
        """
        return Self(1)

    @staticmethod
    @always_inline
    fn zero() -> Self:
        """Creates a `BigInt` with the value zero.

        Returns:
            A new `BigInt` value containing the value zero.
        """
        return Self()

    # ===------------------------------------------------------------------=== #
    # Operator dunders
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __add__(self, rhs: Self) -> Self:
        """Performs addition between two `BigInt` values, element-wise.

        Args:
            rhs: The right-hand side `BigInt` value.

        Returns:
            A new `BigInt` value containing the result of the addition.
        """
        var result = Self(self)

        _ = _inplace_binop[_uadd_with_carry](result._storage, rhs._storage)

        return result

    @always_inline
    fn __iadd__(inout self, rhs: Self):
        """Performs in-place addition between two `BigInt` values, element-wise.

        Args:
            rhs: The right-hand side `BigInt` value.
        """
        _ = _inplace_binop[_uadd_with_carry](self._storage, rhs._storage)

    @always_inline
    fn __sub__(self, rhs: Self) -> Self:
        """Performs subtraction between two `BigInt` values, element-wise.

        Args:
            rhs: The right-hand side `BigInt` value.

        Returns:
            A new `BigInt` value containing the result of the subtraction.
        """
        var result = Self(self)

        _ = _inplace_binop[_usub_with_carry](result._storage, rhs._storage)

        return result

    @always_inline
    fn __isub__(inout self, rhs: Self):
        """Performs in-place subtraction between two `BigInt` values,
        element-wise.

        Args:
            rhs: The right-hand side `BigInt` value.
        """
        _ = _inplace_binop[_usub_with_carry](self._storage, rhs._storage)

    @always_inline
    fn __eq__(self, rhs: Self) -> SIMD[DType.bool, width]:
        """Compares two `BigInt` values for equality, element-wise.

        Args:
            rhs: The right-hand side `BigInt` value.

        Returns:
            A SIMD vector of booleans indicating whether the corresponding
            elements are equal.
        """
        return _compare(self, rhs) == 0

    @always_inline
    fn __ne__(self, rhs: Self) -> SIMD[DType.bool, width]:
        """Compares two `BigInt` values for inequality, element-wise.

        Args:
            rhs: The right-hand side `BigInt` value.

        Returns:
            A SIMD vector of booleans indicating whether the corresponding
            elements are not equal.
        """
        return _compare(self, rhs) != 0

    @always_inline
    fn __lt__(self, rhs: Self) -> SIMD[DType.bool, width]:
        """Compares two `BigInt` values for less-than, element-wise.

        Args:
            rhs: The right-hand side `BigInt` value.

        Returns:
            A SIMD vector of booleans indicating whether the corresponding
            elements are less than those of the right-hand side.
        """
        return _compare(self, rhs) == -1

    @always_inline
    fn __le__(self, rhs: Self) -> SIMD[DType.bool, width]:
        """Compares two `BigInt` values for less-than-or-equal, element-wise.

        Args:
            rhs: The right-hand side `BigInt` value.

        Returns:
            A SIMD vector of booleans indicating whether the corresponding
            elements are less than or equal to those of the right-hand side.
        """
        return _compare(self, rhs) != 1

    @always_inline
    fn __gt__(self, rhs: Self) -> SIMD[DType.bool, width]:
        """Compares two `BigInt` values for greater-than, element-wise.

        Args:
            rhs: The right-hand side `BigInt` value.

        Returns:
            A SIMD vector of booleans indicating whether the corresponding
            elements are greater than those of the right-hand side.
        """
        return _compare(self, rhs) == 1

    @always_inline
    fn __ge__(self, rhs: Self) -> SIMD[DType.bool, width]:
        """Compares two `BigInt` values for greater-than-or-equal, element-wise.

        Args:
            rhs: The right-hand side `BigInt` value.

        Returns:
            A SIMD vector of booleans indicating whether the corresponding
            elements are greater than or equal to those of the right-hand side.
        """
        return _compare(self, rhs) != -1

    @always_inline
    fn __invert__(self) -> Self:
        """Performs bitwise inversion on a `BigInt` value.

        Returns:
            A new `BigInt` value containing the result of the bitwise inversion.
        """
        var result = Self(unsafe_uninitialized=True)

        @parameter
        for i in range(Self.WORD_COUNT):
            result._storage[i] = ~self._storage[i]

        return result

    @always_inline
    fn __lshift__(self, offset: SIMD[word_type, width]) -> Self:
        """Performs a bitwise left shift on a `BigInt` value, element-wise.

        Args:
            offset: The number of bits to shift the value by.

        Returns:
            A new `BigInt` value containing the result of the bitwise left
            shift.
        """
        debug_assert(
            all((offset >= 0) & (offset < bits)),
            "offset must be within bounds",
        )

        var result = Self(unsafe_uninitialized=True)

        _shift[is_left_shift=True](
            val=self._storage,
            offset=offset,
            is_negative=self.is_negative(),
            dst=result._storage,
        )

        return result

    @always_inline
    fn __ilshift__(inout self, offset: SIMD[word_type, width]):
        """Performs an in-place bitwise left shift on a `BigInt` value,
        element-wise.

        Args:
            offset: The number of bits to shift the value by.
        """
        self = self << offset

    @always_inline
    fn __rshift__(self, offset: SIMD[word_type, width]) -> Self:
        """Performs a bitwise right shift on a `BigInt` value, element-wise.

        Args:
            offset: The number of bits to shift the value by.

        Returns:
            A new `BigInt` value containing the result of the bitwise right
            shift.
        """
        debug_assert(
            all((offset >= 0) & (offset < bits)),
            "offset must be within bounds",
        )

        var result = Self(unsafe_uninitialized=True)

        _shift[is_left_shift=False](
            val=self._storage,
            offset=offset,
            is_negative=self.is_negative(),
            dst=result._storage,
        )

        return result

    @always_inline
    fn __irshift__(inout self, offset: SIMD[word_type, width]):
        """Performs an in-place bitwise right shift on a `BigInt` value,
        element-wise.

        Args:
            offset: The number of bits to shift the value by.
        """
        self = self >> offset

    @always_inline
    fn __neg__(self) -> Self:
        """Performs arithmetic negation on a `BigInt` vector.

        Returns:
            A new `BigInt` vector representing the result of the negation.
        """
        var result = ~self
        result += 1
        return result

    @always_inline
    fn __pos__(self) -> Self:
        """Performs the unary plus operation on a `BigInt` vector.

        Returns:
            A new `BigInt` vector that is identical to the original.
        """
        return Self(self)

    # ===------------------------------------------------------------------=== #
    # Methods
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn add_with_overflow(inout self, rhs: Self) -> SIMD[DType.bool, width]:
        """Performs in-place addition with overflow detection, element-wise.

        Args:
            rhs: The right-hand side `BigInt` value.

        Returns:
            A SIMD vector of booleans indicating whether an overflow occurred
            during the addition.
        """

        @parameter
        if signed:
            var lhs_msb = self.get_most_significant_bit()
            var rhs_msb = rhs.get_most_significant_bit()

            _ = _inplace_binop[_uadd_with_carry](self._storage, rhs._storage)

            return (lhs_msb == rhs_msb) & (
                lhs_msb != self.get_most_significant_bit()
            )
        else:
            var carry_out = _inplace_binop[_uadd_with_carry](
                self._storage, rhs._storage
            )
            return carry_out.cast[DType.bool]()

    @always_inline
    fn sub_with_overflow(inout self, rhs: Self) -> SIMD[DType.bool, width]:
        """Performs in-place subtraction with overflow detection, element-wise.

        Args:
            rhs: The right-hand side `BigInt` value.

        Returns:
            A SIMD vector of booleans indicating whether an overflow occurred
            during the subtraction.
        """

        @parameter
        if signed:
            var lhs_msb = self.get_most_significant_bit()
            var rhs_msb = rhs.get_most_significant_bit()

            _ = _inplace_binop[_usub_with_carry](self._storage, rhs._storage)

            return (lhs_msb != rhs_msb) & (
                lhs_msb != self.get_most_significant_bit()
            )
        else:
            var carry_out = _inplace_binop[_usub_with_carry](
                self._storage, rhs._storage
            )
            return carry_out.cast[DType.bool]()

    @always_inline
    fn cast[
        bits: Int, /, *, signed: Bool
    ](self) -> BigInt[
        bits, width = Self.width, signed=signed, word_type = Self.word_type
    ]:
        """Casts the `BigInt` to a new `BigInt` with a different number of bits
        and signedness.

        Parameters:
            bits: The number of bits for the new `BigInt`. Constraints: Must be
                positive and a multiple of the word size in bits of the current
                `BigInt`.
            signed: A boolean indicating whether the new `BigInt` is signed
                (`True`) or unsigned (`False`).

        Returns:
            A new `BigInt` value with the specified number of bits and signedness.
            The SIMD width and word type are preserved.
        """
        var result = BigInt[
            bits, width = Self.width, signed=signed, word_type = Self.word_type
        ](unsafe_uninitialized=True)

        @parameter
        if bits <= self.bits:

            @parameter
            for i in range(result.WORD_COUNT):
                result._storage[i] = self._storage[i]

        else:
            var extension = (self.is_negative() & signed).select(
                max_finite[word_type](), 0
            )

            @parameter
            for i in range(result.WORD_COUNT):

                @parameter
                if i < self.WORD_COUNT:
                    result._storage[i] = self._storage[i]
                else:
                    result._storage[i] = extension

        return result

    @always_inline
    fn cast[type: DType](self) -> SIMD[type, width]:
        """Casts the `BigInt` to a SIMD vector with the same width.

        Parameters:
            type: The type of the SIMD vector to cast to.

        Returns:
            A new SIMD vector containing the values of the `BigInt`.
        """
        constrained[type.is_integral(), "type must be an integral type"]()

        alias TYPE_SIZE = type.bitwidth()
        alias MAX_COUNT = _conditional[
            TYPE_SIZE > bits, Self.WORD_COUNT, TYPE_SIZE // Self.WORD_SIZE
        ]()

        var result = self._storage[0].cast[type]()

        @parameter
        if TYPE_SIZE <= Self.WORD_SIZE:
            return result
        else:

            @parameter
            for i in range(1, MAX_COUNT):
                result += self._storage[i].cast[type]() << (Self.WORD_SIZE * i)

            @parameter
            if signed and TYPE_SIZE > bits:
                alias MASK = ~Scalar[type](0) << bits
                return self.is_negative().select(result | MASK, result)
            else:
                return result

    @always_inline
    fn clear_most_significant_bit(inout self):
        """Clears the most significant bit of the `BigInt`."""
        self._storage[Self.WORD_COUNT - 1] &= _mask_trailing_ones[
            word_type, Self.WORD_SIZE - 1
        ]()

    @always_inline
    fn get_most_significant_bit(self) -> SIMD[DType.bool, width]:
        """Gets the per-element most significant bit of the `BigInt`.

        Returns:
            A SIMD vector containing the most significant bit for each element
            in the `BigInt`.
        """
        var msb = self._storage[Self.WORD_COUNT - 1] >> (Self.WORD_SIZE - 1)
        return msb.cast[DType.bool]()

    @always_inline
    fn set_most_significant_bit(inout self):
        """Sets the most significant bit of the `BigInt`."""
        self._storage[Self.WORD_COUNT - 1] |= _mask_leading_ones[word_type, 1]()

    @always_inline
    fn count_leading_zeros(self) -> SIMD[word_type, width]:
        """Counts the per-element number of leading zeros in the the `BigInt`.

        Returns:
            A SIMD vector containing the number of leading zeros for each element
            in the `BigInt`.
        """
        return _count_leading_zeros(self._storage)

    @always_inline
    fn is_negative(self) -> SIMD[DType.bool, width]:
        """Checks if each element of the `BigInt` is negative.

        Returns:
            A SIMD vector of booleans indicating whether each element is negative.
        """
        return self.get_most_significant_bit() & signed

    @always_inline
    fn is_zero(self) -> SIMD[DType.bool, width]:
        """Checks if each element of the `BigInt` is zero.

        Returns:
            A SIMD vector of booleans indicating whether each element is zero.
        """
        var result = SIMD[DType.bool, width](True)

        @parameter
        for i in range(Self.WORD_COUNT):
            result &= self._storage[i] == 0

            if not any(result):
                break

        return result
