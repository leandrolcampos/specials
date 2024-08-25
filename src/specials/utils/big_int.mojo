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
        lhs.type.is_unsigned(),
        "argument type must be an unsigned, integral type",
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
        "argument type must be an unsigned, integral type",
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
        "argument type must be an unsigned, integral type",
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
        "argument type must be an unsigned, integral type",
    ]()

    var diff_and_carry0 = _usub_with_overflow(lhs, rhs)
    var diff_and_carry1 = _usub_with_overflow(diff_and_carry0[0], carry_in)
    var carry_out = diff_and_carry0[1] | diff_and_carry1[1]

    return diff_and_carry1[0], carry_out


@always_inline
fn _inplace_binop[
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

    var carry_out = SIMD[type, size](0)

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
    type: DType, size: Int
](val: InlineArray[SIMD[type, size], _]) -> SIMD[type, size]:
    """Counts the leading zeros in the internal representation of a `BigInt`."""
    constrained[type.is_integral(), "type must be an integral type"]()

    alias TYPE_BITWIDTH = type.bitwidth()

    var result = SIMD[type, size](0)
    var should_stop = SIMD[DType.bool, size](False)

    @parameter
    for i in reversed(range(val.size)):
        var bit_count = countl_zero(val[i])
        result += should_stop.select(0, bit_count)
        should_stop |= bit_count < TYPE_BITWIDTH

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
fn _gather[
    type: DType,
    size: Int,
](
    ptr: DTypePointer[type, *_],
    offset: SIMD[_, size],
    mask: SIMD[DType.bool, size] = True,
    default: SIMD[type, size] = 0,
) -> SIMD[type, size]:
    """Gathers values from memory using a SIMD vector of offsets."""
    constrained[
        offset.type.is_integral(), "offset type must be an integral type"
    ]()

    alias SHIFT = _iota[offset.type, size]()

    return ptr.gather(offset.fma(size, SHIFT), mask, default)


@always_inline
fn _shift[
    type: DType, size: Int, *, is_left_shift: Bool
](
    *,
    val: InlineArray[SIMD[type, size], _],
    offset: SIMD[type, size],
    is_negative: SIMD[DType.bool, size],
    inout dst: __type_of(val),
):
    """Performs a bitwise shift on the internal representation of a `BigInt`."""
    constrained[type.is_unsigned(), "type must be an unsigned, integral type"]()

    alias TYPE_BITWIDTH = type.bitwidth()

    var val_ptr = DTypePointer(val.unsafe_ptr().bitcast[Scalar[type]]())

    if all(offset == 0):

        @parameter
        for i in range(dst.size):
            dst[i] = val[i]

        return

    @always_inline
    @parameter
    fn at(index: SIMD[DType.index, _]) -> __type_of(index):
        @parameter
        if is_left_shift:
            return dst.size - index - 1
        else:
            return index

    @always_inline
    @parameter
    fn safe_get(index: SIMD[DType.index, size]) -> SIMD[type, size]:
        var is_index_below_size = index < dst.size
        var mask = (index >= 0) & is_index_below_size
        var default = (is_negative & ~is_index_below_size).select(
            SIMD[type, size](-1), 0
        )

        return _gather(val_ptr, index, mask, default)

    var index_offset = offset.cast[DType.index]() // TYPE_BITWIDTH
    var bit_offset = offset % TYPE_BITWIDTH

    @parameter
    for i in range(dst.size):
        var index = Scalar[DType.index](i)
        var part1 = safe_get(at(index + index_offset))
        var part2 = safe_get(at(index + index_offset + 1))

        @parameter
        if is_left_shift:
            dst[int(at(index))] = (bit_offset == 0).select(
                part1,
                part1 << bit_offset | part2 >> (TYPE_BITWIDTH - bit_offset),
            )
        else:
            dst[int(at(index))] = (bit_offset == 0).select(
                part1,
                part1 >> bit_offset | part2 << (TYPE_BITWIDTH - bit_offset),
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
fn _compare(lhs: BigInt, rhs: __type_of(lhs)) -> SIMD[DType.int8, lhs.size]:
    """Compares two `BigInt` values element-wise."""
    alias ONE = SIMD[DType.int8, lhs.size](1)

    var result = SIMD[DType.int8, lhs.size](0)

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
        word_type.is_unsigned(), "word type must be an unsigned, integral type"
    ]()
    constrained[
        bits % word_type.bitwidth() == 0,
        "number of bits must be a multiple of the word type's bitwidth",
    ]()


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
](Copyable, ExplicitlyCopyable, Movable):
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

    alias WORD_TYPE_BITWIDTH = word_type.bitwidth()
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
            self._storage.unsafe_ptr().bitcast[Scalar[word_type]](), BLOCK_SIZE
        )

    @always_inline
    fn __init__(inout self, *, unsafe_uninitialized: Bool):
        """Initializes the `BigInt` vector with uninitialized storage.

        Args:
            unsafe_uninitialized: A boolean indicating whether the internal
                storage should be left uninitialized. In practice, it is always
                set to `True` (it is not actually used inside the constructor).
        """
        _big_int_construction_checks[bits, word_type]()

        _ = unsafe_uninitialized
        self._storage = Self.StorageType(unsafe_uninitialized=True)

    @always_inline
    fn __init__(inout self, value: Int):
        """Initializes the `BigInt` vector with the provided integer value.

        Args:
            value: The integer value to set for each element in the `BigInt`
                vector.
        """
        self.__init__(SIMD[DType.index, size](value))

    @always_inline
    fn __init__(inout self, value: SIMD[_, size]):
        """Initializes the `BigInt` vector with the provided SIMD vector.

        Constraints:
            The value type must be an integral type.

        Args:
            value: The SIMD vector to initialize the `BigInt` vector with.
        """
        constrained[
            value.type.is_integral(),
            "value type must be an integral type",
        ]()
        _big_int_construction_checks[bits, word_type]()

        alias TYPE_BITWIDTH = value.type.bitwidth()

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
            if TYPE_BITWIDTH > Self.WORD_TYPE_BITWIDTH:
                tmp >>= Self.WORD_TYPE_BITWIDTH
            else:
                tmp = 0

    @always_inline
    fn __init__(inout self, other: Self):
        """Initializes a new `BigInt` vector by copying an existing `BigInt`.

        Args:
            other: The `BigInt` vector to copy from.
        """
        self.__copyinit__(other)

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

    # ===------------------------------------------------------------------=== #
    # Operator dunders
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __add__(self, rhs: Self) -> Self:
        """Performs addition between two `BigInt` vectors, element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.

        Returns:
            A new `BigInt` vector containing the result of the addition.
        """
        var result = Self(self)

        _ = _inplace_binop[_uadd_with_carry](result._storage, rhs._storage)

        return result

    @always_inline
    fn __iadd__(inout self, rhs: Self):
        """Performs in-place addition between two `BigInt` vectors, element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.
        """
        _ = _inplace_binop[_uadd_with_carry](self._storage, rhs._storage)

    @always_inline
    fn __sub__(self, rhs: Self) -> Self:
        """Performs subtraction between two `BigInt` vectors, element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.

        Returns:
            A new `BigInt` vector containing the result of the subtraction.
        """
        var result = Self(self)

        _ = _inplace_binop[_usub_with_carry](result._storage, rhs._storage)

        return result

    @always_inline
    fn __isub__(inout self, rhs: Self):
        """Performs in-place subtraction between two `BigInt` vectors,
        element-wise.

        Args:
            rhs: The right-hand side `BigInt` vector.
        """
        _ = _inplace_binop[_usub_with_carry](self._storage, rhs._storage)

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
    fn __lshift__(self, offset: SIMD[word_type, size]) -> Self:
        """Performs a bitwise left shift on a `BigInt` vector, element-wise.

        Args:
            offset: The number of bits to shift the vector by. Must be less than
                `bits`; otherwise, the behavior of this method is undefined.

        Returns:
            A new `BigInt` vector containing the result of the bitwise left
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
    fn __ilshift__(inout self, offset: SIMD[word_type, size]):
        """Performs an in-place bitwise left shift on a `BigInt` vector,
        element-wise.

        Args:
            offset: The number of bits to shift the vector by. Must be less than
                `bits`; otherwise, the behavior of this method is undefined.
        """
        self = self << offset

    @always_inline
    fn __rshift__(self, offset: SIMD[word_type, size]) -> Self:
        """Performs a bitwise right shift on a `BigInt` vector, element-wise.

        Args:
            offset: The number of bits to shift the vector by. Must be less than
                `bits`; otherwise, the behavior of this method is undefined.

        Returns:
            A new `BigInt` vector containing the result of the bitwise right
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
    fn __irshift__(inout self, offset: SIMD[word_type, size]):
        """Performs an in-place bitwise right shift on a `BigInt` vector,
        element-wise.

        Args:
            offset: The number of bits to shift the vector by. Must be less than
                `bits`; otherwise, the behavior of this method is undefined.
        """
        self = self >> offset

    @always_inline
    fn __neg__(self) -> Self:
        """Performs arithmetic negation on a `BigInt` vector, element-wise.

        Returns:
            A new `BigInt` vector representing the result of the negation.
        """
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

    # ===------------------------------------------------------------------=== #
    # Methods
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn add_with_overflow(inout self, rhs: Self) -> SIMD[DType.bool, size]:
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
    fn sub_with_overflow(inout self, rhs: Self) -> SIMD[DType.bool, size]:
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
        bits, size = Self.size, signed=signed, word_type = Self.word_type
    ]:
        """Casts the `BigInt` vector to a new `BigInt` with a different number
        of bits and signedness.

        Parameters:
            bits: The number of bits for the new `BigInt`. Constraints: Must be
                a positive integer and a multiple of the bitwidth of word type.
            signed: A boolean indicating whether the new `BigInt` is signed
                (`True`) or unsigned (`False`).

        Returns:
            A new `BigInt` vector with the specified number of bits and
            signedness. The size and word type are preserved.
        """
        var result = BigInt[
            bits, size = Self.size, signed=signed, word_type = Self.word_type
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
    fn clear_most_significant_bit(inout self):
        """Clears the most significant bit of the `BigInt` vector, element-wise.
        """
        self._storage[Self.WORD_COUNT - 1] &= _mask_trailing_ones[
            word_type, Self.WORD_TYPE_BITWIDTH - 1
        ]()

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
    fn count_leading_zeros(self) -> SIMD[word_type, size]:
        """Counts the number of leading zeros in each element of the `BigInt`
        vector.

        Returns:
            A SIMD vector containing the count of leading zeros for each element
            in the `BigInt` vector.
        """
        return _count_leading_zeros(self._storage)

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
