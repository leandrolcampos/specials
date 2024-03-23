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

"""Provides utilities for table-based numerical methods."""

from memory.unsafe import bitcast
from utils.static_tuple import StaticTuple

from specials._internal import asserting


@always_inline
fn get_hexadecimal_dtype[decimal_dtype: DType]() -> DType:
    """Returns the hexadecimal dtype corresponding to the provided decimal dtype."""

    @parameter
    if decimal_dtype == DType.float32:
        return DType.uint32
    elif decimal_dtype == DType.float64:
        return DType.uint64
    else:
        return DType.invalid


@always_inline
fn _check_float_table_constraints[size: Int, dtype: DType]() -> None:
    """Checks the constraints of the `FloatTable`."""
    asserting.assert_positive["size", size]()
    asserting.assert_float_dtype["dtype", dtype]()


@register_passable("trivial")
struct FloatTable[size: Int, dtype: DType](Sized):
    """Represents a table of floating-point values.

    It is used to implement table lookup algorithms.

    Parameters:
        size: The number of floating-point values in the table.
        dtype: The data type of the floating-point values.

    Constraints:
        The size must be positive. The parameter `dtype` must be `float32` or `float64`.
    """

    var _data: StaticTuple[size, Scalar[dtype]]

    @staticmethod
    fn from_values[*values: Scalar[dtype]]() -> Self:
        """Creates a table from a sequence of floating-point values.

        Parameters:
            values: The sequence of floating-point values.

        Returns:
            A `FloatTable` with the given floating-point values.

        Constraints:
            The number of values must be equal to the parameter `size`.
        """
        _check_float_table_constraints[size, dtype]()

        constrained[
            size == len(VariadicList(values)),
            "The number of values must be equal to the parameter `size`.",
        ]()

        return Self {_data: StaticTuple[size, Scalar[dtype]](values)}

    @staticmethod
    fn from_hexadecimal_values[
        *values: Scalar[get_hexadecimal_dtype[dtype]()]
    ]() -> Self:
        """Creates a table from a sequence of hexadecimal floating-point values.

        The data type used for hexadecimal representation of the floating-point values
        is automatically determined based on `dtype`: `uint32` if `dtype` is `float32`
        or `uint64` if `dtype` is `float64`.

        Parameters:
            values: The sequence of hexadecimal floating-point values.

        Returns:
            A `FloatTable` with the given hexadecimal floating-point values.

        Constraints:
            The number of hexadecimal values must be equal to the parameter `size`.
        """
        _check_float_table_constraints[size, dtype]()

        constrained[
            size == len(VariadicList(values)),
            "The number of hexadecimal values must be equal to the parameter `size`.",
        ]()

        var data = StaticTuple[size, Scalar[dtype]]()

        for i in range(size):
            data[i] = bitcast[dtype](values[i])

        return Self {_data: data}

    @always_inline
    fn __len__(self: Self) -> Int:
        """Returns the number of floating-point values in the table.

        This is known at compile time.

        Returns:
            The number of floating-point values in the table.
        """
        return size

    @always_inline
    fn get[index: Int](self: Self) -> Scalar[dtype]:
        """Returns the floating-point value of the table at the given index.

        Parameters:
            index: The index of the floating-point value to return.

        Returns:
            SIMD vector containing the floating-point value of the table at the given
            index.

        Constraints:
            The index must be in the range `[0, size)`.
        """
        asserting.assert_in_range["index", index, 0, size]()
        return self._data[index]

    @always_inline
    fn unsafe_lookup(self: Self, index: SIMD) -> SIMD[dtype, index.size]:
        """Returns the floating-point values of the table at the given indices.

        For performance reasons, this method does not perform bounds checking.

        Args:
            index: SIMD vector containing the indices of the floating-point values
                to return.

        Returns:
            SIMD vector containing the floating-point values of the table at the
            given indices.

        Constraints:
            The parameter `index.type` must be an integer.
        """
        # TODO: Use the overload of `DTypePointer.simd_load` when it is available.
        # This overload will allow to load a SIMD vector of floating-point values
        # from a SIMD vector of indices. It may require to change the underlying
        # data storage to a `DTypePointer`. See the feature request:
        # https://github.com/modularml/mojo/issues/1626

        asserting.assert_integral_dtype["index.type", index.type]()
        var result = SIMD[dtype, index.size]()

        @unroll
        for i in range(index.size):
            result[i] = self._data[int(index[i])]

        return result

    fn lookup(self: Self, index: SIMD) -> SIMD[dtype, index.size]:
        """Returns the floating-point values of the table at the given indices.

        Args:
            index: SIMD vector containing the indices of the floating-point values
                to return.

        Returns:
            SIMD vector containing the floating-point values of the table at the
            given indices. If an index is out of range, the corresponding result
            is `NaN`.

        Constraints:
            The parameter `index.type` must be an integer.
        """
        asserting.assert_integral_dtype["index.type", index.type]()

        var is_safe = (index >= 0) & (index < size)
        var safe_index = is_safe.select(index, 0)

        return is_safe.select(self.unsafe_lookup(safe_index), math.nan[dtype]())
