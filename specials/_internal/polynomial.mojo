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
# Some of the code in this file is adapted from:
#
# numpy/numpy
# Copyright (c) 2005-2023, NumPy Developers.
# Licensed under BSD 3 clause.
#
# slatec/fnlib (https://www.netlib.org/slatec/fnlib):
# Public-domain software. No copyright restrictions.

"""Provides utilities for efficiently working with polynomials."""

import math
from memory.unsafe import bitcast
from utils.static_tuple import StaticTuple

from specials._internal import asserting
from specials._internal.functional import fori_loop
from specials._internal.limits import FloatLimits
from specials._internal.table import get_hexadecimal_dtype

# TODO: Consider using a trait when it supports defining default method implementations.


@always_inline("nodebug")
fn _check_polynomial_like_constraints[
    num_terms: Int, dtype: DType, simd_width: Int, hexadecimal_dtype: DType
]() -> None:
    """Checks the constraints of a polynomial-like."""
    asserting.assert_positive["num_terms", num_terms]()
    asserting.assert_float_dtype["dtype", dtype]()
    asserting.assert_simd_width["simd_width", simd_width]()

    @parameter
    if dtype == DType.float32:
        constrained[
            hexadecimal_dtype == DType.uint32,
            "The parameter `hexadecimal_dtype` must be equal to `DType.uint32` when "
            + "`dtype` is `DType.float32`.",
        ]()
    else:  # dtype == DType.float64
        constrained[
            hexadecimal_dtype == DType.uint64,
            "The parameter `hexadecimal_dtype` must be equal to `DType.uint64` when "
            + "`dtype` is `DType.float64`.",
        ]()


# ===-------------------------- Chebyshev Series --------------------------=== #


@register_passable("trivial")
struct Chebyshev[
    num_terms: Int,
    dtype: DType,
    simd_width: Int,
    hexadecimal_dtype: DType = get_hexadecimal_dtype[dtype](),
]:
    """Represents a finite Chebyshev series.

    A Chebyshev series with `n + 1` terms is a polynomial of the form

    `p(x) = c[0] * T[0](x) + c[1] * T[1](x) + ... + c[n] * T[n](x)`

    where `x` is the independent variable defined in the interval `[-1, 1]`, `c[i]` is
    the `i`-th coefficient of the series, and `T[i]` is the `i`-th Chebyshev polynomial
    of the first kind.

    Parameters:
        num_terms: The number of terms in the series.
        dtype: The data type of the independent variable `x` in the series.
        simd_width: The SIMD width of the independent variable `x` in the series.
        hexadecimal_dtype: The data type used for hexadecimal representation of the
            coefficients. The default is automatically determined based on `dtype`.

    Constraints:
        The number of terms must be positive. The parameter `dtype` must be `float32`
        or `float64`. The parameter `hexadecimal_dtype` must be `uint32` if `dtype` is
        `float32` or `uint64` if `dtype` is `float64`.
    """

    var _coefficients: StaticTuple[num_terms, SIMD[dtype, simd_width]]

    @staticmethod
    fn from_coefficients[*coefficients: SIMD[dtype, 1]]() -> Self:
        """Generates a Chebyshev series from a sequence of coefficients.

        Parameters:
            coefficients: The sequence of coefficients in order of increasing degree,
                i.e., `(1, 2, 3)` gives `1 * T[0](x) + 2 * T[1](x) + 3 * T[2](x)`.

        Returns:
            A Chebyshev series with the given coefficients.

        Constraints:
            The number of coefficients must be equal to the parameter `num_terms`.
        """
        _check_polynomial_like_constraints[
            num_terms, dtype, simd_width, hexadecimal_dtype
        ]()

        constrained[
            num_terms == len(VariadicList(coefficients)),
            "The number of coefficients must be equal to the parameter `num_terms`.",
        ]()

        var splatted_coefficients = StaticTuple[num_terms, SIMD[dtype, simd_width]]()

        for i in range(num_terms):
            splatted_coefficients[i] = coefficients[i]

        return Self {_coefficients: splatted_coefficients}

    @staticmethod
    fn from_hexadecimal_coefficients[
        *coefficients: SIMD[hexadecimal_dtype, 1]
    ]() -> Self:
        """Generates a Chebyshev series from a sequence of hexadecimal coefficients.

        Parameters:
            coefficients: The sequence of hexadecimal coefficients in order of increasing
                degree, i.e., `(1, 2, 3)` gives `1 * T[0](x) + 2 * T[1](x) + 3 * T[2](x)`.

        Returns:
            A Chebyshev series with the given hexadecimal coefficients.

        Constraints:
            The number of hexadecimal coefficients must be equal to the parameter
            `num_terms`.
        """
        _check_polynomial_like_constraints[
            num_terms, dtype, simd_width, hexadecimal_dtype
        ]()

        constrained[
            num_terms == len(VariadicList(coefficients)),
            (
                "The number of hexadecimal coefficients must be equal to the parameter"
                " `num_terms`."
            ),
        ]()

        var splatted_coefficients = StaticTuple[num_terms, SIMD[dtype, simd_width]]()

        for i in range(num_terms):
            splatted_coefficients[i] = bitcast[dtype](coefficients[i])

        return Self {_coefficients: splatted_coefficients}

    @always_inline
    fn degree(self: Self) -> Int:
        """Returns the degree of the Chebyshev series."""
        return num_terms - 1

    @always_inline
    fn get[index: Int](self: Self) -> SIMD[dtype, simd_width]:
        """Returns the coefficient of the Chebyshev series at the given index.

        Parameters:
            index: The index of the coefficient to return.

        Returns:
            SIMD vector containing the coefficient of the Chebyshev series at the
            given index.

        Constraints:
            The index must be in the range `[0, num_terms)`.
        """
        asserting.assert_in_range["index", index, 0, num_terms]()
        return self._coefficients[index]

    fn truncate[num_terms: Int](self: Self) -> Chebyshev[num_terms, dtype, simd_width]:
        """Truncates the Chebyshev series by discarding high-degree terms.

        Parameters:
            num_terms: The number of terms in the truncated series.

        Returns:
            A truncated Chebyshev series.

        Constraints:
            The number of terms in the truncated series must be positive and less than
            or equal to the number of terms in the original series.
        """
        asserting.assert_in_range["num_terms", num_terms, 1, self.num_terms + 1]()
        var coefficients = StaticTuple[num_terms, SIMD[dtype, simd_width]]()

        @parameter
        fn body_func[i: Int]() -> None:
            coefficients[i] = self.get[i]()

        fori_loop[0, num_terms, 1, body_func]()

        return Chebyshev[num_terms, dtype, simd_width] {_coefficients: coefficients}

    fn __call__(self, x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
        """Evaluates the Chebyshev series at points `x` using the Clenshaw algorithm.

        If the series `p` has `n + 1` terms, this function returns the values:

        `p(x) = c[0] * T[0](x) + c[1] * T[1](x) + ... + c[n] * T[n](x)`.

        Args:
            x: The points at which to evaluate the Chebyshev series. These points should
                be in the interval `[-1, 1]`.

        Returns:
            SIMD vector containing the values of the Chebyshev series at points `x`.
        """
        alias nan: SIMD[dtype, simd_width] = math.nan[dtype]()

        var result = SIMD[dtype, simd_width](0.0)

        @parameter
        if num_terms == 1:
            result = self.get[0]()
        elif num_terms == 2:
            result = self.get[0]() + self.get[1]() * x
        else:
            let two_x = 2.0 * x
            var tmp = SIMD[dtype, simd_width](0.0)
            var c0 = self.get[num_terms - 2]()
            var c1 = self.get[num_terms - 1]()

            @parameter
            fn body_func[i: Int]() -> None:
                tmp = c0
                c0 = self.get[i]() - c1
                c1 = math.fma(c1, two_x, tmp)

            fori_loop[num_terms - 3, -1, -1, body_func]()

            result = c0 + c1 * x

        return math.select(math.abs(x) > 1.0, nan, result)

    fn economize[error_tolerance: SIMD[dtype, 1]](self) -> Int:
        """Economizes the Chebyshev series by minimizing the number of terms.

        Given a Chebyshev series `p` with `n` terms, this function returns the minimum
        number of terms `m` such that

        `|p(x) - p_m(x)| <= |c[m]| + ... + |c[n-1]| <= error_tolerance`

        holds for all `x` in the interval [-1, 1], where `p_m` is the Chebyshev series
        obtained by truncating `p` to `m <= n` terms.

        Parameters:
            error_tolerance: Tolerance for the approximation error between the original
                series and its truncated version with `m` terms.

        Returns:
            The minimum number of terms `m` required to ensure the approximation error
            between the original series and its truncated version with `m` terms is less
            than or equal to `error_tolerance`.

        Constraints:
            The error tolerance must be positive.
        """
        asserting.assert_positive[dtype, "error_tolerance", error_tolerance]()

        var num_terms_required = num_terms
        var error = SIMD[dtype, 1](0.0)

        @parameter
        fn body_func[i: Int]() -> Bool:
            # For any coefficient `c` of the Chebyshev series `p`, the following condition
            # holds: `c[0] == c[1] == ... == c[simd_width - 1]`.
            let value = self.get[i]()[0]
            # TODO: Use `math.abs` when the problem evaluating it in compile-time is fixed.
            # https://github.com/modularml/mojo/issues/1244
            if value < 0:
                error -= value
            else:
                error += value

            if error > error_tolerance:
                num_terms_required = i + 1
                return False

            return True

        fori_loop[num_terms - 1, 0, -1, body_func]()

        return num_terms_required


# ===---------------------------- Power Series ----------------------------=== #


@register_passable("trivial")
struct Polynomial[
    num_terms: Int,
    dtype: DType,
    simd_width: Int,
    hexadecimal_dtype: DType = get_hexadecimal_dtype[dtype](),
]:
    """Represents a finite Power series, commonly known as a polynomial.

    A Power series with `n + 1` terms is a polynomial of the form

    `p(x) = c[0] + c[1] * x + ... + c[n] * x**n`

    where `x` is the independent variable and `c[i]` is the `i`-th coefficient of the
    series.

    Parameters:
        num_terms: The number of terms in the series.
        dtype: The data type of the independent variable `x` in the series.
        simd_width: The SIMD width of the independent variable `x` in the series.
        hexadecimal_dtype: The data type used for hexadecimal representation of the
            coefficients. The default is automatically determined based on `dtype`.

    Constraints:
        The number of terms must be positive. The parameter `dtype` must be `float32`
        or `float64`. The parameter `hexadecimal_dtype` must be `uint32` if `dtype` is
        `float32` or `uint64` if `dtype` is `float64`.
    """

    var _coefficients: StaticTuple[num_terms, SIMD[dtype, simd_width]]

    @staticmethod
    fn from_coefficients[*coefficients: SIMD[dtype, 1]]() -> Self:
        """Generates a Power series from a sequence of coefficients.

        Parameters:
            coefficients: The sequence of coefficients in order of increasing degree,
                i.e., `(1, 2, 3)` gives `1 + 2 * x + 3 * x**2`.

        Returns:
            A Power series with the given coefficients.

        Constraints:
            The number of coefficients must be equal to the parameter `num_terms`.
        """
        _check_polynomial_like_constraints[
            num_terms, dtype, simd_width, hexadecimal_dtype
        ]()

        constrained[
            num_terms == len(VariadicList(coefficients)),
            "The number of coefficients must be equal to the parameter `num_terms`.",
        ]()

        var splatted_coefficients = StaticTuple[num_terms, SIMD[dtype, simd_width]]()

        for i in range(num_terms):
            splatted_coefficients[i] = coefficients[i]

        return Self {_coefficients: splatted_coefficients}

    @staticmethod
    fn from_hexadecimal_coefficients[
        *coefficients: SIMD[hexadecimal_dtype, 1]
    ]() -> Self:
        """Generates a Power series from a sequence of hexadecimal coefficients.

        Parameters:
            coefficients: The sequence of hexadecimal coefficients in order of
                increasing degree, i.e., `(1, 2, 3)` gives `1 + 2 * x + 3 * x**2`.

        Returns:
            A Power series with the given hexadecimal coefficients.

        Constraints:
            The number of hexadecimal coefficients must be equal to the parameter
            `num_terms`.
        """
        _check_polynomial_like_constraints[
            num_terms, dtype, simd_width, hexadecimal_dtype
        ]()

        constrained[
            num_terms == len(VariadicList(coefficients)),
            (
                "The number of hexadecimal coefficients must be equal to the parameter"
                " `num_terms`."
            ),
        ]()

        var splatted_coefficients = StaticTuple[num_terms, SIMD[dtype, simd_width]]()

        for i in range(num_terms):
            splatted_coefficients[i] = bitcast[dtype](coefficients[i])

        return Self {_coefficients: splatted_coefficients}

    @always_inline
    fn degree(self: Self) -> Int:
        """Returns the degree of the Power series."""
        return num_terms - 1

    @always_inline
    fn get[index: Int](self: Self) -> SIMD[dtype, simd_width]:
        """Returns the coefficient of the Power series at the given index.

        Parameters:
            index: The index of the coefficient to return.

        Returns:
            SIMD vector containing the coefficient of the Power series at the
            given index.

        Constraints:
            The index must be in the range `[0, num_terms)`.
        """
        asserting.assert_in_range["index", index, 0, num_terms]()
        return self._coefficients[index]

    fn truncate[num_terms: Int](self: Self) -> Polynomial[num_terms, dtype, simd_width]:
        """Truncates the Power series by discarding high-degree terms.

        Parameters:
            num_terms: The number of terms in the truncated series.

        Returns:
            A truncated Power series.

        Constraints:
            The number of terms in the truncated series must be positive and less than
            or equal to the number of terms in the original series.
        """
        asserting.assert_in_range["num_terms", num_terms, 1, self.num_terms + 1]()
        var coefficients = StaticTuple[num_terms, SIMD[dtype, simd_width]]()

        @parameter
        fn body_func[i: Int]() -> None:
            coefficients[i] = self.get[i]()

        fori_loop[0, num_terms, 1, body_func]()

        return Polynomial[num_terms, dtype, simd_width] {_coefficients: coefficients}

    fn __call__(self, x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
        """Evaluates the Power series at points `x` using the Horner's scheme.

        If the series `p` has `n + 1` terms, this function returns the values:

        `p(x) = c[0] + c[1] * x + ... + c[n] * x**n`.

        Args:
            x: The points at which to evaluate the Power series.

        Returns:
            SIMD vector containing the values of the Power series at points `x`.
        """
        var result = self.get[num_terms - 1]()

        @parameter
        if num_terms > 1:

            @parameter
            fn body_func[i: Int]() -> None:
                result = math.fma(result, x, self.get[i]())

            fori_loop[num_terms - 2, -1, -1, body_func]()

        return result
