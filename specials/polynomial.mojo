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
from utils.static_tuple import StaticTuple

from ._internal import asserting
from ._internal.functional import fori_loop
from ._internal.limits import FloatLimits


# ===----------------------- Chebyshev polynomials ------------------------=== #


@register_passable("trivial")
struct Chebyshev[num_terms: Int, dtype: DType, simd_width: Int]:
    """Represents a finite Chebyshev series.

    A `n + 1`-term Chebyshev series is a polynomial of the form

    `p(x) = c[0] * T[0](x) + c[1] * T[1](x) + ... + c[n] * T[n](x)`

    where `x` is the independent variable defined in the interval `[-1, 1]`, c[i] is the
    `i`-th coefficient of the series, and `T[i](x)` is the `i`-th Chebyshev polynomial
    of the first kind.

    Parameters:
        num_terms: The number of terms in the series.
        dtype: The data type of the independent variable `x` in the series (float32 or
            float64).
        simd_width: The SIMD width of the independent variable `x` in the series.

    Constraints:
        The number of terms must be positive. The data type must be a floating-point of
        single (float32) or double (float64) precision. The SIMD width must be positive
        and a power of two.
    """

    var _coefficients: StaticTuple[num_terms, SIMD[dtype, simd_width]]

    @staticmethod
    fn from_coefficients[*coefficients: FloatLiteral]() -> Self:
        """Generates a Chebyshev series from a sequence of coefficients.

        Parameters:
            coefficients: The sequence of coefficients in order of increasing degree,
                i.e., `(1, 2, 3)` gives `1 * T[0](x) + 2 * T[1](x) + 3 * T[2](x)`.

        Returns:
            A Chebyshev series with the given coefficients.

        Constraints:
            The number of coefficients must be equal to the parameter `num_terms`.
        """
        _check_chebyshev_constraints[num_terms, dtype, simd_width]()

        constrained[
            num_terms == len(VariadicList(coefficients)),
            "The number of coefficients must be equal to the parameter `num_terms`.",
        ]()

        var splatted_coefficients = StaticTuple[num_terms, SIMD[dtype, simd_width]]()

        for i in range(num_terms):
            splatted_coefficients[i] = coefficients[i]

        return Self {_coefficients: splatted_coefficients}

    fn degree(self: Self) -> Int:
        """Returns the degree of the Chebyshev series."""
        return num_terms - 1

    fn get[index: Int](self: Self) -> SIMD[dtype, simd_width]:
        """Returns the coefficient at the given index.

        Parameters:
            index: The index of the coefficient to return.

        Returns:
            The coefficient at the given index.

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

        for i in range(num_terms):
            coefficients[i] = self._coefficients[i]

        return Chebyshev[num_terms, dtype, simd_width] {_coefficients: coefficients}


@always_inline("nodebug")
fn _check_chebyshev_constraints[
    num_terms: Int, dtype: DType, simd_width: Int
]() -> None:
    """Checks the constraints of a Chebyshev series."""
    asserting.assert_positive["num_terms", num_terms]()
    asserting.assert_float_dtype["dtype", dtype]()
    asserting.assert_simd_width["simd_width", simd_width]()


@always_inline
fn _chebyshev_eval_impl[
    num_terms: Int, dtype: DType, simd_width: Int
](
    p: Chebyshev[num_terms, dtype, simd_width],
    x: SIMD[dtype, simd_width],
) -> SIMD[
    dtype, simd_width
]:
    """Implementation of `chebyshev_eval`."""
    alias eps: SIMD[dtype, simd_width] = FloatLimits[dtype].eps
    alias nan: SIMD[dtype, simd_width] = math.nan[dtype]()

    var result = SIMD[dtype, simd_width](0.0)

    @parameter
    if num_terms == 1:
        result = p.get[0]()
    elif num_terms == 2:
        result = p.get[0]() + p.get[1]() * x
    else:
        let two_x = 2.0 * x
        var tmp = SIMD[dtype, simd_width](0.0)
        var c0 = p.get[num_terms - 2]()
        var c1 = p.get[num_terms - 1]()

        @parameter
        fn body_func[i: Int]() -> None:
            tmp = c0
            c0 = p.get[i]() - c1
            c1 = math.fma(c1, two_x, tmp)

        fori_loop[num_terms - 3, -1, -1, body_func]()

        result = c0 + c1 * x

    return math.select(math.abs(x) < 1.0 + eps, result, nan)


fn chebyshev_eval[
    num_terms: Int, dtype: DType, simd_width: Int
](
    p: Chebyshev[num_terms, dtype, simd_width],
    x: SIMD[dtype, simd_width],
) -> SIMD[
    dtype, simd_width
]:
    """Evaluates the Chebyshev series `p` at points `x`.

    If the series `p` has `n + 1` terms, this function returns the values:

    `p(x) = c[0] * T[0](x) + c[1] * T[1](x) + ... + c[n] * T[n](x)`.

    Args:
        p: The Chebyshev series to evaluate.
        x: The points at which to evaluate the Chebyshev series. These points should be
            in the interval `[-1, 1]`.

    Returns:
        The values of the Chebyshev series `p` at points `x`.
    """
    return _chebyshev_eval_impl(p, x)


@always_inline
fn _chebyshev_init_impl[
    num_terms: Int,
    dtype: DType,
    simd_width: Int,
    p: Chebyshev[num_terms, dtype, simd_width],
    requested_accuracy: SIMD[dtype, 1],
]() -> Int:
    """Implementation of `chebyshev_init`."""
    asserting.assert_positive[dtype, "requested_accuracy", requested_accuracy]()

    var num_terms_required = num_terms
    var error = SIMD[dtype, 1](0.0)
    var stop = False

    @parameter
    fn body_func[i: Int]() -> None:
        # For any coefficient `c` of the Chebyshev series `p`, the following condition
        # holds: `c[0] == c[1] == ... == c[simd_width - 1]`.
        let value = p.get[i]()[0]
        # TODO: Use `math.abs` when the problem evaluating it in compile-time is fixed.
        # https://github.com/modularml/mojo/issues/1244
        if value < 0:
            error -= value
        else:
            error += value

        if error > requested_accuracy and not stop:
            num_terms_required = i + 1
            stop = True

    fori_loop[num_terms - 1, 0, -1, body_func]()

    return num_terms_required


fn chebyshev_init[
    num_terms: Int,
    dtype: DType,
    simd_width: Int,
    p: Chebyshev[num_terms, dtype, simd_width],
    requested_accuracy: SIMD[dtype, 1],
]() -> Int:
    """Initializes a Chebyshev series `p` known at compile time.

    Let `p` be a `n`-term Chebyshev series. This function returns the minimum number of
    terms `m` of this series such that

    `abs(p(x) - p_m(x)) <= abs(c[m]) + ... + abs(c[n-1]) <= requested_accuracy`

    for all `x` in the interval `[-1, 1]`, where `p_m = p.truncate[m]()`.

    Parameters:
        num_terms: The number of terms in the original series.
        dtype: The data type of the independent variable `x` in the series (float32 or
            float64).
        simd_width: The SIMD width of the independent variable `x` in the series.
        p: The Chebyshev series to initialize.
        requested_accuracy: The requested accuracy.

    Returns:
        The minimum number of terms required to ensure the requested accuracy.

    Constraints:
        The requested accuracy must be positive.
    """
    return _chebyshev_init_impl[num_terms, dtype, simd_width, p, requested_accuracy]()
