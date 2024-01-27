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

"""Tests for polynomial utilities."""

from specials._internal import polynomial as P
from specials._internal.testing import UnitTest


fn test_chebyshev() raises:
    let unit_test = UnitTest("test_chebyshev")

    let p = P.Chebyshev[4, DType.float64, 1].from_coefficients[4.0, 3.0, 2.0, 1.0]()

    unit_test.assert_equal(p.degree(), 3)
    unit_test.assert_equal(p.get[0](), 4.0)
    unit_test.assert_equal(p.get[1](), 3.0)
    unit_test.assert_equal(p.get[2](), 2.0)
    unit_test.assert_equal(p.get[3](), 1.0)

    let p_truncated = p.truncate[1]()

    unit_test.assert_equal(p_truncated.degree(), 0)
    unit_test.assert_equal(p_truncated.get[0](), 4.0)


fn test_chebyshev_hex[dtype: DType]() raises:
    let unit_test = UnitTest("test_polynomial_hex" + str(dtype))

    let p: P.Chebyshev[2, dtype, 1]

    @parameter
    if dtype == DType.float32:
        p = P.Chebyshev[2, dtype, 1].from_hexadecimal_coefficients[
            0x3F80_0000,
            0x3F00_0000,
        ]()
    else:
        p = P.Chebyshev[2, dtype, 1].from_hexadecimal_coefficients[
            0x3FF00000_00000000,
            0x3FE00000_00000000,
        ]()

    unit_test.assert_equal(p.degree(), 1)
    unit_test.assert_equal(p.get[0](), 1.0)
    unit_test.assert_equal(p.get[1](), 0.5)


fn test_chebyshev_economize() raises:
    let unit_test = UnitTest("test_chebyshev_economize")

    alias p = P.Chebyshev[4, DType.float64, 1].from_coefficients[4.0, 3.0, 2.0, 1.0]()
    let num_terms = p.economize[3.0]()

    unit_test.assert_equal(num_terms, 2)


fn test_chebyshev_evaluate() raises:
    let unit_test = UnitTest("test_chebyshev_evaluate")

    let p = P.Chebyshev[4, DType.float64, 4].from_coefficients[4.0, 3.0, 2.0, 1.0]()
    let x = SIMD[DType.float64, 4](-1.0, -0.5, 0.5, 1.0)

    # Expected values computed using `numpy.polynomial.Chebyshev`.
    let expected = SIMD[DType.float64, 4](2.0, 2.5, 3.5, 10.0)
    let actual = p(x)

    unit_test.assert_all_close(actual, expected, 0.0, 1e-12)


fn test_polynomial() raises:
    let unit_test = UnitTest("test_polynomial")

    let p = P.Polynomial[4, DType.float64, 1].from_coefficients[4.0, 3.0, 2.0, 1.0]()

    unit_test.assert_equal(p.degree(), 3)
    unit_test.assert_equal(p.get[0](), 4.0)
    unit_test.assert_equal(p.get[1](), 3.0)
    unit_test.assert_equal(p.get[2](), 2.0)
    unit_test.assert_equal(p.get[3](), 1.0)

    let p_truncated = p.truncate[1]()

    unit_test.assert_equal(p_truncated.degree(), 0)
    unit_test.assert_equal(p_truncated.get[0](), 4.0)


fn test_polynomial_hex[dtype: DType]() raises:
    let unit_test = UnitTest("test_polynomial_hex" + str(dtype))

    let p: P.Polynomial[2, dtype, 1]

    @parameter
    if dtype == DType.float32:
        p = P.Polynomial[2, dtype, 1].from_hexadecimal_coefficients[
            0x3F80_0000,
            0x3F00_0000,
        ]()
    else:
        p = P.Polynomial[2, dtype, 1].from_hexadecimal_coefficients[
            0x3FF00000_00000000,
            0x3FE00000_00000000,
        ]()

    unit_test.assert_equal(p.degree(), 1)
    unit_test.assert_equal(p.get[0](), 1.0)
    unit_test.assert_equal(p.get[1](), 0.5)


fn test_polynomial_evaluate() raises:
    let unit_test = UnitTest("test_polynomial_evaluate")

    let p = P.Polynomial[4, DType.float64, 4].from_coefficients[4.0, 3.0, 2.0, 1.0]()
    let x = SIMD[DType.float64, 4](-10.0, -2.5, 0.0, 1.0)

    # Expected values computed using `numpy.polynomial.Polynomial`.
    let expected = SIMD[DType.float64, 4](-826.0, -6.625, 4.0, 10.0)
    let actual = p(x)

    unit_test.assert_all_close(actual, expected, 0.0, 1e-12)


fn main() raises:
    # Chebyshev Series
    test_chebyshev()
    test_chebyshev_hex[DType.float32]()
    test_chebyshev_hex[DType.float64]()
    test_chebyshev_economize()
    test_chebyshev_evaluate()

    # Power Series
    test_polynomial()
    test_polynomial_hex[DType.float32]()
    test_polynomial_hex[DType.float64]()
    test_polynomial_evaluate()
