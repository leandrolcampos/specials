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

from sys.info import simdwidthof

from specials._internal import polynomial as P
from specials._internal.testing import UnitTest


fn test_chebyshev[dtype: DType]() raises:
    let unit_test = UnitTest("test_chebyshev_" + str(dtype))

    alias simd_width = simdwidthof[dtype]()
    let p = P.Chebyshev[4, dtype, simd_width].from_coefficients[4.0, 3.0, 2.0, 1.0]()

    unit_test.assert_equal(len(p), 4)
    unit_test.assert_equal(p.degree(), 3)
    unit_test.assert_equal(p.get[0](), SIMD[dtype, simd_width](4.0))
    unit_test.assert_equal(p.get[1](), 3.0)
    unit_test.assert_equal(p.get[2](), 2.0)
    unit_test.assert_equal(p.get[3](), 1.0)


fn test_chebyshev_small[dtype: DType]() raises:
    let unit_test = UnitTest("test_chebyshev_small_" + str(dtype))

    alias simd_width = simdwidthof[dtype]()
    let p = P.Chebyshev[1, dtype, simd_width].from_coefficients[1.0]()

    unit_test.assert_equal(len(p), 1)
    unit_test.assert_equal(p.degree(), 0)
    unit_test.assert_equal(p.get[0](), SIMD[dtype, simd_width](1.0))


fn test_chebyshev_hexadecimal[dtype: DType]() raises:
    let unit_test = UnitTest("test_chebyshev_hexadecimal_" + str(dtype))

    alias simd_width = simdwidthof[dtype]()
    let p: P.Chebyshev[2, dtype, simd_width]

    @parameter
    if dtype == DType.float32:
        p = P.Chebyshev[2, dtype, simd_width].from_hexadecimal_coefficients[
            0x3F80_0000,
            0x3F00_0000,
        ]()
    else:
        p = P.Chebyshev[2, dtype, simd_width].from_hexadecimal_coefficients[
            0x3FF00000_00000000,
            0x3FE00000_00000000,
        ]()

    unit_test.assert_equal(len(p), 2)
    unit_test.assert_equal(p.degree(), 1)
    unit_test.assert_equal(p.get[0](), SIMD[dtype, simd_width](1.0))
    unit_test.assert_equal(p.get[1](), 0.5)


fn test_chebyshev_truncate[dtype: DType]() raises:
    let unit_test = UnitTest("test_chebyshev_truncate_" + str(dtype))

    alias simd_width = simdwidthof[dtype]()
    let p = P.Chebyshev[4, dtype, simd_width].from_coefficients[4.0, 3.0, 2.0, 1.0]()

    let q = p.truncate[1]()

    unit_test.assert_equal(len(q), 1)
    unit_test.assert_equal(q.degree(), 0)
    unit_test.assert_equal(q.get[0](), SIMD[dtype, simd_width](4.0))

    let r = p.truncate[4]()

    unit_test.assert_equal(len(r), 4)
    unit_test.assert_equal(r.degree(), 3)
    unit_test.assert_equal(r.get[0](), SIMD[dtype, simd_width](4.0))
    unit_test.assert_equal(r.get[1](), 3.0)
    unit_test.assert_equal(r.get[2](), 2.0)
    unit_test.assert_equal(r.get[3](), 1.0)


fn test_chebyshev_economize[dtype: DType]() raises:
    let unit_test = UnitTest("test_chebyshev_economize_" + str(dtype))

    alias simd_width = simdwidthof[dtype]()
    alias p = P.Chebyshev[4, dtype, simd_width].from_coefficients[4.0, 2.0, 1.0, 0.5]()
    alias num_terms = p.economize[1.5]()

    unit_test.assert_equal(num_terms, 2)

    let q = p.truncate[num_terms]()

    unit_test.assert_equal(len(q), 2)
    unit_test.assert_equal(q.degree(), 1)
    unit_test.assert_equal(q.get[0](), SIMD[dtype, simd_width](4.0))
    unit_test.assert_equal(q.get[1](), 2.0)


fn test_chebyshev_evaluate[dtype: DType]() raises:
    let unit_test = UnitTest("test_chebyshev_evaluate_" + str(dtype))

    alias simd_width = 4
    let p = P.Chebyshev[4, dtype, simd_width].from_coefficients[4.0, 3.0, 2.0, 1.0]()
    let x = SIMD[dtype, simd_width](-1.0, -0.5, 0.5, 1.0)

    # Expected values computed using `numpy.polynomial.Chebyshev`.
    let expected = SIMD[dtype, simd_width](2.0, 2.5, 3.5, 10.0)
    let actual = p(x)

    unit_test.assert_all_close(actual, expected, 0.0, 1e-12)


fn test_polynomial[dtype: DType]() raises:
    let unit_test = UnitTest("test_polynomial_" + str(dtype))

    alias simd_width = simdwidthof[dtype]()
    let p = P.Polynomial[4, dtype, simd_width].from_coefficients[4.0, 3.0, 2.0, 1.0]()

    unit_test.assert_equal(len(p), 4)
    unit_test.assert_equal(p.degree(), 3)
    unit_test.assert_equal(p.get[0](), SIMD[dtype, simd_width](4.0))
    unit_test.assert_equal(p.get[1](), 3.0)
    unit_test.assert_equal(p.get[2](), 2.0)
    unit_test.assert_equal(p.get[3](), 1.0)


fn test_polynomial_small[dtype: DType]() raises:
    let unit_test = UnitTest("test_polynomial_small_" + str(dtype))

    alias simd_width = simdwidthof[dtype]()
    let p = P.Polynomial[1, dtype, simd_width].from_coefficients[1.0]()

    unit_test.assert_equal(len(p), 1)
    unit_test.assert_equal(p.degree(), 0)
    unit_test.assert_equal(p.get[0](), SIMD[dtype, simd_width](1.0))


fn test_polynomial_hexadecimal[dtype: DType]() raises:
    let unit_test = UnitTest("test_polynomial_hexadecimal" + str(dtype))

    alias simd_width = simdwidthof[dtype]()
    let p: P.Polynomial[2, dtype, simd_width]

    @parameter
    if dtype == DType.float32:
        p = P.Polynomial[2, dtype, simd_width].from_hexadecimal_coefficients[
            0x3F80_0000,
            0x3F00_0000,
        ]()
    else:
        p = P.Polynomial[2, dtype, simd_width].from_hexadecimal_coefficients[
            0x3FF00000_00000000,
            0x3FE00000_00000000,
        ]()

    unit_test.assert_equal(len(p), 2)
    unit_test.assert_equal(p.degree(), 1)
    unit_test.assert_equal(p.get[0](), SIMD[dtype, simd_width](1.0))
    unit_test.assert_equal(p.get[1](), 0.5)


fn test_polynomial_truncate[dtype: DType]() raises:
    let unit_test = UnitTest("test_polynomial_truncate_" + str(dtype))

    alias simd_width = simdwidthof[dtype]()
    let p = P.Polynomial[4, dtype, simd_width].from_coefficients[4.0, 3.0, 2.0, 1.0]()

    let q = p.truncate[1]()

    unit_test.assert_equal(len(q), 1)
    unit_test.assert_equal(q.degree(), 0)
    unit_test.assert_equal(q.get[0](), SIMD[dtype, simd_width](4.0))

    let r = p.truncate[4]()

    unit_test.assert_equal(len(r), 4)
    unit_test.assert_equal(r.degree(), 3)
    unit_test.assert_equal(r.get[0](), SIMD[dtype, simd_width](4.0))
    unit_test.assert_equal(r.get[1](), 3.0)
    unit_test.assert_equal(r.get[2](), 2.0)
    unit_test.assert_equal(r.get[3](), 1.0)


fn test_polynomial_evaluate[dtype: DType]() raises:
    let unit_test = UnitTest("test_polynomial_evaluate_" + str(dtype))

    alias simd_width = 4
    let p = P.Polynomial[4, dtype, simd_width].from_coefficients[4.0, 3.0, 2.0, 1.0]()
    let x = SIMD[dtype, simd_width](-10.0, -2.5, 0.0, 1.0)

    # Expected values computed using `numpy.polynomial.Polynomial`.
    let expected = SIMD[dtype, simd_width](-826.0, -6.625, 4.0, 10.0)
    let actual = p(x)

    unit_test.assert_all_close(actual, expected, 0.0, 1e-12)


fn main() raises:
    # Chebyshev Series
    test_chebyshev[DType.float32]()
    test_chebyshev[DType.float64]()

    test_chebyshev_small[DType.float32]()
    test_chebyshev_small[DType.float64]()

    test_chebyshev_hexadecimal[DType.float32]()
    test_chebyshev_hexadecimal[DType.float64]()

    test_chebyshev_truncate[DType.float32]()
    test_chebyshev_truncate[DType.float64]()

    test_chebyshev_economize[DType.float32]()
    test_chebyshev_economize[DType.float64]()

    test_chebyshev_evaluate[DType.float32]()
    test_chebyshev_evaluate[DType.float64]()

    # Power Series
    test_polynomial[DType.float32]()
    test_polynomial[DType.float64]()

    test_polynomial_small[DType.float32]()
    test_polynomial_small[DType.float64]()

    test_polynomial_hexadecimal[DType.float32]()
    test_polynomial_hexadecimal[DType.float64]()

    test_polynomial_truncate[DType.float32]()
    test_polynomial_truncate[DType.float64]()

    test_polynomial_evaluate[DType.float32]()
    test_polynomial_evaluate[DType.float64]()
