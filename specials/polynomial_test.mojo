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

import testing

from specials import polynomial as P


fn test_chebyshev() raises:
    print("# test_chebyshev")

    let p = P.Chebyshev[4, DType.float64, 1].from_coefficients[4.0, 3.0, 2.0, 1.0]()

    _ = testing.assert_equal(p.degree(), 3)
    _ = testing.assert_equal(p.get[0](), 4.0)
    _ = testing.assert_equal(p.get[1](), 3.0)
    _ = testing.assert_equal(p.get[2](), 2.0)
    _ = testing.assert_equal(p.get[3](), 1.0)

    let p_truncated = p.truncate[1]()

    _ = testing.assert_equal(p_truncated.degree(), 0)
    _ = testing.assert_equal(p_truncated.get[0](), 4.0)


fn test_chebyshev_init() raises:
    print("# test_chebyshev_init")

    alias p = P.Chebyshev[4, DType.float64, 1].from_coefficients[4.0, 3.0, 2.0, 1.0]()
    let num_terms = P.chebyshev_init[p.num_terms, p.dtype, p.simd_width, p, 3.0]()

    _ = testing.assert_equal(num_terms, 2)


fn test_chebyshev_eval() raises:
    print("# test_chebyshev_eval")

    let p = P.Chebyshev[4, DType.float64, 4].from_coefficients[4.0, 3.0, 2.0, 1.0]()
    let x = SIMD[DType.float64, 4](-1.0, -0.5, 0.5, 1.0)

    # Expected values computed using `numpy.polynomial.Chebyshev`.
    let expected = SIMD[DType.float64, 4](2.0, 2.5, 3.5, 10.0)
    let actual = P.chebyshev_eval(p, x)

    _ = testing.assert_almost_equal(actual, expected, 0.0, 1e-12)


fn main() raises:
    test_chebyshev()
    test_chebyshev_init()
    test_chebyshev_eval()
