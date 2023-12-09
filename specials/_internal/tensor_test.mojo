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

"""Tests for tensor utilities."""

import math
import random
import testing

from algorithm import vectorize
from sys.info import simdwidthof

from specials._internal.tensor import elementwise
from specials._internal.testing import UnitTest


fn test_elemwise_tensor_scalar[
    dtype: DType, force_sequential: Bool = False
](*shape: Int) raises:
    random.seed(42)

    let x = random.rand[dtype](shape)
    let y = SIMD[dtype, 1](1.5)
    let res = elementwise[math.add, force_sequential=force_sequential](x, y)

    let unit_test = UnitTest("test_elemwise_tensor_scalar_" + str(x.spec()))
    let rtol: SIMD[dtype, 1]

    @parameter
    if dtype == DType.float32:
        rtol = 1e-6
    else:  # dtype == DType.float64
        rtol = 1e-12

    # The for loop is simple, direct, and so, good for testing.
    for i in range(x.num_elements()):
        unit_test.assert_almost_equal(x[i] + y, res[i], 0.0, rtol)


fn test_elemwise_tensor_tensor[
    dtype: DType, force_sequential: Bool = False
](*shape: Int) raises:
    random.seed(42)

    let x = random.rand[dtype](shape)
    let y = random.rand[dtype](x.shape())
    let res = elementwise[math.add, force_sequential=force_sequential](x, y)

    let unit_test = UnitTest("test_elemwise_tensor_tensor_" + str(x.spec()))
    let rtol: SIMD[dtype, 1]

    @parameter
    if dtype == DType.float32:
        rtol = 1e-6
    else:  # dtype == DType.float64
        rtol = 1e-12

    # The for loop is simple, direct, and so, good for testing.
    for i in range(x.num_elements()):
        unit_test.assert_almost_equal(x[i] + y[i], res[i], 0.0, rtol)


fn main() raises:
    # The elementwise function for a binary operation: tensor-scalar.
    test_elemwise_tensor_scalar[DType.float32]()
    test_elemwise_tensor_scalar[DType.float32](16, 16)
    test_elemwise_tensor_scalar[DType.float32](524_289)

    test_elemwise_tensor_scalar[DType.float32, force_sequential=True]()
    test_elemwise_tensor_scalar[DType.float32, force_sequential=True](16, 16)

    test_elemwise_tensor_scalar[DType.float64]()
    test_elemwise_tensor_scalar[DType.float64](16, 16)
    test_elemwise_tensor_scalar[DType.float64](262_145)

    # The elementwise function for a binary operation: tensor-tensor.
    test_elemwise_tensor_tensor[DType.float32]()
    test_elemwise_tensor_tensor[DType.float32](16, 16)
    test_elemwise_tensor_tensor[DType.float32](262_145)

    test_elemwise_tensor_tensor[DType.float32, force_sequential=True]()
    test_elemwise_tensor_tensor[DType.float32, force_sequential=True](16, 16)

    test_elemwise_tensor_tensor[DType.float64]()
    test_elemwise_tensor_tensor[DType.float64](16, 17)
    test_elemwise_tensor_tensor[DType.float64](131_073)
