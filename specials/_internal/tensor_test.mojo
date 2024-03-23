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

from specials._internal.tensor import elementwise
from specials._internal.testing import UnitTest


fn test_elemwise_tensor[
    dtype: DType, force_sequential: Bool = False
](*shape: Int) raises:
    random.seed(42)

    var x = random.rand[dtype](shape)
    var res = elementwise[math.cos, force_sequential=force_sequential](x)

    var unit_test = UnitTest(
        "test_elemwise_tensor_" + str(x.spec()) + "_" + str(force_sequential)
    )
    var rtol: Scalar[dtype]

    @parameter
    if dtype == DType.float32:
        rtol = 1e-6
    else:  # dtype == DType.float64
        rtol = 1e-12

    # The for loop is simple, direct, and so, good for testing.
    for i in range(x.num_elements()):
        unit_test.assert_all_close(res[i], math.cos(x[i]), 0.0, rtol)


fn test_elemwise_tensor_scalar[
    dtype: DType, force_sequential: Bool = False
](*shape: Int) raises:
    random.seed(42)

    var x = random.rand[dtype](shape)
    var y = Scalar[dtype](1.5)
    var res = elementwise[math.add, force_sequential=force_sequential](x, y)

    var unit_test = UnitTest(
        "test_elemwise_tensor_scalar_" + str(x.spec()) + "_" + str(force_sequential)
    )
    var rtol: Scalar[dtype]

    @parameter
    if dtype == DType.float32:
        rtol = 1e-6
    else:  # dtype == DType.float64
        rtol = 1e-12

    # The for loop is simple, direct, and so, good for testing.
    for i in range(x.num_elements()):
        unit_test.assert_all_close(res[i], x[i] + y, 0.0, rtol)


fn test_elemwise_tensor_tensor[
    dtype: DType, force_sequential: Bool = False
](*shape: Int) raises:
    random.seed(42)

    var x = random.rand[dtype](shape)
    var y = random.rand[dtype](x.shape())
    var res = elementwise[math.add, force_sequential=force_sequential](x, y)

    var unit_test = UnitTest(
        "test_elemwise_tensor_tensor_" + str(x.spec()) + "_" + str(force_sequential)
    )
    var rtol: Scalar[dtype]

    @parameter
    if dtype == DType.float32:
        rtol = 1e-6
    else:  # dtype == DType.float64
        rtol = 1e-12

    # The for loop is simple, direct, and so, good for testing.
    for i in range(x.num_elements()):
        unit_test.assert_all_close(res[i], x[i] + y[i], 0.0, rtol)


fn main() raises:
    # The elementwise function for a binary operator: tensor.
    test_elemwise_tensor[DType.float32]()
    test_elemwise_tensor[DType.float32](16, 16)
    test_elemwise_tensor[DType.float32](16_384)

    test_elemwise_tensor[DType.float32, force_sequential=True]()
    test_elemwise_tensor[DType.float32, force_sequential=True](16, 16)

    test_elemwise_tensor[DType.float64]()
    test_elemwise_tensor[DType.float64](16, 16)
    test_elemwise_tensor[DType.float64](8_192)

    # The elementwise function for a binary operator: tensor-scalar.
    test_elemwise_tensor_scalar[DType.float32]()
    test_elemwise_tensor_scalar[DType.float32](16, 16)
    test_elemwise_tensor_scalar[DType.float32](262_144)

    test_elemwise_tensor_scalar[DType.float32, force_sequential=True]()
    test_elemwise_tensor_scalar[DType.float32, force_sequential=True](16, 16)

    test_elemwise_tensor_scalar[DType.float64]()
    test_elemwise_tensor_scalar[DType.float64](16, 16)
    test_elemwise_tensor_scalar[DType.float64](131_072)

    # The elementwise function for a binary operator: tensor-tensor.
    test_elemwise_tensor_tensor[DType.float32]()
    test_elemwise_tensor_tensor[DType.float32](16, 16)
    test_elemwise_tensor_tensor[DType.float32](131_072)

    test_elemwise_tensor_tensor[DType.float32, force_sequential=True]()
    test_elemwise_tensor_tensor[DType.float32, force_sequential=True](16, 16)

    test_elemwise_tensor_tensor[DType.float64]()
    test_elemwise_tensor_tensor[DType.float64](16, 17)
    test_elemwise_tensor_tensor[DType.float64](65_536)
