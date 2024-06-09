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

"""Utilities for working with Tensors."""

import benchmark
import math

from algorithm.functional import vectorize
from python import Python
from python.object import PythonObject
from tensor import Tensor, TensorShape
from tensor.random import rand

from specials.utils import functional


alias UnaryOperator = fn[type: DType, width: Int] (SIMD[type, width]) -> SIMD[
    type, width
]
"""
Signature of a function that performs an elementwise operation on a single SIMD
vector.
"""

alias BinaryOperator = fn[type: DType, width: Int] (
    SIMD[type, width], SIMD[type, width]
) -> SIMD[type, width]
"""
Signature of a function that performs an elementwise operation on two SIMD
vectors.
"""


# ===----------------------------------------------------------------------=== #
# Elementwise
# ===----------------------------------------------------------------------=== #


@always_inline
fn _elementwise_impl[
    func: UnaryOperator,
    type: DType,
    simd_width: Int,
    force_sequential: Bool,
](x: Tensor[type]) -> Tensor[type]:
    """Implements the elementwise operation on a tensor."""
    var result = Tensor[type](x.shape())
    var num_elements = x.num_elements()

    @always_inline
    @parameter
    fn inner_func[simd_width: Int](index: Int):
        var a = x.load[width=simd_width](index)
        result.store[width=simd_width](index, func(a))

    @parameter
    if force_sequential:
        vectorize[inner_func, simd_width](num_elements)
    else:
        functional.elementwise[inner_func, simd_width=simd_width](num_elements)

    return result ^


@always_inline
fn _elementwise_impl[
    func: BinaryOperator,
    type: DType,
    simd_width: Int,
    force_sequential: Bool,
](x: Tensor[type], scalar: Scalar[type]) -> Tensor[type]:
    """Implements the elementwise operation on a tensor and a scalar."""
    var result = Tensor[type](x.shape())
    var num_elements = x.num_elements()

    @always_inline
    @parameter
    fn inner_func[simd_width: Int](index: Int):
        var a = x.load[width=simd_width](index)
        var b = SIMD[type, simd_width](scalar)
        result.store[width=simd_width](index, func(a, b))

    @parameter
    if force_sequential:
        vectorize[inner_func, simd_width](num_elements)
    else:
        functional.elementwise[inner_func, simd_width=simd_width](num_elements)

    return result ^


@always_inline
fn _elementwise_impl[
    func: BinaryOperator,
    type: DType,
    simd_width: Int,
    force_sequential: Bool,
](x: Tensor[type], y: Tensor[type]) -> Tensor[type]:
    """Implements the elementwise operation on two tensors."""
    var result = Tensor[type](x.shape())
    var num_elements = x.num_elements()

    @always_inline
    @parameter
    fn inner_func[simd_width: Int](index: Int):
        var a = x.load[width=simd_width](index)
        var b = y.load[width=simd_width](index)
        result.store[width=simd_width](index, func(a, b))

    @parameter
    if force_sequential:
        vectorize[inner_func, simd_width](num_elements)
    else:
        functional.elementwise[inner_func, simd_width=simd_width](num_elements)

    return result ^


fn elementwise[
    func: UnaryOperator,
    *,
    type: DType,
    simd_width: Int = simdwidthof[type](),
    force_sequential: Bool = False,
](x: Tensor[type]) -> Tensor[type]:
    """Applies an unary operator to a tensor element-wise.

    Constraints:
        The type must be a floating-point of single or double precision.

    Parameters:
        func: The unary operator to apply to the input.
        type: The input and output type.
        simd_width: The SIMD vector to use. Default is the vector size of the
             type on the host system.
        force_sequential: Whether to force sequential execution. Default is
            `False`.

    Args:
        x: The tensor.

    Returns:
        The result of applying the unary operator to the input.
    """
    constrained[
        type == DType.float32 or type == DType.float64,
        "tensor type must be a floating-point of single (`float32`) "
        + "or double (`float64`) precision.",
    ]()

    var result = _elementwise_impl[func, type, simd_width, force_sequential](x)
    return result ^


fn elementwise[
    func: BinaryOperator,
    *,
    type: DType,
    simd_width: Int = simdwidthof[type](),
    force_sequential: Bool = False,
](x: Tensor[type], scalar: Scalar[type]) -> Tensor[type]:
    """Applies a binary operator to a tensor and a scalar element-wise.

    Constraints:
        The type must be a floating-point of single or double precision.

    Parameters:
        func: The binary operator to apply to the inputs.
        type: The input and output type.
        simd_width: The SIMD vector to use. Default is the vector size of the
             type on the host system.
        force_sequential: Whether to force sequential execution. Default is
            `False`.

    Args:
        x: The tensor.
        scalar: The scalar.

    Returns:
        The result of applying the binary operator to the inputs.
    """
    constrained[
        type == DType.float32 or type == DType.float64,
        "tensor type must be a floating-point of single (`float32`) "
        + "or double (`float64`) precision.",
    ]()

    var result = _elementwise_impl[func, type, simd_width, force_sequential](
        x, scalar
    )
    return result ^


fn elementwise[
    func: BinaryOperator,
    *,
    type: DType,
    simd_width: Int = simdwidthof[type](),
    force_sequential: Bool = False,
](x: Tensor[type], y: Tensor[type]) raises -> Tensor[type]:
    """Applies a binary operator to two tensors element-wise.

    Constraints:
        The type must be a floating-point of single or double precision. And it
        will raise an exception if the arguments do not have the same shape.

    Parameters:
        func: The binary operator to apply to the inputs.
        type: The input and output type.
        simd_width: The SIMD width to use. Default is the vector size of the
            type on the host system.
        force_sequential: Whether to force sequential execution. Default is
            `False`.

    Args:
        x: The first tensor.
        y: The second tensor.

    Returns:
       The result of applying the binary operator to the inputs.
    """
    constrained[
        type == DType.float32 or type == DType.float64,
        "tensor type must be a floating-point of single (`float32`) "
        + "or double (`float64`) precision.",
    ]()

    if x.shape() != y.shape():
        raise Error("The arguments `x` and `y` must have the same shape.")

    var result = _elementwise_impl[func, type, simd_width, force_sequential](
        x, y
    )
    return result ^


# ===----------------------------------------------------------------------=== #
# Benchmark
# ===----------------------------------------------------------------------=== #


fn run_benchmark[
    func: UnaryOperator,
    *,
    type: DType,
    simd_width: Int = simdwidthof[type](),
    force_sequential: Bool = False,
](
    x: Tensor[type],
    num_warmup: Int = 2,
    max_iters: Int = 100_000,
    min_runtime_secs: SIMD[DType.float64, 1] = 0.5,
    max_runtime_secs: SIMD[DType.float64, 1] = 1,
) -> benchmark.Report:
    """Runs a benchmark for a unary operator applied to a tensor element-wise.

    Benchmarking continues until `min_runtime_secs` has elapsed and either
    `max_iters` OR `max_runtime_secs` is achieved.

    Constraints:
        The type must be a floating-point of single or double precision.

    Parameters:
        func: The binary operator to apply to the input. This is the function
            that will be benchmarked.
        type: The input type.
        simd_width: The SIMD width to use. Default is the vector size of the
            type on the host system.
        force_sequential: Whether to force sequential execution. Default is
            `False`.

    Args:
        x: The tensor.
        num_warmup: Number of warmup iterations to run before starting
            benchmarking. Default is 2.
        max_iters: Max number of iterations to run. Default is 100_000.
        min_runtime_secs: Lower bound on benchmarking time in secs. Default
            is 0.5.
        max_runtime_secs: Upper bound on benchmarking time in secs. Default
            is 1.

    Returns:
        A report containing statistics of the benchmark.
    """

    @always_inline
    @parameter
    fn test_fn():
        _ = elementwise[
            func,
            type=type,
            simd_width=simd_width,
            force_sequential=force_sequential,
        ](x)

    return benchmark.run[test_fn](
        num_warmup=num_warmup,
        max_iters=max_iters,
        min_runtime_secs=min_runtime_secs,
        max_runtime_secs=max_runtime_secs,
    )


fn run_benchmark[
    func: BinaryOperator,
    *,
    type: DType,
    simd_width: Int = simdwidthof[type](),
    force_sequential: Bool = False,
](
    x: Tensor[type],
    y: Tensor[type],
    num_warmup: Int = 2,
    max_iters: Int = 100_000,
    min_runtime_secs: SIMD[DType.float64, 1] = 0.5,
    max_runtime_secs: SIMD[DType.float64, 1] = 1,
) raises -> benchmark.Report:
    """
    Runs a benchmark for a binary operator applied to two tensors element-wise.

    Benchmarking continues until `min_runtime_secs` has elapsed and either
    `max_iters` OR `max_runtime_secs` is achieved.

    Constraints:
        The type must be a floating-point of single or double precision. And it
        will raise an exception if the arguments do not have the same shape.

    Parameters:
        func: The binary operator to apply to the inputs. This is the function
            that will be benchmarked.
        type: The input type.
        simd_width: The SIMD width to use. Default is the vector size of the
            type on the host system.
        force_sequential: Whether to force sequential execution. Default is
            `False`.

    Args:
        x: The first tensor.
        y: The second tensor.
        num_warmup: Number of warmup iterations to run before starting
            benchmarking. Default is 2.
        max_iters: Max number of iterations to run. Default is 100_000.
        min_runtime_secs: Lower bound on benchmarking time in secs. Default
            is 0.5.
        max_runtime_secs: Upper bound on benchmarking time in secs. Default
            is 1.

    Returns:
        A report containing statistics of the benchmark.
    """
    if x.shape() != y.shape():
        raise Error("The arguments `x` and `y` must have the same shape.")

    @always_inline
    @parameter
    fn test_fn():
        try:
            _ = elementwise[
                func,
                type=type,
                simd_width=simd_width,
                force_sequential=force_sequential,
            ](x, y)
        except Error:
            pass

    return benchmark.run[test_fn](
        num_warmup=num_warmup,
        max_iters=max_iters,
        min_runtime_secs=min_runtime_secs,
        max_runtime_secs=max_runtime_secs,
    )


# ===----------------------------------------------------------------------=== #
# Random
# ===----------------------------------------------------------------------=== #


fn random_uniform[
    type: DType, *, simd_width: Int = simdwidthof[type]()
](
    min_value: Scalar[type],
    max_value: Scalar[type],
    owned shape: TensorShape,
) raises -> Tensor[type]:
    """Generates a tensor with random values drawn from a uniform distribution.

    Constraints:
        The type must be a floating-point of single or double precision. And it
        will raise an exception if `min_value >= max_value`.

    Parameters:
        type: The type of the tensor.
        simd_width: The SIMD width to use. Defaults to the vector size of the
            type on the host system.

    Args:
        min_value: The lower bound of the uniform distribution.
        max_value: The upper bound of the uniform distribution.
        shape: The tensor shape.

    Returns:
        The tensor with random values drawn from a uniform distribution.
    """
    constrained[
        type == DType.float32 or type == DType.float64,
        "tensor type must be a floating-point of single (`float32`) "
        + "or double (`float64`) precision.",
    ]()

    if min_value >= max_value:
        raise Error("`min_value` must be less than `max_value`.")

    var raw = rand[type](shape)
    var scaled = elementwise[math.mul, simd_width=simd_width](
        raw, max_value - min_value
    )
    var shifted = elementwise[math.add, simd_width=simd_width](
        scaled, min_value
    )

    return shifted


fn random_uniform[
    type: DType, *, simd_width: Int = simdwidthof[type]()
](
    min_value: Scalar[type],
    max_value: Scalar[type],
    *shape: Int,
) raises -> Tensor[type]:
    """Generates a tensor with random values drawn from a uniform distribution.

    Constraints:
        The type must be a floating-point of single or double precision. And it
        will raise an exception if `min_value >= max_value`.

    Parameters:
        type: The type of the tensor.
        simd_width: The SIMD width to use. Defaults to the vector size of the
            type on the host system.

    Args:
        min_value: The lower bound of the uniform distribution.
        max_value: The upper bound of the uniform distribution.
        shape: The tensor shape.

    Returns:
        The tensor with random values drawn from a uniform distribution.
    """
    return random_uniform[type, simd_width=simd_width](
        min_value, max_value, TensorShape(shape)
    )


# ===----------------------------------------------------------------------=== #
# NumPy
# ===----------------------------------------------------------------------=== #


fn tensor_to_numpy_array[type: DType](x: Tensor[type]) raises -> PythonObject:
    """Converts a tensor to a NumPy array.

    Constraints:
        Will raise an exception if it is not possible to convert the tensor to
        a NumPy array with the same shape, data type, and elements.

    Parameters:
        type: The type of the tensor.

    Args:
        x: The tensor.

    Returns:
        The NumPy array with the same shape, type, and elements as the tensor.
    """
    var builtins = Python.import_module("builtins")
    var np = Python.import_module("numpy")

    var shape = builtins.tuple(
        builtins.map(builtins.int, builtins.str(x.shape().__str__()).split("x"))
    )
    var num_elements = x.num_elements()

    var numpy_array = np.zeros(num_elements, type.__str__())

    for i in range(num_elements):
        _ = np.put(numpy_array, i, x[i])

    return numpy_array.reshape(shape)
