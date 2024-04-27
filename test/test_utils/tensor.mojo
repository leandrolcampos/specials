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

from algorithm.functional import parallelize, vectorize
from python import Python
from python.object import PythonObject
from sys.info import num_physical_cores, simdwidthof
from tensor import Tensor
from tensor.random import rand


alias UnaryOperator = fn[type: DType, width: Int] (SIMD[type, width]) -> SIMD[
    type, width
]
"""Signature of a function that performs an elementwise operation on a single SIMD vector."""

alias BinaryOperator = fn[type: DType, width: Int] (
    SIMD[type, width], SIMD[type, width]
) -> SIMD[type, width]
"""Signature of a function that performs an elementwise operation on two SIMD vectors."""


# ===---------------------------- elementwise -----------------------------=== #


fn _elementwise_impl[
    func: UnaryOperator,
    type: DType,
    width: Int,
    force_sequential: Bool,
](x: Tensor[type], inout result: Tensor[type]) -> None:
    """Implements the elementwise operation on a tensor."""
    var num_elements = x.num_elements()
    var remaining_elements = num_elements
    var first_remaining_element = 0

    @parameter
    if not force_sequential:
        var num_worker = num_physical_cores()
        var num_work_items = num_worker
        var num_simds_per_work_item = (num_elements // width) // num_work_items

        # TODO: This threshold is a heuristic. We should use Mojo's autotuning.
        var parallel_threshold: Int
        if type == DType.float32:
            parallel_threshold = 16_384 // num_worker
        else:  # type == DType.float64
            parallel_threshold = 8_192 // num_worker

        if (
            num_elements >= parallel_threshold
            and num_simds_per_work_item >= 1
            and num_worker >= 2
        ):
            var num_elements_per_work_item = num_simds_per_work_item * width
            remaining_elements = math.fma(
                -num_elements_per_work_item, num_work_items, remaining_elements
            )

            @parameter
            fn execute_work_item(work_item_id: Int):
                var first_element = work_item_id * num_elements_per_work_item

                @parameter
                fn subtask_func[width: Int](index: Int):
                    var index_shifted = first_element + index

                    result.store[width](
                        index_shifted,
                        func[type, width](x.load[width](index_shifted)),
                    )

                vectorize[subtask_func, width](num_elements_per_work_item)

            parallelize[execute_work_item](num_work_items, num_worker)

        # Calculate the starting index for the remaining elements.
        first_remaining_element = num_elements - remaining_elements

    # Process the remaining elements sequentially.

    if remaining_elements > 0:

        @parameter
        fn body_func[width: Int](index: Int):
            var index_shifted = first_remaining_element + index

            result.store[width](
                index_shifted,
                func[type, width](x.load[width](index_shifted)),
            )

        vectorize[body_func, width](remaining_elements)


fn _elementwise_impl[
    func: BinaryOperator,
    type: DType,
    width: Int,
    force_sequential: Bool,
](x: Tensor[type], scalar: Scalar[type], inout result: Tensor[type]) -> None:
    """Implements the elementwise operation on a tensor and a scalar."""
    var num_elements = x.num_elements()
    var remaining_elements = num_elements
    var first_remaining_element = 0

    @parameter
    if not force_sequential:
        var num_worker = num_physical_cores()
        var num_work_items = num_worker
        var num_simds_per_work_item = (num_elements // width) // num_work_items

        # TODO: This threshold is a heuristic. We should use Mojo's autotuning.
        var parallel_threshold: Int
        if type == DType.float32:
            parallel_threshold = 262_144 // num_worker
        else:  # type == DType.float64
            parallel_threshold = 131_072 // num_worker

        if (
            num_elements >= parallel_threshold
            and num_simds_per_work_item >= 1
            and num_worker >= 2
        ):
            var num_elements_per_work_item = num_simds_per_work_item * width
            remaining_elements = math.fma(
                -num_elements_per_work_item, num_work_items, remaining_elements
            )

            @parameter
            fn execute_work_item(work_item_id: Int):
                var first_element = work_item_id * num_elements_per_work_item

                @parameter
                fn subtask_func[width: Int](index: Int):
                    var index_shifted = first_element + index

                    result.store[width](
                        index_shifted,
                        func[type, width](
                            x.load[width](index_shifted),
                            SIMD[type, width](scalar),
                        ),
                    )

                vectorize[subtask_func, width](num_elements_per_work_item)

            parallelize[execute_work_item](num_work_items, num_worker)

        # Calculate the starting index for the remaining elements.
        first_remaining_element = num_elements - remaining_elements

    # Process the remaining elements sequentially.

    if remaining_elements > 0:

        @parameter
        fn body_func[width: Int](index: Int):
            var index_shifted = first_remaining_element + index

            result.store[width](
                index_shifted,
                func[type, width](
                    x.load[width](index_shifted),
                    SIMD[type, width](scalar),
                ),
            )

        vectorize[body_func, width](remaining_elements)


fn _elementwise_impl[
    func: BinaryOperator,
    type: DType,
    width: Int,
    force_sequential: Bool,
](x: Tensor[type], y: Tensor[type], inout result: Tensor[type]) -> None:
    """Implements the elementwise operation on two tensors."""
    var num_elements = x.num_elements()
    var remaining_elements = num_elements
    var first_remaining_element = 0

    @parameter
    if not force_sequential:
        var num_worker = num_physical_cores()
        var num_work_items = num_worker
        var num_simds_per_work_item = (num_elements // width) // num_work_items

        # TODO: This threshold is a heuristic. We should use Mojo's autotuning.
        var parallel_threshold: Int
        if type == DType.float32:
            parallel_threshold = 131_072 // num_worker
        else:  # type == DType.float64
            parallel_threshold = 65_536 // num_worker

        if (
            num_elements >= parallel_threshold
            and num_simds_per_work_item >= 1
            and num_worker >= 2
        ):
            var num_elements_per_work_item = num_simds_per_work_item * width
            remaining_elements = math.fma(
                -num_elements_per_work_item, num_work_items, remaining_elements
            )

            @parameter
            fn execute_work_item(work_item_id: Int):
                var first_element = work_item_id * num_elements_per_work_item

                @parameter
                fn subtask_func[width: Int](index: Int):
                    var index_shifted = first_element + index

                    result.store[width](
                        index_shifted,
                        func[type, width](
                            x.load[width](index_shifted),
                            y.load[width](index_shifted),
                        ),
                    )

                vectorize[subtask_func, width](num_elements_per_work_item)

            parallelize[execute_work_item](num_work_items, num_worker)

        # Calculate the starting index for the remaining elements.
        first_remaining_element = num_elements - remaining_elements

    # Process the remaining elements sequentially.

    if remaining_elements > 0:

        @parameter
        fn body_func[width: Int](index: Int):
            var index_shifted = first_remaining_element + index

            result.store[width](
                index_shifted,
                func[type, width](
                    x.load[width](index_shifted),
                    y.load[width](index_shifted),
                ),
            )

        vectorize[body_func, width](remaining_elements)


fn elementwise[
    func: UnaryOperator,
    type: DType,
    width: Int = simdwidthof[type](),
    force_sequential: Bool = False,
](x: Tensor[type]) -> Tensor[type]:
    """Applies an unary operator to a tensor element-wise.

    Parameters:
        func: The unary operator to apply to the input.
        type: The input and output type.
        width: The SIMD vector width to use. Default is the vector size of
            the type on the host system.
        force_sequential: Whether to force sequential execution. Default is
            `False`.

    Args:
        x: The tensor.

    Returns:
        The result of applying the unary operator to the input.

    Constraints:
        The type must be a floating-point of single or double precision.
    """
    constrained[
        type == DType.float32 or type == DType.float64,
        "tensor type must be a floating-point of single (`float32`) "
        + "or double (`float64`) precision.",
    ]()

    var result = Tensor[type](x.shape())
    _elementwise_impl[func, type, width, force_sequential](x, result)

    return result


fn elementwise[
    func: BinaryOperator,
    type: DType,
    width: Int = simdwidthof[type](),
    force_sequential: Bool = False,
](x: Tensor[type], scalar: Scalar[type]) -> Tensor[type]:
    """Applies a binary operator to a tensor and a scalar element-wise.

    Parameters:
        func: The binary operator to apply to the inputs.
        type: The input and output type.
        width: The SIMD vector width to use. Default is the vector size of
            the type on the host system.
        force_sequential: Whether to force sequential execution. Default is
            `False`.

    Args:
        x: The tensor.
        scalar: The scalar.

    Returns:
        The result of applying the binary operator to the inputs.

    Constraints:
        The type must be a floating-point of single or double precision.
    """
    constrained[
        type == DType.float32 or type == DType.float64,
        "tensor type must be a floating-point of single (`float32`) "
        + "or double (`float64`) precision.",
    ]()

    var result = Tensor[type](x.shape())
    _elementwise_impl[func, type, width, force_sequential](x, scalar, result)

    return result


fn elementwise[
    func: BinaryOperator,
    type: DType,
    width: Int = simdwidthof[type](),
    force_sequential: Bool = False,
](x: Tensor[type], y: Tensor[type]) raises -> Tensor[type]:
    """Applies a binary operator to two tensors element-wise.

    Parameters:
        func: The binary operator to apply to the inputs.
        type: The input and output type.
        width: The SIMD vector width to use. Default is the vector size of
            the type on the host system.
        force_sequential: Whether to force sequential execution. Default is
            `False`.

    Args:
        x: The first tensor.
        y: The second tensor.

    Returns:
       The result of applying the binary operator to the inputs.

    Constraints:
        The type must be a floating-point of single or double precision. And it
        will raise an exception if the arguments do not have the same shape.
    """
    constrained[
        type == DType.float32 or type == DType.float64,
        "tensor type must be a floating-point of single (`float32`) "
        + "or double (`float64`) precision.",
    ]()

    if x.shape() != y.shape():
        raise Error("The arguments `x` and `y` must have the same shape.")

    var result = Tensor[type](x.shape())
    _elementwise_impl[func, type, width, force_sequential](x, y, result)

    return result


# ===----------------------------- benchmark ------------------------------=== #


fn run_benchmark[
    func: UnaryOperator,
    type: DType,
    width: Int = simdwidthof[type](),
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

    Parameters:
        func: The binary operator to apply to the input. This is the function
            that will be benchmarked.
        type: The input type.
        width: The SIMD vector width to use. Default is the vector size of
            the type on the host system.
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

    Constraints:
        The type must be a floating-point of single or double precision.
    """

    @always_inline
    @parameter
    fn test_fn():
        _ = elementwise[func, type, width, force_sequential](x)

    return benchmark.run[test_fn](
        num_warmup=num_warmup,
        max_iters=max_iters,
        min_runtime_secs=min_runtime_secs,
        max_runtime_secs=max_runtime_secs,
    )


fn run_benchmark[
    func: BinaryOperator,
    type: DType,
    width: Int = simdwidthof[type](),
    force_sequential: Bool = False,
](
    x: Tensor[type],
    scalar: Scalar[type],
    num_warmup: Int = 2,
    max_iters: Int = 100_000,
    min_runtime_secs: SIMD[DType.float64, 1] = 0.5,
    max_runtime_secs: SIMD[DType.float64, 1] = 1,
) -> benchmark.Report:
    """
    Runs a benchmark for a binary operator applied to a tensor and a scalar
    element-wise.

    Benchmarking continues until `min_runtime_secs` has elapsed and either
    `max_iters` OR `max_runtime_secs` is achieved.

    Parameters:
        func: The binary operator to apply to the inputs. This is the function
            that will be benchmarked.
        type: The input type.
        width: The SIMD vector width to use. Default is the vector size of
            the type on the host system.
        force_sequential: Whether to force sequential execution. Default is
            `False`.

    Args:
        x: The tensor.
        scalar: The scalar.
        num_warmup: Number of warmup iterations to run before starting
            benchmarking. Default is 2.
        max_iters: Max number of iterations to run. Default is 100_000.
        min_runtime_secs: Lower bound on benchmarking time in secs. Default
            is 0.5.
        max_runtime_secs: Upper bound on benchmarking time in secs. Default
            is 1.

    Returns:
        A report containing statistics of the benchmark.

    Constraints:
        The type must be a floating-point of single or double precision.
    """

    @always_inline
    @parameter
    fn test_fn():
        _ = elementwise[func, type, width, force_sequential](x, scalar)

    return benchmark.run[test_fn](
        num_warmup=num_warmup,
        max_iters=max_iters,
        min_runtime_secs=min_runtime_secs,
        max_runtime_secs=max_runtime_secs,
    )


fn run_benchmark[
    func: BinaryOperator,
    type: DType,
    width: Int = simdwidthof[type](),
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

    Parameters:
        func: The binary operator to apply to the inputs. This is the function
            that will be benchmarked.
        type: The input type.
        width: The SIMD vector width to use. Default is the vector size of
            the type on the host system.
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

    Constraints:
        The type must be a floating-point of single or double precision. And it
        will raise an exception if the arguments do not have the same shape.
    """
    if x.shape() != y.shape():
        raise Error("The arguments `x` and `y` must have the same shape.")

    @always_inline
    @parameter
    fn test_fn():
        try:
            _ = elementwise[func, type, width, force_sequential](x, y)
        except Error:
            pass

    return benchmark.run[test_fn](
        num_warmup=num_warmup,
        max_iters=max_iters,
        min_runtime_secs=min_runtime_secs,
        max_runtime_secs=max_runtime_secs,
    )


# ===----------------------------- random ---------------------------------=== #


fn random_uniform[
    type: DType, width: Int = simdwidthof[type]()
](
    min_value: Scalar[type],
    max_value: Scalar[type],
    *shape: Int,
) raises -> Tensor[type]:
    """Generates a tensor with random values drawn from a uniform distribution.

    Parameters:
        type: The type of the tensor.
        width: The SIMD vector width to use. Defaults to the vector size of
            the type on the host system.

    Args:
        min_value: The lower bound of the uniform distribution.
        max_value: The upper bound of the uniform distribution.
        shape: The tensor shape.

    Returns:
        The tensor with random values drawn from a uniform distribution.

    Constraints:
        The type must be a floating-point of single or double precision. And it
        will raise an exception if `min_value >= max_value`.
    """
    constrained[
        type == DType.float32 or type == DType.float64,
        "tensor type must be a floating-point of single (`float32`) "
        + "or double (`float64`) precision.",
    ]()

    if min_value >= max_value:
        raise Error("`min_value` must be less than `max_value`.")

    var raw = rand[type](shape)
    var scaled = elementwise[math.mul, type, width](raw, max_value - min_value)
    var shifted = elementwise[math.add, type, width](scaled, min_value)

    return shifted


# ===------------------------------- numpy --------------------------------=== #


fn tensor_to_numpy_array[type: DType](x: Tensor[type]) raises -> PythonObject:
    """Converts a tensor to a NumPy array.

    Parameters:
        type: The type of the tensor.

    Args:
        x: The tensor.

    Returns:
        The NumPy array with the same shape, type, and elements as the tensor.

    Constraints:
        Will raise an exception if it is not possible to convert the tensor to
        a NumPy array with the same shape, data type, and elements.
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
