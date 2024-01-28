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
import random

from algorithm.functional import parallelize, vectorize
from python import Python
from python.object import PythonObject
from sys.info import num_physical_cores, simdwidthof
from tensor import Tensor

from specials._internal.asserting import assert_float_dtype


alias UnaryOperator = fn[dtype: DType, simd_width: Int] (
    SIMD[dtype, simd_width]
) -> SIMD[dtype, simd_width]
"""Signature of a function that performs an elementwise operation on a single SIMD vector."""

alias BinaryOperator = fn[dtype: DType, simd_width: Int] (
    SIMD[dtype, simd_width], SIMD[dtype, simd_width]
) -> SIMD[dtype, simd_width]
"""Signature of a function that performs an elementwise operation on two SIMD vectors."""


# ===---------------------------- elementwise -----------------------------=== #


fn _elementwise_impl[
    func: UnaryOperator,
    dtype: DType,
    simd_width: Int,
    force_sequential: Bool,
](x: Tensor[dtype], inout result: Tensor[dtype]) -> None:
    """Implements the elementwise operation on a tensor."""
    let num_elements = x.num_elements()
    var remaining_elements = num_elements
    var first_remaining_element = 0

    @parameter
    if not force_sequential:
        let num_worker = num_physical_cores()
        let num_work_items = num_worker
        let num_simds_per_work_item = (num_elements // simd_width) // num_work_items

        # TODO: This threshold is a heuristic. We should use Mojo's autotuning.
        let parallel_threshold: Int
        if dtype == DType.float32:
            parallel_threshold = 16_384 // num_worker
        else:  # dtype == DType.float64
            parallel_threshold = 8_192 // num_worker

        if (
            num_elements >= parallel_threshold
            and num_simds_per_work_item >= 1
            and num_worker >= 2
        ):
            let num_elements_per_work_item = num_simds_per_work_item * simd_width
            remaining_elements -= num_simds_per_work_item * num_work_items * simd_width

            @parameter
            fn execute_work_item(work_item_id: Int):
                let first_element = work_item_id * num_elements_per_work_item

                @parameter
                fn subtask_func[simd_width: Int](index: Int):
                    let index_shifted = first_element + index

                    result.simd_store[simd_width](
                        index_shifted,
                        func[dtype, simd_width](x.simd_load[simd_width](index_shifted)),
                    )

                vectorize[simd_width, subtask_func](num_elements_per_work_item)

            parallelize[execute_work_item](num_work_items, num_worker)

        # Calculate the starting index for the remaining elements.
        first_remaining_element = num_elements - remaining_elements

    # Process the remaining elements sequentially.

    if remaining_elements > 0:

        @parameter
        fn body_func[simd_width: Int](index: Int):
            let index_shifted = first_remaining_element + index

            result.simd_store[simd_width](
                index_shifted,
                func[dtype, simd_width](x.simd_load[simd_width](index_shifted)),
            )

        vectorize[simd_width, body_func](remaining_elements)


fn _elementwise_impl[
    func: BinaryOperator,
    dtype: DType,
    simd_width: Int,
    force_sequential: Bool,
](x: Tensor[dtype], scalar: SIMD[dtype, 1], inout result: Tensor[dtype]) -> None:
    """Implements the elementwise operation on a tensor and a scalar."""
    let num_elements = x.num_elements()
    var remaining_elements = num_elements
    var first_remaining_element = 0

    @parameter
    if not force_sequential:
        let num_worker = num_physical_cores()
        let num_work_items = num_worker
        let num_simds_per_work_item = (num_elements // simd_width) // num_work_items

        # TODO: This threshold is a heuristic. We should use Mojo's autotuning.
        let parallel_threshold: Int
        if dtype == DType.float32:
            parallel_threshold = 262_144 // num_worker
        else:  # dtype == DType.float64
            parallel_threshold = 131_072 // num_worker

        if (
            num_elements >= parallel_threshold
            and num_simds_per_work_item >= 1
            and num_worker >= 2
        ):
            let num_elements_per_work_item = num_simds_per_work_item * simd_width
            remaining_elements -= num_simds_per_work_item * num_work_items * simd_width

            @parameter
            fn execute_work_item(work_item_id: Int):
                let first_element = work_item_id * num_elements_per_work_item

                @parameter
                fn subtask_func[simd_width: Int](index: Int):
                    let index_shifted = first_element + index

                    result.simd_store[simd_width](
                        index_shifted,
                        func[dtype, simd_width](
                            x.simd_load[simd_width](index_shifted),
                            SIMD[dtype, simd_width](scalar),
                        ),
                    )

                vectorize[simd_width, subtask_func](num_elements_per_work_item)

            parallelize[execute_work_item](num_work_items, num_worker)

        # Calculate the starting index for the remaining elements.
        first_remaining_element = num_elements - remaining_elements

    # Process the remaining elements sequentially.

    if remaining_elements > 0:

        @parameter
        fn body_func[simd_width: Int](index: Int):
            let index_shifted = first_remaining_element + index

            result.simd_store[simd_width](
                index_shifted,
                func[dtype, simd_width](
                    x.simd_load[simd_width](index_shifted),
                    SIMD[dtype, simd_width](scalar),
                ),
            )

        vectorize[simd_width, body_func](remaining_elements)


fn _elementwise_impl[
    func: BinaryOperator,
    dtype: DType,
    simd_width: Int,
    force_sequential: Bool,
](x: Tensor[dtype], y: Tensor[dtype], inout result: Tensor[dtype]) -> None:
    """Implements the elementwise operation on two tensors."""
    let num_elements = x.num_elements()
    var remaining_elements = num_elements
    var first_remaining_element = 0

    @parameter
    if not force_sequential:
        let num_worker = num_physical_cores()
        let num_work_items = num_worker
        let num_simds_per_work_item = (num_elements // simd_width) // num_work_items

        # TODO: This threshold is a heuristic. We should use Mojo's autotuning.
        let parallel_threshold: Int
        if dtype == DType.float32:
            parallel_threshold = 131_072 // num_worker
        else:  # dtype == DType.float64
            parallel_threshold = 65_536 // num_worker

        if (
            num_elements >= parallel_threshold
            and num_simds_per_work_item >= 1
            and num_worker >= 2
        ):
            let num_elements_per_work_item = num_simds_per_work_item * simd_width
            remaining_elements -= num_simds_per_work_item * num_work_items * simd_width

            @parameter
            fn execute_work_item(work_item_id: Int):
                let first_element = work_item_id * num_elements_per_work_item

                @parameter
                fn subtask_func[simd_width: Int](index: Int):
                    let index_shifted = first_element + index

                    result.simd_store[simd_width](
                        index_shifted,
                        func[dtype, simd_width](
                            x.simd_load[simd_width](index_shifted),
                            y.simd_load[simd_width](index_shifted),
                        ),
                    )

                vectorize[simd_width, subtask_func](num_elements_per_work_item)

            parallelize[execute_work_item](num_work_items, num_worker)

        # Calculate the starting index for the remaining elements.
        first_remaining_element = num_elements - remaining_elements

    # Process the remaining elements sequentially.

    if remaining_elements > 0:

        @parameter
        fn body_func[simd_width: Int](index: Int):
            let index_shifted = first_remaining_element + index

            result.simd_store[simd_width](
                index_shifted,
                func[dtype, simd_width](
                    x.simd_load[simd_width](index_shifted),
                    y.simd_load[simd_width](index_shifted),
                ),
            )

        vectorize[simd_width, body_func](remaining_elements)


fn elementwise[
    func: UnaryOperator,
    dtype: DType,
    simd_width: Int = simdwidthof[dtype](),
    force_sequential: Bool = False,
](x: Tensor[dtype]) -> Tensor[dtype]:
    """Applies an unary operator to a tensor element-wise.

    Parameters:
        func: The unary operator to apply to the input.
        dtype: The input and output data type.
        simd_width: The SIMD vector width to use. Defaults to the vector size of
            the data type on the host system.
        force_sequential: Whether to force sequential execution (default `False`).

    Args:
        x: The tensor.

    Returns:
        The result of applying the unary operator to the input.

    Constraints:
        The data type must be a floating-point of single or double precision.
    """
    assert_float_dtype["dtype", dtype]()

    var result = Tensor[dtype](x.shape())
    _elementwise_impl[func, dtype, simd_width, force_sequential](x, result)

    return result


fn elementwise[
    func: BinaryOperator,
    dtype: DType,
    simd_width: Int = simdwidthof[dtype](),
    force_sequential: Bool = False,
](x: Tensor[dtype], scalar: SIMD[dtype, 1]) -> Tensor[dtype]:
    """Applies a binary operator to a tensor and a scalar element-wise.

    Parameters:
        func: The binary operator to apply to the inputs.
        dtype: The input and output data type.
        simd_width: The SIMD vector width to use. Defaults to the vector size of
            the data type on the host system.
        force_sequential: Whether to force sequential execution (default `False`).

    Args:
        x: The tensor.
        scalar: The scalar.

    Returns:
        The result of applying the binary operator to the inputs.

    Constraints:
        The data type must be a floating-point of single or double precision.
    """
    assert_float_dtype["dtype", dtype]()

    var result = Tensor[dtype](x.shape())
    _elementwise_impl[func, dtype, simd_width, force_sequential](x, scalar, result)

    return result


fn elementwise[
    func: BinaryOperator,
    dtype: DType,
    simd_width: Int = simdwidthof[dtype](),
    force_sequential: Bool = False,
](x: Tensor[dtype], y: Tensor[dtype]) raises -> Tensor[dtype]:
    """Applies a binary operator to two tensors element-wise.

    Parameters:
        func: The binary operator to apply to the inputs.
        dtype: The input and output data type.
        simd_width: The SIMD vector width to use. Defaults to the vector size of
            the data type on the host system.
        force_sequential: Whether to force sequential execution (default `False`).

    Args:
        x: The first tensor.
        y: The second tensor.

    Returns:
       The result of applying the binary operator to the inputs.

    Constraints:
        The data type must be a floating-point of single or double precision. And it
        will raise an exception if the arguments do not have the same shape.
    """
    assert_float_dtype["dtype", dtype]()
    if x.shape() != y.shape():
        raise Error("The arguments `x` and `y` must have the same shape.")

    var result = Tensor[dtype](x.shape())
    _elementwise_impl[func, dtype, simd_width, force_sequential](x, y, result)

    return result


# ===----------------------------- benchmark ------------------------------=== #


fn run_benchmark[
    func: UnaryOperator,
    dtype: DType,
    simd_width: Int = simdwidthof[dtype](),
    force_sequential: Bool = False,
](
    x: Tensor[dtype],
    num_warmup: Int = 2,
    max_iters: Int = 100_000,
    min_runtime_secs: SIMD[DType.float64, 1] = 0.5,
    max_runtime_secs: SIMD[DType.float64, 1] = 1,
) -> benchmark.Report:
    """Runs a benchmark for a unary operator applied to a tensor element-wise.

    Benchmarking continues until `min_runtime_secs` has elapsed and either `max_iters`
    OR `max_runtime_secs` is achieved.

    Parameters:
        func: The binary operator to apply to the input. This is the function that will
            be benchmarked.
        dtype: The input data type.
        simd_width: The SIMD vector width to use. Defaults to the vector size of
            the data type on the host system.
        force_sequential: Whether to force sequential execution (default `False`).

    Args:
        x: The tensor.
        num_warmup: Number of warmup iterations to run before starting benchmarking
            (default `2`).
        max_iters: Max number of iterations to run (default `100_000`).
        min_runtime_secs: Lower bound on benchmarking time in secs (default `0.5`).
        max_runtime_secs: Upper bound on benchmarking time in secs (default `1`).

    Returns:
        A report containing statistics of the benchmark.

    Constraints:
        The data type must be a floating-point of single or double precision.
    """

    @always_inline
    @parameter
    fn test_fn():
        _ = elementwise[func, dtype, simd_width, force_sequential](x)

    return benchmark.run[test_fn](
        num_warmup=num_warmup,
        max_iters=max_iters,
        min_runtime_secs=min_runtime_secs,
        max_runtime_secs=max_runtime_secs,
    )


fn run_benchmark[
    func: BinaryOperator,
    dtype: DType,
    simd_width: Int = simdwidthof[dtype](),
    force_sequential: Bool = False,
](
    x: Tensor[dtype],
    scalar: SIMD[dtype, 1],
    num_warmup: Int = 2,
    max_iters: Int = 100_000,
    min_runtime_secs: SIMD[DType.float64, 1] = 0.5,
    max_runtime_secs: SIMD[DType.float64, 1] = 1,
) -> benchmark.Report:
    """
    Runs a benchmark for a binary operator applied to a tensor and a scalar element-wise.

    Benchmarking continues until `min_runtime_secs` has elapsed and either `max_iters`
    OR `max_runtime_secs` is achieved.

    Parameters:
        func: The binary operator to apply to the inputs. This is the function that will
            be benchmarked.
        dtype: The input data type.
        simd_width: The SIMD vector width to use. Defaults to the vector size of
            the data type on the host system.
        force_sequential: Whether to force sequential execution (default `False`).

    Args:
        x: The tensor.
        scalar: The scalar.
        num_warmup: Number of warmup iterations to run before starting benchmarking
            (default `2`).
        max_iters: Max number of iterations to run (default `100_000`).
        min_runtime_secs: Lower bound on benchmarking time in secs (default `0.5`).
        max_runtime_secs: Upper bound on benchmarking time in secs (default `1`).

    Returns:
        A report containing statistics of the benchmark.

    Constraints:
        The data type must be a floating-point of single or double precision.
    """

    @always_inline
    @parameter
    fn test_fn():
        _ = elementwise[func, dtype, simd_width, force_sequential](x, scalar)

    return benchmark.run[test_fn](
        num_warmup=num_warmup,
        max_iters=max_iters,
        min_runtime_secs=min_runtime_secs,
        max_runtime_secs=max_runtime_secs,
    )


fn run_benchmark[
    func: BinaryOperator,
    dtype: DType,
    simd_width: Int = simdwidthof[dtype](),
    force_sequential: Bool = False,
](
    x: Tensor[dtype],
    y: Tensor[dtype],
    num_warmup: Int = 2,
    max_iters: Int = 100_000,
    min_runtime_secs: SIMD[DType.float64, 1] = 0.5,
    max_runtime_secs: SIMD[DType.float64, 1] = 1,
) raises -> benchmark.Report:
    """Runs a benchmark for a binary operator applied to two tensors element-wise.

    Benchmarking continues until `min_runtime_secs` has elapsed and either `max_iters`
    OR `max_runtime_secs` is achieved.

    Parameters:
        func: The binary operator to apply to the inputs. This is the function that will
            be benchmarked.
        dtype: The input data type.
        simd_width: The SIMD vector width to use. Defaults to the vector size of
            the data type on the host system.
        force_sequential: Whether to force sequential execution (default `False`).

    Args:
        x: The first tensor.
        y: The second tensor.
        num_warmup: Number of warmup iterations to run before starting benchmarking
            (default `2`).
        max_iters: Max number of iterations to run (default `100_000`).
        min_runtime_secs: Lower bound on benchmarking time in secs (default `0.5`).
        max_runtime_secs: Upper bound on benchmarking time in secs (default `1`).

    Returns:
        A report containing statistics of the benchmark.

    Constraints:
        The data type must be a floating-point of single or double precision. And it
        will raise an exception if the arguments do not have the same shape.
    """
    if x.shape() != y.shape():
        raise Error("The arguments `x` and `y` must have the same shape.")

    @always_inline
    @parameter
    fn test_fn():
        try:
            _ = elementwise[func, dtype, simd_width, force_sequential](x, y)
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
    dtype: DType, simd_width: Int = simdwidthof[dtype]()
](
    min_value: SIMD[dtype, 1],
    max_value: SIMD[dtype, 1],
    *shape: Int,
) raises -> Tensor[
    dtype
]:
    """Generates a tensor with random values drawn from a uniform distribution.

    Parameters:
        dtype: The data type of the tensor.
        simd_width: The SIMD vector width to use. Defaults to the vector size of
            the data type on the host system.

    Args:
        min_value: The lower bound of the uniform distribution.
        max_value: The upper bound of the uniform distribution.
        shape: The tensor shape.

    Returns:
        The tensor with random values drawn from a uniform distribution.

    Constraints:
        The data type must be a floating-point of single or double precision. And it
        will raise an exception if `min_value >= max_value`.
    """
    assert_float_dtype["dtype", dtype]()

    if min_value >= max_value:
        raise Error("`min_value` must be less than `max_value`.")

    let raw = random.rand[dtype](shape)
    let scaled = elementwise[math.mul, dtype, simd_width](raw, max_value - min_value)
    let shifted = elementwise[math.add, dtype, simd_width](scaled, min_value)

    return shifted


# ===------------------------------- numpy --------------------------------=== #


fn tensor_to_numpy_array[dtype: DType](x: Tensor[dtype]) raises -> PythonObject:
    """Converts a tensor to a NumPy array.

    Parameters:
        dtype: The data type of the tensor.

    Args:
        x: The tensor.

    Returns:
        The NumPy array with the same shape, data type, and elements as the tensor.

    Constraints:
        Will raise an exception if it is not possible to convert the tensor to a NumPy
        array with the same shape, data type, and elements.
    """
    let builtins = Python.import_module("builtins")
    let np = Python.import_module("numpy")

    let shape = builtins.tuple(
        builtins.map(builtins.int, builtins.str(x.shape().__str__()).split("x"))
    )
    let num_elements = x.num_elements()

    let numpy_array = np.zeros(num_elements, dtype.__str__())

    for i in range(num_elements):
        _ = np.put(numpy_array, i, x[i])

    return numpy_array.reshape(shape)
