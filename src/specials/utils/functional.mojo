# ===----------------------------------------------------------------------=== #
# Copyright 2024 The Specials Authors.
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

"""Implements higher-order functions."""

import math

from algorithm.functional import parallelize
from sys.info import num_performance_cores, simdwidthof


# ===----------------------------------------------------------------------=== #
# Elementwise
# ===----------------------------------------------------------------------=== #


fn elementwise[
    func: fn[Int] (Int, /) capturing -> None,
    *,
    simd_width: Int,
](num_elements: Int):
    """Executes `func[simd_width](index)`, possibly as subtasks, for each index
    in the range `[0, num_elements)`.

    Constraints:
        The SIMD width must be a positive integer value.

    Parameters:
        func: The function to apply at each index. It should not return a value.
        simd_width: The SIMD width to use.

    Arguments:
        num_elements: The number of elements to process.
    """
    elementwise[func, simd_width=simd_width](
        num_elements, min_simds_per_core=1024 // num_performance_cores()
    )


fn elementwise[
    func: fn[Int] (Int, /) capturing -> None,
    *,
    simd_width: Int,
](num_elements: Int, *, min_simds_per_core: Int):
    """Executes `func[simd_width](index)`, possibly as subtasks, for each index
    in the range `[0, num_elements)`.

    Constraints:
        The SIMD width must be a positive integer value.

    Parameters:
        func: The function to apply at each index. It should not return a value.
        simd_width: The SIMD width to use.

    Arguments:
        num_elements: The number of elements to process.
        min_simds_per_core: The minimum number of SIMD vectors per performance
            core to try to execute the `func` function in parallel.
    """
    constrained[simd_width > 0, "SIMD width must be a positive integer value"]()

    debug_assert(
        num_elements > 0, "number of elements must be a positive integer value"
    )
    debug_assert(
        min_simds_per_core > 0,
        "minimum SIMD vectors per core must be a positive integer value",
    )

    var num_simds = num_elements // simd_width
    var num_cores = num_performance_cores()
    var num_workers = 1

    if num_simds >= (num_cores * max(min_simds_per_core, 1)):
        num_workers = num_cores

    var num_tasks = num_workers
    var num_elements_per_tasks = (num_simds // num_tasks) * simd_width
    var num_remaining_elements = (
        num_elements - num_elements_per_tasks * num_tasks
    )

    @always_inline
    @parameter
    fn task_fn(task_id: Int):
        var start = task_id * num_elements_per_tasks
        var end: Int

        if task_id == (num_tasks - 1):
            end = num_simds * simd_width
        else:
            end = start + num_elements_per_tasks

        for index in range(start, end, simd_width):
            func[simd_width](index)

    if num_tasks > 1:
        parallelize[task_fn](num_tasks, num_workers)
    else:
        task_fn(0)

    if num_remaining_elements > 0:
        var start = num_elements - simd_width

        if start >= 0:
            func[simd_width](start)
        else:
            start = num_elements - num_remaining_elements
            for index in range(start, num_elements):
                func[1](index)


# ===----------------------------------------------------------------------=== #
# For Loop
# ===----------------------------------------------------------------------=== #


@always_inline
fn _fori_loop_impl[
    func: fn[Int, /] () capturing -> None,
    start: Int,
    end: Int,
    step: Int,
]():
    """Implements `fori_loop` using recursion."""

    @parameter
    if (step > 0 and start < end) or (step < 0 and start > end):
        func[start]()
        _fori_loop_impl[func, start + step, end, step]()


@always_inline
fn fori_loop[
    func: fn[Int, /] () capturing -> None,
    start: Int,
    end: Int,
    step: Int,
]():
    """Applies a for loop.

    Constraints:
        The step must be a non-zero integer value.

    Parameters:
        func: The function to apply at each iteration. The function should take
            a single Int parameter and not return a value.
        start: The loop index lower bound (inclusive).
        end: The loop index upper bound (exclusive).
        step: The loop index increment.
    """
    constrained[step != 0, "step must be a non-zero integer value"]()

    _fori_loop_impl[func, start, end, step]()


@always_inline
fn _fori_loop_impl[
    func: fn[Int, /] () capturing -> Bool,
    start: Int,
    end: Int,
    step: Int,
]():
    """Implements `fori_loop` with conditional execution using recursion."""

    @parameter
    if (step > 0 and start < end) or (step < 0 and start > end):
        if func[start]():
            _fori_loop_impl[func, start + step, end, step]()


@always_inline
fn fori_loop[
    func: fn[Int, /] () capturing -> Bool,
    start: Int,
    end: Int,
    step: Int,
]():
    """Applies a for loop with conditional execution.

    Constraints:
        The step must be a non-zero integer value.

    Parameters:
        func: The function to apply at each iteration. The function should take
            a single Int parameter and return a boolean value. The loop continues
            only if the result of `func` is `True`; otherwise, it terminates.
        start: The loop index lower bound (inclusive).
        end: The loop index upper bound (exclusive).
        step: The loop index increment.
    """
    constrained[step != 0, "step must be a non-zero integer value"]()

    _fori_loop_impl[func, start, end, step]()
