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

"""Implements higher-order functions."""

from .asserting import assert_non_zero


fn _fori_loop_impl[
    start: Int, stop: Int, step: Int, body_func: fn[index: Int] () capturing -> None
]() -> None:
    """Implements `fori_loop` using recursion."""

    @parameter
    if (step > 0 and start < stop) or (step < 0 and start > stop):
        body_func[start]()
        _fori_loop_impl[start + step, stop, step, body_func]()


fn fori_loop[
    start: Int, stop: Int, step: Int, body_func: fn[index: Int] () capturing -> None
]() -> None:
    """Applies a for loop.

    Parameters:
        start: The loop index lower bound (inclusive).
        stop: The loop index upper bound (exclusive).
        step: The loop index increment.
        body_func: The function to apply at each iteration. The function should take
            a single Int parameter and not return a value.

    Constraints:
        The `step` parameter must be non-zero.
    """
    assert_non_zero["step", step]()

    _fori_loop_impl[start, stop, step, body_func]()


fn _fori_loop_impl[
    start: Int, stop: Int, step: Int, body_func: fn[index: Int] () capturing -> Bool
]() -> None:
    """Implements `fori_loop` with conditional execution using recursion."""

    @parameter
    if (step > 0 and start < stop) or (step < 0 and start > stop):
        if body_func[start]():
            _fori_loop_impl[start + step, stop, step, body_func]()


fn fori_loop[
    start: Int, stop: Int, step: Int, body_func: fn[index: Int] () capturing -> Bool
]() -> None:
    """Applies a for loop with conditional execution.

    Parameters:
        start: The loop index lower bound (inclusive).
        stop: The loop index upper bound (exclusive).
        step: The loop index increment.
        body_func: The function to apply at each iteration. The function should take
            a single Int parameter and return a boolean value. The loop continues only
            if the result of `body_func` is `True`; otherwise, it terminates.

    Constraints:
        The `step` parameter must be non-zero.
    """
    assert_non_zero["step", step]()

    _fori_loop_impl[start, stop, step, body_func]()
