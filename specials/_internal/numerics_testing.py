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
#
# Some of the code in this file is adapted from:
#
# tensorflow/probability:
# Copyright 2018 The TensorFlow Probability Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Numerical testing utilities."""

import numpy as np


def py_relative_error(result, truth):
    """Computes the relative error of `result` relative `truth`.

    The relative error is defined as `abs((result - truth) / truth)`. The computation
    of that difference and ratio are done in 64-bit precision.

    Args:
        result: Array of values whose deviation to assess.
        truth: Array of values presumed correct. Must broadcast with `result`.

    Returns:
        Float64 array of elementwise relative error values.
    """
    result = np.array(result, dtype=np.float64)
    truth = np.array(truth, dtype=np.float64)

    truth_is_zero = np.equal(truth, 0.0)
    safe_truth = np.where(truth_is_zero, np.ones_like(truth), truth)

    relerr = np.abs((result - truth) / safe_truth)
    relerr = np.where(truth_is_zero, np.inf, relerr)
    relerr = np.where(result == truth, np.zeros_like(truth), relerr)

    return relerr


def py_accuracy_in_significant_digits(relerr):
    """Computes the number of significant digits of accuracy from the relative error.

    The number of significant digits of accuracy is defined as `-log10(2 * relerr)`,
    which is the usual definition: `n` significant digits of accuracy in a value allows
    an error of `5` in the `(n + l)`st decimal place.

    To avoid taking the logarithm of `O`, the relative error is bounded below by the
    `epsneg` value of `float64`: the smallest positive floating-point number such that
    `1.0 - epsneg != 1.0`.

    The computation of this metric is done in 64-bit precision.

    Args:
        relerr: Array of relative error values.

    Returns:
        Float64 array of elementwise significant digits of accuracy.
    """
    relerr = np.array(relerr, dtype=np.float64)
    return -np.log10(2.0 * np.maximum(relerr, np.finfo(np.float64).epsneg))
