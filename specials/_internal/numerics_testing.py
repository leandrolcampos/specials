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
#
# References:
#
# Beebe, N. H. (2017). The Mathematical-Function Computation Handbook:
#   Programming Using the MathCW Portable Software Library.
# Springer International Publishing.
# https://doi.org/10.1007/978-3-319-64110-2
#
# Brown, B. W., & Levy, L. B. (1994). Certification of Algorithm 708:
#   significant-digit computation of the incomplete beta.
# ACM Transactions on Mathematical Software (TOMS), 20(3), 393-397.
# https://dl.acm.org/doi/10.1145/192115.192155

"""Numerical testing utilities."""

import numpy as np


def py_relative_error(result, truth):
    """Computes the relative error of `result` relative to `truth`.

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

    To avoid taking the logarithm of `0`, the relative error is bounded below by the
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


def py_kahan_ulp(x, output_dtype):
    """Computes the Kahan-ulp function at `x` element-wise.

    Ulp stands for "unit in the last place".

    If `x` is a floating-point number, the Kahan-ulp at `x` is the distance between
    the two floating-point numbers nearest `x`, even if `x` is not contained in the
    interval between them.

    Args:
        x: Array of floating-point values.
        output_dtype: Data type of the output array.

    Returns:
        Array of element-wise Kahan-ulp values.

    Raises:
        TypeError: If `x.dtype` or `output_dtype` is not a floating-point type.
        TypeError: If `output_dtype` has higher precision than `x.dtype`.
    """
    x = np.array(x)

    dtype = x.dtype
    output_dtype = np.dtype(output_dtype)

    if dtype.kind != "f":
        raise TypeError("`x.dtype` must be a floating-point type")

    if output_dtype.kind != "f":
        raise TypeError("`output_dtype` must be a floating-point type")

    if np.finfo(output_dtype).precision > np.finfo(dtype).precision:
        raise TypeError("`output_dtype` cannot have higher precision than `x.dtype`")

    dtype = dtype.type
    output_dtype = output_dtype.type

    xmin = np.finfo(output_dtype).smallest_normal
    xeps = np.finfo(output_dtype).eps
    emax = np.finfo(output_dtype).maxexp - 1
    emin = np.finfo(output_dtype).minexp - 1
    t = np.finfo(output_dtype).nmant
    esub = emin - t + 1
    one = dtype(1.0)

    result = np.empty_like(x)
    result = np.where(np.isnan(x.astype(output_dtype)), np.nan, result)
    result = np.where(
        np.isinf(x.astype(output_dtype)), np.ldexp(one, emax - t + 1), result
    )
    result = np.where(np.abs(x) <= xmin, np.ldexp(one, esub), result)

    is_main_branch = np.isfinite(x.astype(output_dtype)) & (np.abs(x) > xmin)
    y = np.where(is_main_branch, x, one)

    _, e = np.frexp(y)
    s = np.ldexp(np.abs(y), -e)
    e -= t - 1

    e = np.where((s - one) < (0.5 * xeps), e - 1, e)
    e = np.maximum(e, esub)
    result = np.where(is_main_branch, np.ldexp(one, e), result)

    result = result.astype(output_dtype)
    result = np.where(result == 0.0, xmin, result)

    if result.shape == ():
        return output_dtype(result.item(0))

    return result


def _promote_dtype(dtype):
    """Promotes `dtype` to a higher precision floating-point type."""
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise TypeError("`dtype` must be a floating-point type")

    if np.finfo(dtype).bits < 32:
        return np.float32

    if np.finfo(dtype).bits < 64:
        return np.float64

    if np.finfo(dtype).bits < 128:
        return np.float128

    return dtype


def py_error_in_ulps(result, truth):
    """Computes the error in ulps of `result` relative to `truth`.

    Ulp stands for "unit in the last place".

    The error in ulps is defined as `abs((result - truth) / kahan_ulp(truth))`. See the
    `kahan_ulp` function for more details.

    If `truth.dtype` is not a floating-point type, it is cast to the following type:
        - `np.float32` if `result.dtype.bits < 32`;
        - `np.float64` if `32 <= result.dtype.bits < 64`;
        - `np.float128` if `64 <= result.dtype.bits < 128`; or
        - `result.dtype` otherwise.

    Args:
        result: Array of values whose deviation to assess.
        truth: Array of values presumed correct. Must broadcast with `result`.

    Returns:
        Array of element-wise error in ulps values.

    Raises:
        TypeError: If `result.dtype` is not a floating-point type.
        TypeError: If `result.dtype` has higher precision than `truth.dtype`.
    """
    result = np.array(result)

    if result.dtype.kind != "f":
        raise TypeError("`result.dtype` must be a floating-point type")

    truth = np.array(truth)

    if truth.dtype.kind != "f":
        truth = truth.astype(_promote_dtype(result.dtype))

    ulp_truth = py_kahan_ulp(truth, output_dtype=result.dtype)

    return np.abs((result - truth).astype(result.dtype) / ulp_truth)
