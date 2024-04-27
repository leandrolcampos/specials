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
# Muller, J. M. (2005). On the definition of ulp (x).
# [Research Report] Laboratoire de l’informatique duparallélisme. 2005, 2+11p.
# https://inria.hal.science/inria-00070503/file/RR2005-09.pdf

"""Numerical testing utilities."""

import numpy as np


def _promote_dtype(dtype):
    """Promotes `dtype` to a higher precision floating-point type."""
    dtype = np.dtype(dtype)

    assert dtype in (np.float32, np.float64)

    if dtype == np.float32:
        return np.float64

    return np.longdouble


def relative_error(result, truth):
    """Computes the relative error of `result` relative to `truth`.

    The relative error is defined as `abs((result - truth) / truth)`. The
    computation of that difference and ratio are done in the next higher
    precision compared to the working precision.

    The data types `result.dtype` and `truth.dtype` must correspond to the
    working precision and a higher precision floating-point types, respectively.

    Args:
        result: A NumPy array of values whose deviation to assess.
        truth: A NumPy Array of values presumed correct. Must broadcast with
            `result`.

    Returns:
        A NumPy array of element-wise relative error values. The data type of
        the output array is the next higher precision floating-point type
        compared to the data type of `result`:
            - `np.float64` if `result.dtype` is `np.float32`; or
            - `np.longdouble` if `result.dtype` is `np.float64`.

    Raises:
        TypeError: If `result.dtype` is not a floating-point of single
            (`float32`) or double (`float64`) precision.
        TypeError: If `truth.dtype` is a floating-point type but does not have
            higher precision than `result.dtype`.
    """
    result = np.array(result)

    if result.dtype not in (np.float32, np.float64):
        raise TypeError(
            "`result.dtype` must be a floating-point of single (`float32`) or "
            "double (`float64`) precision"
        )

    output_dtype = _promote_dtype(result.dtype)
    truth = np.array(truth)

    if truth.dtype.kind == "f":
        if np.finfo(truth.dtype).precision <= np.finfo(result.dtype).precision:
            raise TypeError(
                "When `truth.dtype` is a floating-point type, it must have "
                "higher precision than `result.dtype`"
            )

    result = result.astype(output_dtype)
    truth = truth.astype(output_dtype)

    truth_is_zero = np.equal(truth, 0.0)
    largest_subnormal = np.nextafter(np.finfo(truth.dtype).smallest_normal, 0.0)
    truth_is_subnormal = ~truth_is_zero & (np.abs(truth) <= largest_subnormal)

    safe_denominator = np.where(
        truth_is_zero,
        np.ones_like(truth),
        np.where(
            (result > 0) & (truth > 0) & truth_is_subnormal,
            largest_subnormal,
            truth,
        ),
    )

    relerr = np.abs((result - truth) / safe_denominator)
    relerr = np.where(truth_is_zero, np.inf, relerr)
    relerr = np.where(result == truth, np.zeros_like(truth), relerr)

    return relerr


def kahan_ulp(x, target_dtype):
    """Computes the Kahan-ulp function at `x` element-wise.

    Ulp stands for "unit in the last place".

    If `x` is an arbitrary real number, the Kahan-ulp at `x` is the distance
    between the two floating-point numbers nearest `x`, even if `x` is not
    contained in the interval between them.

    The computation of the Kahan-ulp is done in the next higher precision
    compared to the working precision.

    Args:
        x: Array of floating-point values. The data type of this array must
            have higher precision than `target_dtype`.
        target_dtype: Data type corresponding to the working precision.

    Returns:
        Array of element-wise Kahan-ulp values. The data type of the output
        array is the next higher precision floating-point type compared to
        the data type of `x`:
            - `np.float64` if `result.dtype` is `np.float32`; or
            - `np.longdouble` if `result.dtype` is `np.float64`.

    Raises:
        TypeError: If `target_dtype` is not a floating-point of single
            (`float32`) or double (`float64`) precision.
        TypeError: If `x.dtype` is a floating-point type but does not have
            higher precision than `target_dtype`.
    """
    target_dtype = np.dtype(target_dtype)
    if target_dtype not in (np.float32, np.float64):
        raise TypeError(
            "`target_dtype` must be a floating-point of single (`float32`) or "
            "double (`float64`) precision"
        )

    x = np.array(x)
    if x.dtype.kind == "f":
        if np.finfo(x.dtype).precision <= np.finfo(target_dtype).precision:
            raise TypeError(
                "When `x.dtype` is a floating-point type, it must have higher "
                "precision than `target_dtype`"
            )

    higher_precision_dtype = _promote_dtype(target_dtype)
    x = x.astype(higher_precision_dtype)

    x = np.abs(x)
    one = higher_precision_dtype(1)

    emax = np.finfo(target_dtype).maxexp
    emin = np.finfo(target_dtype).minexp + 1
    t = np.finfo(target_dtype).nmant
    xsml = np.ldexp(one, emin)
    xmax = np.finfo(target_dtype).max

    is_normal = (x >= xsml) & (x <= xmax)

    result = np.empty_like(x)
    result = np.where(np.isnan(x), np.nan, result)
    result = np.where(x < xsml, np.ldexp(one, emin - (t + 1)), result)
    result = np.where(x > xmax, np.ldexp(one, emax - (t + 1)), result)

    safe_x = np.where(is_normal, x, one)
    expmin = np.log2(safe_x).astype(np.int32)
    exponent = expmin - t

    powermin = np.ldexp(one, expmin)
    exponent = np.where(
        safe_x / powermin <= one + np.ldexp(one, -(t + 2)),
        exponent - 1,
        exponent,
    )

    result = np.where(is_normal, np.ldexp(one, exponent), result)

    if result.shape == ():
        return higher_precision_dtype(result.item(0))

    return result


def error_in_ulps(result, truth):
    """Computes the error in ulps of `result` relative to `truth`.

    Ulp stands for "unit in the last place".

    The error in ulps is defined as `abs((result - truth) / kahan_ulp(truth))`.
    The computation of that difference and ratio are done in the next higher
    precision compared to the working precision.

    The data types `result.dtype` and `truth.dtype` must correspond to the
    working precision and a higher precision floating-point types, respectively.

    Args:
        result: Array of values whose deviation to assess.
        truth: Array of values presumed correct. Must broadcast with `result`.

    Returns:
        A NumPy array of element-wise error in ulps values. The data type of
        the output array is the next higher precision floating-point type
        compared to the data type of `result`:
            - `np.float64` if `result.dtype` is `np.float32`; or
            - `np.longdouble` if `result.dtype` is `np.float64`.

    Raises:
        TypeError: If `result.dtype` is not a floating-point of single
            (`float32`) or double (`float64`) precision.
        TypeError: If `truth.dtype` is a floating-point type but does not have
            higher precision than `result.dtype`.
    """
    result = np.array(result)

    if result.dtype not in (np.float32, np.float64):
        raise TypeError(
            "`result.dtype` must be a floating-point of single (`float32`) or "
            "double (`float64`) precision"
        )

    target_dtype = result.dtype
    output_dtype = _promote_dtype(target_dtype)
    truth = np.array(truth)

    if truth.dtype.kind == "f":
        if np.finfo(truth.dtype).precision <= np.finfo(result.dtype).precision:
            raise TypeError(
                "When `truth.dtype` is a floating-point type, it must have "
                "higher precision than `result.dtype`"
            )

    result = result.astype(output_dtype)
    truth = truth.astype(output_dtype)

    ulp_truth = kahan_ulp(truth, target_dtype=target_dtype)

    return np.abs((result - truth) / ulp_truth)
