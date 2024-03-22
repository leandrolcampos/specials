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

"""Tests for numerical testing utilities."""

import numpy as np
import pytest


import numerics_testing


@pytest.mark.parametrize(
    "result_dtype, truth_dtype",
    [
        (np.float32, np.dtype("O")),
        (np.float32, np.float64),
        (np.float64, np.dtype("O")),
        (np.float64, np.longdouble),
    ],
)
def test_py_relative_error(result_dtype, truth_dtype):
    result = np.array([0.0, 1.1, 2.0, 3.0], dtype=result_dtype)
    truth = np.array([0.0, 1.0, 0.0, 3.0], dtype=truth_dtype)

    output_dtype = numerics_testing._promote_dtype(result_dtype)
    actual = numerics_testing.py_relative_error(result, truth)
    expected = np.array([0.0, 0.1, np.inf, 0.0], dtype=output_dtype)

    np.testing.assert_equal(actual.dtype, output_dtype)
    if result_dtype == np.float32:
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0.0)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0.0)


def test_py_relative_error_invalid_result_dtype():
    result = np.array([0.0, 1.1, 2.0, 3.0], dtype=np.float16)
    truth = np.array([0.0, 1.0, 0.0, 3.0], dtype=np.float32)

    with pytest.raises(TypeError, match="single .* or double"):
        _ = numerics_testing.py_relative_error(result, truth)


def test_py_relative_error_invalid_truth_dtype():
    result = np.array([0.0, 1.1, 2.0, 3.0], dtype=np.float32)
    truth = np.array([0.0, 1.0, 0.0, 3.0], dtype=np.float32)

    with pytest.raises(TypeError, match="higher precision"):
        _ = numerics_testing.py_relative_error(result, truth)


@pytest.mark.parametrize(
    "x, target_dtype, expected",
    [
        (
            np.array(
                [np.nan, -1.0, 0.0, 1.0, 5.0, np.finfo(np.float64).max, np.inf],
                dtype=np.longdouble,
            ),
            np.float64,
            np.array(
                [
                    np.nan,
                    1.1102230246251565404e-16,
                    4.940656458412465442e-324,
                    1.1102230246251565404e-16,
                    8.8817841970012523234e-16,
                    1.9958403095347198117e292,
                    1.9958403095347198117e292,
                ],
                dtype=np.longdouble,
            ),
        ),
        (
            np.array(
                [np.nan, -1.0, 0.0, 1.0, 5.0, np.finfo(np.float32).max, np.inf],
                dtype=np.float64,
            ),
            np.float32,
            np.array(
                [
                    np.nan,
                    5.960464477539063e-08,
                    1.401298464324817e-45,
                    5.960464477539063e-08,
                    4.76837158203125e-07,
                    2.028240960365167e31,
                    2.028240960365167e31,
                ],
                dtype=np.float64,
            ),
        ),
    ],
)
def test_py_kahan_ulp(x, target_dtype, expected):
    output_dtype = numerics_testing._promote_dtype(target_dtype)
    actual = numerics_testing.py_kahan_ulp(x, target_dtype)

    np.testing.assert_equal(actual.dtype, output_dtype)
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0.0)


def test_py_kahan_ulp_invalid_x_dtype():
    x = np.array([0.0, 1.0], dtype=np.float32)
    target_dtype = np.float32

    with pytest.raises(TypeError, match="higher precision"):
        numerics_testing.py_kahan_ulp(x, target_dtype)


def test_py_kahan_ulp_invalid_target_dtype():
    x = np.array([0.0, 1.0], dtype=np.float64)
    target_dtype = np.int32

    with pytest.raises(TypeError, match="single .* or double"):
        numerics_testing.py_kahan_ulp(x, target_dtype)


@pytest.mark.parametrize(
    "result, truth, expected",
    [
        (
            np.float64(1.0) + np.finfo(np.float64).eps,
            np.longdouble(1.0),
            np.longdouble(2.0),
        ),
        (
            np.float64(1.0) + np.finfo(np.float64).eps,
            np.int64(1.0),
            np.longdouble(2.0),
        ),
        (
            np.float32(1.0) + np.finfo(np.float32).eps,
            np.float64(1.0),
            np.float64(2.0),
        ),
        (
            np.float32(1.0) + np.finfo(np.float32).eps,
            np.int32(1.0),
            np.float64(2.0),
        ),
    ],
)
def test_py_error_in_ulps(result, truth, expected):
    output_dtype = numerics_testing._promote_dtype(result.dtype)
    actual = numerics_testing.py_error_in_ulps(result, truth)

    np.testing.assert_equal(actual.dtype, output_dtype)
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0.0)


def test_py_error_in_ulps_invalid_result_dtype():
    result = np.int32(1)
    truth = np.float64(1.0)

    with pytest.raises(TypeError, match="single .* or double"):
        numerics_testing.py_error_in_ulps(result, truth)


def test_py_error_in_ulps_invalid_truth_dtype():
    result = np.float64(1.0)
    truth = np.float32(1.0)

    with pytest.raises(TypeError, match="higher precision"):
        numerics_testing.py_error_in_ulps(result, truth)
