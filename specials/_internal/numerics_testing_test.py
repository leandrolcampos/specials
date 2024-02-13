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


def test_py_relative_error():
    result = np.array([0.0, 1.1, 2.0, 3.0])
    truth = np.array([0.0, 1.0, 0.0, 3.0])

    actual = numerics_testing.py_relative_error(result, truth)
    expected = np.array([0.0, 0.1, np.inf, 0.0])

    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0.0)


def test_py_accuracy_in_significant_digits():
    relerr = np.array([0.0, 0.1, np.inf, 0.0])

    actual = numerics_testing.py_accuracy_in_significant_digits(relerr)
    expected = np.array(
        [15.653559774527022, 0.6989700043360187, -np.inf, 15.653559774527022]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0.0)


@pytest.mark.parametrize(
    "x, output_dtype, expected",
    [
        (
            np.array(
                [np.nan, -1.0, 0.0, 1.0, np.finfo(np.float64).max, np.inf],
                dtype=np.float128,
            ),
            np.float64,
            np.array(
                [
                    np.nan,
                    4.440892098500626e-16,
                    5e-324,
                    4.440892098500626e-16,
                    3.99168061906944e292,
                    3.99168061906944e292,
                ],
                dtype=np.float64,
            ),
        ),
        (
            np.array(
                [np.nan, -1.0, 0.0, 1.0, np.finfo(np.float32).max, np.inf],
                dtype=np.float64,
            ),
            np.float32,
            np.array(
                [
                    np.nan,
                    2.3841858e-07,
                    1e-45,
                    2.3841858e-07,
                    4.056482e31,
                    4.056482e31,
                ],
                dtype=np.float32,
            ),
        ),
    ],
)
def test_py_kahan_ulp(x, output_dtype, expected):
    actual = numerics_testing.py_kahan_ulp(x, output_dtype)

    if output_dtype == np.float32:
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0.0)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0.0)


def test_py_kahan_ulp_invalid_dtype():
    x = np.array([0.0, 1.0], dtype=np.int64)
    output_dtype = np.float32

    with pytest.raises(TypeError):
        numerics_testing.py_kahan_ulp(x, output_dtype)


def test_py_kahan_ulp_invalid_output_dtype():
    x = np.array([0.0, 1.0], dtype=np.float64)
    output_dtype = np.int32

    with pytest.raises(TypeError):
        numerics_testing.py_kahan_ulp(x, output_dtype)


def test_py_kahan_ulp_higher_precision_output_dtype():
    x = np.array([0.0, 1.0], dtype=np.float32)
    output_dtype = np.float64

    with pytest.raises(TypeError):
        numerics_testing.py_kahan_ulp(x, output_dtype)


@pytest.mark.parametrize(
    "result, truth, expected",
    [
        (np.float64(1.0) + np.finfo(np.float64).eps, np.float128(1.0), np.float64(0.5)),
        (np.float64(1.0) + np.finfo(np.float64).eps, np.float64(1.0), np.float64(0.5)),
        (np.float64(1.0) + np.finfo(np.float64).eps, np.int64(1), np.float64(0.5)),
        (np.float32(1.0) + np.finfo(np.float32).eps, np.float64(1.0), np.float32(0.5)),
    ],
)
def test_py_error_in_ulps(result, truth, expected):
    actual = numerics_testing.py_error_in_ulps(result, truth)

    if result.dtype == np.float32:
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0.0)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0.0)


def test_py_error_in_ulps_invalid_dtype():
    result = np.int32(1)
    truth = np.float64(1.0)

    with pytest.raises(TypeError):
        numerics_testing.py_error_in_ulps(result, truth)


def test_py_error_in_ulps_higher_precision_result_dtype():
    result = np.float64(1)
    truth = np.float32(1.0)

    with pytest.raises(TypeError):
        numerics_testing.py_error_in_ulps(result, truth)
