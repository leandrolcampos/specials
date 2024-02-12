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

from numerics_testing import py_relative_error, py_accuracy_in_significant_digits


def test_py_relative_error():
    result = np.array([0.0, 1.1, 2.0, 3.0])
    truth = np.array([0.0, 1.0, 0.0, 3.0])

    actual = py_relative_error(result, truth)
    expected = np.array([0.0, 0.1, np.inf, 0.0])

    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0.0)


def test_py_accuracy_in_significant_digits():
    relerr = np.array([0.0, 0.1, np.inf, 0.0])

    actual = py_accuracy_in_significant_digits(relerr)
    expected = np.array(
        [15.653559774527022, 0.6989700043360187, -np.inf, 15.653559774527022]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0.0)
