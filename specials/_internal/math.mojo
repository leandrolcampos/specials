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

"""Defines math utilities for internal use."""

import math


fn log[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Performs elementwise natural log (base E) of a SIMD vector.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        simd_width: The width of the input and output SIMD vector.

    Args:
        x: Vector to perform logarithm operation on.

    Returns:
        Vector containing result of performing natural log base E on x.
    """
    # TODO: Use stdlib function `math.log` when its accuracy loss is fixed.
    # https://github.com/modularml/mojo/issues/1303
    alias log10_e: SIMD[dtype, simd_width] = 0.43429448190325182765112891891661
    return math.log10(x) / log10_e
