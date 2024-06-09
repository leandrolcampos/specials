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

import random

from python import Python
from python.object import PythonObject
from sys.info import simdwidthof
from tensor import Tensor
from utils.static_tuple import StaticTuple

from test_utils.tensor import (
    elementwise,
    random_uniform,
    run_benchmark,
    tensor_to_numpy_array,
    UnaryOperator,
)


fn _solution_report[
    solution_name: StringLiteral,
    func: UnaryOperator,
    *,
    type: DType,
    simd_width: Int,
    force_sequential: Bool = False,
](x: Tensor[type], truth: PythonObject) raises -> PythonObject:
    """Computes the evaluation metrics for a numerical solution in Mojo."""
    var builtins = Python.import_module("builtins")
    var np = Python.import_module("numpy")
    var py_utils = Python.import_module("test_utils")

    var result = elementwise[func, type=type, simd_width=simd_width](x)
    var msecs = run_benchmark[
        func,
        type=type,
        simd_width=simd_width,
        force_sequential=force_sequential,
    ](x).mean("ms")
    var err = py_utils.numerics_testing.error_in_ulps(
        tensor_to_numpy_array(result), truth
    )

    var report = builtins.list()
    _ = report.append(solution_name)
    _ = report.append(py_utils.benchmark.format_float(np.max(err)))
    _ = report.append(
        py_utils.benchmark.format_float(np.sqrt(np.mean(np.square(err))))
    )
    _ = report.append(py_utils.benchmark.format_float(msecs))

    return report


fn run_experiment[
    *,
    num_domains: Int,
    specials_func: UnaryOperator,
    mojo_func: UnaryOperator,
    type: DType,
    simd_width: Int = simdwidthof[type](),
    force_sequential: Bool = False,
    mojo_func_name: StringLiteral = "Mojo Stdlib",
](
    experiment_name: StringLiteral,
    *,
    num_samples: Int,
    min_values: StaticTuple[Scalar[type], num_domains],
    max_values: StaticTuple[Scalar[type], num_domains],
    truth_func: PythonObject,
    python_func: PythonObject,
    python_func_name: StringLiteral = "Python",
) raises:
    """Runs a given experiment."""
    var builtins = Python.import_module("builtins")
    var py_utils = Python.import_module("test_utils")

    random.seed(42)

    var domain_names = builtins.list()
    var data = builtins.list()

    var num_solutions = 3

    for i in range(len(max_values)):
        var min_value = min_values[i]
        var max_value = max_values[i]

        _ = domain_names.append(
            py_utils.benchmark.format_domain_name(min_value, max_value)
        )

        var a = random_uniform[type, simd_width=simd_width](
            min_value, max_value, num_samples
        )
        var a_arr = tensor_to_numpy_array(a)

        # Truth function
        var truth = truth_func(a_arr)

        # Specials function
        var specials_report = _solution_report[
            "Specials",
            specials_func,
            type=type,
            simd_width=simd_width,
            force_sequential=force_sequential,
        ](a, truth)
        _ = data.append(specials_report)

        # Mojo function
        var mojo_report = _solution_report[
            mojo_func_name,
            mojo_func,
            type=type,
            simd_width=simd_width,
            force_sequential=force_sequential,
        ](a, truth)
        _ = data.append(mojo_report)

        # Python function
        var python_report = py_utils.benchmark.solution_report(
            python_func_name, python_func, a_arr, truth
        )
        _ = data.append(python_report)

    _ = py_utils.benchmark.print_experiment_results(
        data,
        domain_names,
        num_solutions,
        String(experiment_name) + " (" + str(type) + ")",
    )
