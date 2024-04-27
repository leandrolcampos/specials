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

from timeit import timeit

import numpy as np

from tabulate import tabulate, SEPARATING_LINE  # type: ignore[import-untyped]

from test_utils.python import numerics_testing


def format_domain_name(min_value, max_value):
    """Formats domain name for printing."""
    values = [min_value, max_value]
    formatted = []

    for value in values:
        if value == int(value):
            if np.abs(value) > 10e3:
                formatted.append(f"{int(value):.0e}")
            else:
                formatted.append(f"{int(value)}")
        else:
            if np.abs(value) < 0.001:
                formatted.append(f"{value:.0e}")
            elif np.log10(np.abs(value)) >= 3:
                formatted.append(f"{value:.1e}")
            else:
                formatted.append(f"{value:.1f}")

    return f"{formatted[0]},{formatted[1]}"


def format_float(value):
    """Formats float values for printing."""
    if (value == 0.0) or (0.01 <= np.abs(value) < 1e9):
        return f"{value:,.3f}"
    else:
        return f"{value:.2e}"


def benchmark(func, *args):
    """Computes the average execution time of a Python function."""
    # Warmup phase
    _ = timeit(lambda: func(*args), number=2)

    msecs = 1000 * timeit(lambda: func(*args), number=100) / 100
    return msecs


def solution_report(solution_name, func, x_arr, truth):
    """Computes the evaluation metrics for a numerical solution in Python."""
    result = func(x_arr)
    msecs = benchmark(func, x_arr)
    err = numerics_testing.error_in_ulps(result, truth)

    return [
        solution_name,
        format_float(np.max(err)),
        format_float(np.sqrt(np.mean(np.square(err)))),
        format_float(msecs),
    ]


def print_experiment_results(data, domain_names, num_solutions, experiment_name):
    """Prints the evaluation metrics for all numerical solutions."""
    headers = [
        "\nDomain",
        "\nSolution",
        "Maximum Error\nObserved (ulps)",
        "RMS Error\nObserved (ulps)",
        "Average Execution\nTime (msecs)",
    ]

    # Insert domain names
    current_domain = 0
    for i, report in enumerate(data):
        if i % num_solutions == 0:
            domain_name = domain_names[current_domain]
            data[i].insert(0, domain_name)
            current_domain += 1
        else:
            data[i].insert(0, "")

    # Insert horizontal lines between domains
    for index in range(num_solutions, len(data) + num_solutions, num_solutions + 1):
        data.insert(index, SEPARATING_LINE)

    print(f"\nExperiment: {experiment_name}\n")

    colalign = ("left", "left", "right", "right", "right")
    table = tabulate(
        data, headers, tablefmt="simple", colalign=colalign, disable_numparse=True
    )

    print(table)
