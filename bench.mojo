from random import seed
from tensor import Tensor, TensorShape
from sys.info import simdwidthof

from benchmark import benchmark
from benchmark import Bench, BenchConfig, Bencher, BenchId, Unit, keep, run

import specials

from test_utils.tensor import run_benchmark

# ===----------------------------------------------------------------------===#
# Benchmark Data
# ===----------------------------------------------------------------------===#
alias input_type = DType.float32


fn make_input(
    begin: Scalar[input_type], end: Scalar[input_type], num: Int
) -> Tensor[input_type]:
    var shape = TensorShape(num)

    if num == 1:
        return Tensor[input_type](shape, begin)

    var step = (end - begin) / (num - 1)

    var result = List[Scalar[input_type]]()

    for i in range(num):
        result.append(begin + step * i)

    return Tensor[input_type](shape, result)


var input = make_input(0, 10_000, 1_000_000)


# ===----------------------------------------------------------------------===#
# Benchmark Main
# ===----------------------------------------------------------------------===#
def main():
    seed()
    var msecs = run_benchmark[
        specials.log,
        type=input_type,
        simd_width = simdwidthof[input_type](),
        force_sequential=True,
    ](input, num_warmup=100).mean("ms")
    print(msecs)
