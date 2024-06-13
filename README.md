> [!WARNING]
> Specials is under development and the API is subject to change.

# Welcome to Specials

Specials is a [Mojo](https://www.modular.com/mojo) package designed to provide highly-optimized and hardware-acceleration-friendly [special functions](https://en.wikipedia.org/wiki/Special_functions) implementations for AI computing.

Mojo combines the usability of Python with the performance of C, unlocking unparalleled programmability of AI hardware and extensibility of AI models. This makes Mojo an ideal choice for implementing special functions that require not only numerical accuracy and stability but also high performance.

## Table of Contents

- [Special Functions in AI](#special-functions-in-ai)
- [Why Mojo 🔥 for Specials?](#why-mojo--for-specials)
- [Why the Focus on Special Functions?](#why-the-focus-on-special-functions)
- [Mojo Version Requirement](#mojo-version-requirement)
- [Getting Started](#getting-started)
- [Example Usage](#example-usage)
- [Benchmarks](#benchmarks)
- [Some Implementations Available](#some-implementations-available)
  * [Elementary Functions](#elementary-functions)
- [Contributing](#contributing)
- [References](#references)

## Special Functions in AI

Special functions are integral to various AI applications. For instance:

- The Gaussian Error Linear Unit (GELU) [[1](#hendrycks2016)], a high-performing neural network activation function, is based on the [Gauss error](https://en.wikipedia.org/wiki/Error_function) function.

- Numerical methods for [Bessel](https://en.wikipedia.org/wiki/Bessel_function), [incomplete beta](https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function), and [incomplete gamma](https://en.wikipedia.org/wiki/Incomplete_gamma_function) functions enable implicit differentiation [[2](#figurnov2018)] of cumulative distribution functions, facilitating the training of probabilistic models with von Mises, gamma, and beta latent variables.

Recognizing the relevance of special functions in the field, major AI frameworks provide implementations for many of them. Both PyTorch and JAX include dedicated modules, namely [`torch.special`](https://pytorch.org/docs/stable/special.html) and [`jax.scipy.special`](https://jax.readthedocs.io/en/latest/jax.scipy.html#module-jax.scipy.special). TensorFlow and TensorFlow Probability incorporate them into their mathematical functions APIs, accessible through [`tf.math`](https://www.tensorflow.org/api_docs/python/tf/math) and [`tfp.math`](https://www.tensorflow.org/probability/api_docs/python/tfp/math).

## Why Mojo 🔥 for Specials?

By adopting the Mojo programming language, Specials enhances the implementation of special functions with several key advantages:

- **Simplified Complexity with High Performance.** Mojo combines Python's simplicity with C-level performance, eliminating the need for wrapping low-level language code.

- **Support for Hardware Accelerators.** Mojo is designed to leverage the power of GPUs, TPUs, and other AI hardware without the need for C++ or CUDA.

- **Highly Accurate and Optimized Implementations.** Mojo enables the implementation of state-of-the-art numerical methods, ensuring numerical accuracy and stability as well as high performance.

## Why the Focus on Special Functions?

Special functions are particular mathematical functions that play a fundamental role in various scientific and industrial disciplines, about which many useful properties are known. They find extensive applications in physics, engineering, chemistry, computer science, and statistics, being prized for their ability to provide closed-form solutions to complex problems in these fields.

## Mojo Version Requirement

Specials requires Mojo `24.4.0`. Make sure you have the correct Mojo version installed before using this package.

## Getting Started

To get started, access a Mojo programming environment directly via the setup instructions on the Mojo [installation page](https://docs.modular.com/mojo/manual/get-started/).

Clone the Specials repository to your machine:

```bash
git clone https://github.com/leandrolcampos/specials.git
```

Considering that Mojo SDK as well as our benchmarks and tests depend on an existing installed version of Python, follow the instructions below to create and activate a Python virtual environment with Conda:

1. Install Conda by following the 
   [Quick command-line install instructions](https://docs.conda.io/projects/miniconda/en/latest/#quick-command-line-install). Ensure Conda is initialized for your shell:

   ```bash
   ~/miniconda3/bin/conda init zsh
   # or
   ~/miniconda3/bin/conda init bash
   ```

2. Restart your shell.

3. Navigate to the cloned Specials repository and run the following commands to create and activate a Conda environment named `specials`:

   ```bash
   conda env create -f python_environment.yml
   conda activate specials
   ```

**Optional:** If using Visual Studio Code, consider adding the following items to `mojo.lsp.includeDirs` setting in the user or remote scope:

- `/path/to/repo/src`
- `/path/to/repo/test`

Replace `/path/to/repo` with the absolute path of the cloned Specials repository.

## Example Usage

The following code snippet shows how to compute `exp(x) - 1` in a numerically stable way for a given SIMD vector:

```python
>>> import specials
>>> var x = SIMD[DType.float64, 4](0.0, 1e-18, 0.2, 1.0)
>>> var result = specials.expm1(x)
>>> print(result)
[0.0, 1.0000000000000001e-18, 0.22140275816016985, 1.7182818284590453]
```

## Benchmarks

The [`benchmarks`](./benchmarks/) directory contains a collection of Mojo notebooks designed to compare the accuracy and runtime performance of functions from the Specials package against those found in well-known Mojo and Python packages.

These benchmarks aim to highlight the correctness and efficiency of Specials implementations.

Please note that for us, **Accuracy > Performance**: when forced to choose between FLOPS and numerical accuracy, we always prefer numerical accuracy.

## Some Implementations Available

### Elementary Functions

| Function | Description |
|----------|-------------|
| `exp(x)` | The exponential function |
| `exp2(x)` | The base-2 exponential function |
| `expm1(x)` | The expression `exp(x) - 1` evaluated in a numerically stable way when `x` is near zero |
| `log(x)` | The logarithm function |
| `log1p(x)` | The expression `log(1 + x)` evaluated in a numerically stable way when `x` is near zero |

## Contributing

We are not accepting pull requests at this time. However, you can contribute by reporting issues or suggesting features through the creation of a GitHub issue [here](https://github.com/leandrolcampos/specials/issues).

## References

[<a id="hendrycks2016">1</a>]
Hendrycks, Dan, and Kevin Gimpel. "Gaussian error linear units (gelus)." _arXiv preprint arXiv:1606.08415_ (2016). [[Link](https://arxiv.org/abs/1606.08415)]

[<a id="figurnov2018">2</a>]
Figurnov, Mikhail, Shakir Mohamed, and Andriy Mnih. "Implicit reparameterization gradients." _Advances in neural information processing systems_ 31 (2018). [[Link](https://arxiv.org/abs/1805.08498)]
