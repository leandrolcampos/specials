> [!WARNING]
> Specials is under development and the API is subject to change.

# Welcome to Specials

Specials is a [Mojo](https://www.modular.com/mojo) package designed to provide highly-optimized and hardware acceleration-friendly [special functions](https://en.wikipedia.org/wiki/Special_functions) implementations for AI computing.

Special functions are particular mathematical functions that play a fundamental role in various scientific and industrial disciplines, about which many useful properties are known. They find extensive applications in physics, engineering, chemistry, computer science, and statistics, being prized for their ability to provide closed-form solutions to complex problems in these fields.

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

We can give some examples of special function applications in AI:

- The Gaussian Error Linear Unit (GELU) [[2](#hendrycks2016)], a high-performing neural network activation function, is defined based on the [Gauss error](https://en.wikipedia.org/wiki/Error_function) function.

- Using numerical methods for [Bessel](https://en.wikipedia.org/wiki/Bessel_function), [incomplete beta](https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function), and [incomplete gamma](https://en.wikipedia.org/wiki/Incomplete_gamma_function) functions, we can implicitly differentiate [[1](#figurnov2018)] cumulative distribution functions that are expressed in terms of these special functions, and then train probabilistic models with, for instance, von Mises, gamma, and beta latent variables.

Recognizing the relevance of special functions in the field, major AI frameworks provide implementations for many of them. Both PyTorch and Jax include dedicated modules, namely [`torch.special`](https://pytorch.org/docs/stable/special.html) and [`jax.scipy.special`](https://jax.readthedocs.io/en/latest/jax.scipy.html#module-jax.scipy.special). On the other hand, TensorFlow and TensorFlow Probability incorporate them into their mathematical functions APIs, accessible through [`tf.math`](https://www.tensorflow.org/api_docs/python/tf/math) and [`tfp.math`](https://www.tensorflow.org/probability/api_docs/python/tfp/math). It is important to note that, for many of these functions, these frameworks still do not support partial derivatives with respect to all their arguments.

## Why Mojo 🔥 for Specials?

By adopting the Mojo programming language, Specials enhances the implementation of special functions with several key advantages:

- **Simplifying Complexity Without Compromising Performance.** Unlike traditional approaches that involve wrapping low-level language code, Mojo simplifies the complexity associated with adding and maintaining special function implementations. It seamlessly combines Python's simplicity with Fortran and C-level performance, all within a single language.

- **Support for Hardware Accelerators.** Specials recognizes the growing need for harnessing the power of GPUs, TPUs, and other exotic hardware types in AI computation. Mojo is explicitly designed to leverage the multitude of low-level AI hardware, without the need for C++ or CUDA.

- **Enabling Highly Accurate and Optimized Implementations.** Mojo opens the door to state-of-the-art numerical methods for a wide range of special functions. Implementing these methods in frameworks like TensorFlow and PyTorch can be a challenging task. See my own experience [here](https://github.com/tensorflow/probability/pulls?q=is%3Apr+is%3Aclosed+author%3Aleandrolcampos+%28betainc+OR+cdf+in%3Atitle%29+created%3A%3E2022-05-01) contributing some special functions implementations to TensorFlow Probability: I found writing them in terms of primitive operators available in Python less complex than dealing with multiple backends in C++. Mojo simplifies this process, providing a unified solution that enables Specials to fully leverage vectors, threads, and AI hardware units.

With Mojo and Specials, AI developers and researchers can use special functions to build powerful machine learning models, achieving not only numerical accuracy and stability but also performance.

## Why the Focus on Special Functions?

Beyond the practical importance of special functions in scientific and industrial applications, finding accurate and efficient ways to work with them can be an enjoyable brain-teaser for those who love math and computer science.

## Mojo Version Requirement

Specials requires Mojo `24.2.*`. Make sure you have the correct Mojo version installed before using this package.

## Getting Started

To get started, access a Mojo programming environment directly via the setup instructions on the Mojo [installation page](https://docs.modular.com/mojo/manual/get-started/).

Git clone the Specials repository to your machine using the following command:

```bash
git clone https://github.com/leandrolcampos/specials.git
```

Considering that Mojo SDK as well as our benchmarks and tests depend on an existing installed version of Python, follow the instructions below to create, activate, and configure a Python virtual environment with Conda:

1. Install Conda by following the 
   [Quick command-line install instructions](https://docs.conda.io/projects/miniconda/en/latest/#quick-command-line-install).

   Make sure to initialize Conda for the shell or shells you use, for example:

   ```bash
   ~/miniconda3/bin/conda init zsh
   ```

   Or:

   ```bash
   ~/miniconda3/bin/conda init bash
   ```

2. Restart your shell.

3. Go to the cloned Specials repository and run the following commands to create and activate a Conda environment named `specials`:

   ```bash
   conda env create -f python_environment.yml
   conda activate specials
   ```

4. Run these five commands to configure Mojo to use the Python shared library from `specials` environment when it is active:

   ```bash
   mkdir -p $CONDA_PREFIX/etc/conda/activate.d
   export MOJO_PYTHON_LIBRARY="$(find $CONDA_PREFIX/lib -iname 'libpython*.[s,d]*' | sort -r | head -n 1)"
   echo "export MOJO_PYTHON_LIBRARY=\"$MOJO_PYTHON_LIBRARY\"" > \
       $CONDA_PREFIX/etc/conda/activate.d/export-mojo.sh
   mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
   echo "unset MOJO_PYTHON_LIBRARY" > $CONDA_PREFIX/etc/conda/deactivate.d/unset-mojo.sh   
   ```

**Optional:** If you are planning to play with Specials code using Visual Studio Code, consider adding the following lines to the project's [workspace settings](https://code.visualstudio.com/docs/getstarted/settings#_workspace-settings):

```json
"mojo.lsp.includeDirs": [
    "/path/to/repo/src",
    "/path/to/repo/test"
]
```

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

## Some Implementations Available

### Elementary Functions

> [!NOTE]
> Although the Mojo standard library implements all or most of the elementary functions found in Specials, we have decided to implement them in the package as a matter of priority. For us, **Accuracy `>` Performance**: when forced to choose between FLOPS and numerical accuracy, we prefer numerical accuracy.

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

[<a id="figurnov2018">1</a>]
Figurnov, Mikhail, Shakir Mohamed, and Andriy Mnih. "Implicit reparameterization gradients." _Advances in neural information processing systems_ 31 (2018). [[Link](https://arxiv.org/abs/1805.08498)]

[<a id="hendrycks2016">2</a>]
Hendrycks, Dan, and Kevin Gimpel. "Gaussian error linear units (gelus)." _arXiv preprint arXiv:1606.08415_ (2016). [[Link](https://arxiv.org/abs/1606.08415)]
