# Benchmarks

This directory contains a collection of Mojo notebooks designed to compare the accuracy and runtime performance of functions from the Specials package against those found in well-known Mojo and Python packages.

The benchmarks aim to highlight the correctness and efficiency of our implementations.

Please note that for us, **Accuracy > Performance**: when forced to choose between FLOPS and numerical accuracy, we always prefer numerical accuracy.

## Getting Started

Before running the benchmarks, ensure you have completed the initial setup for the project. This includes installing Mojo, cloning the Specials repository, and setting up and activating a Python environment with Conda. Detailed setup instructions are available [here](../README.md#getting-started).

Once the initial setup is complete, build the Specials package to ensure all components are ready for benchmarking. Run the following command in the `specials/benchmarks` directory of the cloned repository:

```bash
../scripts/prepare-benchmarks.sh
```

## Running

The benchmarks can be run in [Visual Studio Code](https://github.com/modularml/mojo/tree/main/examples/notebooks#get-started-in-vs-code) or [JupyterLab](https://github.com/modularml/mojo/tree/main/examples/notebooks#get-started-with-jupyterlab) with Mojo kernel support. Follow the respective guides to set up your environment.

Once set up, open the desired notebook in the `specials/benchmarks` directory to begin running the benchmarks.
