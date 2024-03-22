# Benchmarks

This directory contains a collection of Mojo notebooks designed to compare the accuracy and runtime performance of functions from the Specials package against those found in well-known Mojo and Python packages.

The benchmarks aim to highlight the correctness and efficiency of our implementations.

## Getting Started

To get started, access a Mojo programming environment directly via the setup instructions on the Mojo [installation page](https://docs.modular.com/mojo/manual/get-started/).

Considering that Mojo SDK and our benchmarks depend on an existing installed version of Python, follow the instructions in [Set up a Python environment with Conda](https://docs.modular.com/mojo/manual/python/#set-up-a-python-environment-with-conda) to create and activate a Python virtual environment.

Git clone the Specials repository to your machine using the following command:

```bash
git clone https://github.com/leandrolcampos/specials.git
```

Build the Specials package by running the following command in the `specials/benchmarks` directory of the cloned repository:

```bash
mojo package ../specials -o specials.mojopkg
```

## Running

The benchmarks can be run in [Visual Studio Code](https://github.com/modularml/mojo/tree/main/examples/notebooks#get-started-in-vs-code) or [JupyterLab](https://github.com/modularml/mojo/tree/main/examples/notebooks#get-started-with-jupyterlab) with Mojo kernel support. Follow the respective guides to set up your environment.

Once set up, navigate to the `specials/benchmarks` directory and open the desired notebook to begin running the benchmarks.