# Scripts

This directory contains scripts to build, prepare for benchmarks, and test the Specials package.

## Overview

- `build-specials.sh`: This script packages the Specials for release into the `build` directory.
- `prepare-benchmarks.sh`: This script prepares the benchmark environment by installing the needed Python dependencies and packaging the Specials and the `test_utils` separately into the `benchmarks` directory.
- `run-tests.sh`: This script prepares the test environment according to the specified build mode (`debug` or `release`) by installing the needed Python dependencies, packaging the Specials and the `test_utils` separately into the `build` directory, and running the tests.

The `test_utils` package contains utilities designed exclusively for internal use in benchmarks and testing of the Specials code. It is not intended for external use and should not be relied upon as part of the public API or used in production environments.

## Getting Started

Before using these scripts, ensure you have completed the initial setup for the project. This includes installing Mojo, cloning the Specials repository, and setting up and activating a Python environment with Conda. Detailed setup instructions are available [here](../README.md#getting-started).

Following these steps will prepare the necessary environment to effectively build, benchmark, and test the Specials.

## Usage

### Building the Specials

To package the Specials, navigate to the repository root and run:

```bash
./scripts/build-specials.sh
```

This command compiles the Specials into a `.mojopkg` package suitable for release, and places it in the `build` directory.

### Preparing for Benchmarks

If you are setting up for benchmarks, use:

```bash
./scripts/prepare-benchmarks.sh
```

This command prepares the benchmark environment by installing the needed Python dependencies and packaging the Specials and the `test_utils` separately into the `benchmarks` directory.

### Running Tests

To run the tests, specify the build mode as either `debug` or `release`. By default, the script uses the `debug` mode:

```bash
./scripts/run-tests.sh
```

In the `debug` mode, the script sets the `debug-level` to `full` and defines `MOJO_ENABLE_ASSERTIONS`, providing detailed debugging output and enabling runtime assertions for thorough testing.

Alternatively, you can run the tests in the `release` build mode:

```bash
./scripts/run-tests.sh release
```

This command runs the tests in the `release` mode, optimizing performance by setting the `debug-level` to `none` and not defining `MOJO_ENABLE_ASSERTIONS`. This mode ensures that the build is optimized and stable for production use.
