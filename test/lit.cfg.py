# type: ignore
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

"""Configuration for the lit test runner used by the Specials test suite."""

import os
import platform
import shutil

from pathlib import Path

import lit.formats
import lit.llvm


config.test_format = lit.formats.ShTest(True)

# name: The name of this test suite
config.name = "Specials"

# suffixes: A list of file extensions to treat as test files
config.suffixes = [".mojo", ".py"]

# excludes: A list of files and directories ignored by the lit
# test runner when scanning for tests
# `test_utils` does not contain tests, just source code that we
# run `mojo package` on to be used by other tests
config.excludes = ["lit.cfg.py", "setup.py", "test_utils"]

# test_source_root: The root path where tests are located
test_dir = Path(__file__).parent.resolve()
config.test_source_root = test_dir

# The `run-tests.sh` script creates the build directory for us
build_dir = Path(__file__).parent.parent / "build"

# The tests are executed inside this build directory to avoid
# polluting the source tree
config.test_exec_root = build_dir / "test_output"

# This makes the OS name available for `REQUIRE` directives,
# e.g., `# REQUIRE: darwin`
config.available_features.add(platform.system().lower())


# Check if the `not` binary from LLVM is available
def has_not():
    return shutil.which("not") is not None


if has_not():
    config.available_features.add("has_not")

build_mode = os.getenv("BUILD_MODE", "debug")

if build_mode == "debug":
    assertion_flag = "-D MOJO_ENABLE_ASSERTIONS"
    debug_level = "--debug-level full"
else:
    assertion_flag = ""
    debug_level = "--debug-level none"

config.substitutions.insert(0, ("%mojo", "mojo run"))
config.substitutions.insert(1, ("%build_dir", f"-I {str(build_dir / build_mode)}"))
config.substitutions.insert(2, ("%assertion_flag", assertion_flag))
config.substitutions.insert(3, ("%debug_level", "--debug-level full"))
config.substitutions.insert(4, ("%sanitize_checks", ""))
config.substitutions.insert(5, ("%pytest", "pytest"))

# Pass through several environment variables to the underlying
# subprocesses that run the tests
lit.llvm.initialize(lit_config, config)
lit.llvm.llvm_config.with_system_environment(
    [
        "MODULAR_HOME",
        "PATH",
    ]
)
