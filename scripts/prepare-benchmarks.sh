#!/usr/bin/env bash
##===----------------------------------------------------------------------===##
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
##===----------------------------------------------------------------------===##

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$(realpath "${SCRIPT_DIR}/..")
BUILD_DIR="${REPO_ROOT}/benchmarks"
mkdir -p "${BUILD_DIR}"

echo "Packaging up the Specials"
SPECIALS_PATH="${REPO_ROOT}/src/specials"
FULL_SPECIALS_PACKAGE_PATH="${BUILD_DIR}/specials.mojopkg"
mojo package "${SPECIALS_PATH}" -o "${FULL_SPECIALS_PACKAGE_PATH}"

echo Successfully created "${FULL_SPECIALS_PACKAGE_PATH}"

echo "Installing the Python dependencies of test_utils"
python3 -m pip install -q -e "${REPO_ROOT}"/test

echo "Packaging up the test_utils"
TEST_UTILS_PATH="${REPO_ROOT}/test/test_utils"
FULL_TEST_UTILS_PACKAGE_PATH="${BUILD_DIR}/test_utils.mojopkg"
mojo package "${TEST_UTILS_PATH}" -o "${FULL_TEST_UTILS_PACKAGE_PATH}"

echo Successfully created "${FULL_TEST_UTILS_PACKAGE_PATH}"
