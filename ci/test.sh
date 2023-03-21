#!/bin/bash
#
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -ex

type="$1"
case $type in
  "pre-merge" | "")
    ut_args=""
    ;;
  "nightly")
    ut_args="--runslow"
    ;;
  *)
    echo "Unknown test type: $type"; exit 1;;
esac
bench_args=""

# environment
nvidia-smi
which python

# spark-rapids-ml and dependencies
cd python
pip install -r requirements_dev.txt && pip install -e .

# unit tests
./run_test.sh $ut_args

# benchmark
./run_benchmark.sh $bench_args
