#!/bin/bash -xe
# Copyright (c) 2024, NVIDIA CORPORATION.
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


# copies files to s3 bucket

if [[ -z $BENCHMARK_HOME ]]; then
    echo "please export BENCHMARK_HOME per README.md"
    exit 1
fi

SPARK_RAPIDS_ML_HOME='../..'

echo "**** copying benchmarking related files to ${BENCHMARK_HOME} ****"

aws s3 cp ../../../notebooks/aws-emr/init-bootstrap-action.sh s3://${BENCHMARK_HOME}/init-bootstrap-action.sh

pushd ${SPARK_RAPIDS_ML_HOME}/benchmark
zip -r - benchmark > benchmark.zip
aws s3 cp benchmark.zip s3://${BENCHMARK_HOME}/benchmark.zip
popd

pushd ${SPARK_RAPIDS_ML_HOME}
aws s3 cp benchmark/benchmark_runner.py s3://${BENCHMARK_HOME}/benchmark_runner.py
popd

pushd ${SPARK_RAPIDS_ML_HOME}/src
zip -r - spark_rapids_ml >spark_rapids_ml.zip
aws s3 cp spark_rapids_ml.zip s3://${BENCHMARK_HOME}/spark_rapids_ml.zip
popd
