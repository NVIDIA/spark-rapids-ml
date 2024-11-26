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


# copies files to dbfs 

if [[ -z $DB_PROFILE ]]; then
    echo "please export DB_PROFILE per README.md"
    exit 1
fi

if [[ -z $BENCHMARK_HOME ]]; then
    echo "please export BENCHMARK_HOME per README.md"
    exit 1
fi

if [[ -z $WS_BENCHMARK_HOME ]]; then
    echo "please expoert WS_BENCHMARK_HOME per README.md"
    exit 1
fi

SPARK_RAPIDS_ML_HOME='../..'

echo "**** copying benchmarking related files to ${BENCHMARK_HOME} ****"

INIT_SCRIPT_DIR="${WS_BENCHMARK_HOME}/init_scripts"
SPARK_RAPIDS_ML_ZIP="${BENCHMARK_HOME}/zips/spark-rapids-ml.zip"
BENCHMARK_ZIP="${BENCHMARK_HOME}/zips/benchmark.zip"
BENCHMARK_SCRIPTS="${BENCHMARK_HOME}/scripts"
databricks fs mkdirs dbfs:${BENCHMARK_HOME}/zips --profile $DB_PROFILE
databricks fs mkdirs dbfs:${BENCHMARK_HOME}/scripts --profile $DB_PROFILE

pushd ${SPARK_RAPIDS_ML_HOME}/benchmark && rm -f benchmark.zip && \
zip -r benchmark.zip benchmark && \
databricks fs cp benchmark.zip dbfs:${BENCHMARK_ZIP} --profile ${DB_PROFILE} ${DB_OVERWRITE}
popd

pushd ${SPARK_RAPIDS_ML_HOME} && \
ls benchmark
databricks fs cp benchmark/benchmark_runner.py dbfs:${BENCHMARK_SCRIPTS}/benchmark_runner.py --profile  ${DB_PROFILE} ${DB_OVERWRITE}
popd

pushd ${SPARK_RAPIDS_ML_HOME}/src && rm -f spark-rapids-ml.zip && \
zip -r spark-rapids-ml.zip spark_rapids_ml && \
databricks fs cp spark-rapids-ml.zip dbfs:${SPARK_RAPIDS_ML_ZIP} --profile ${DB_PROFILE} ${DB_OVERWRITE}
popd

# create workspace directory
databricks workspace mkdirs ${INIT_SCRIPT_DIR} --profile ${DB_PROFILE} ${DB_OVERWRITE}
# point cpu and gpu cluster init scripts to new files and upload
for init_script in init-pip-cuda-11.8.sh init-cpu.sh
do
# NOTE: on linux delete the .bu after -i
    if base64 --help | grep '\-w'; then
        # linux
        base64_cmd="base64 -w 0"
    else
        # mac os
        base64_cmd="base64 -i"
    fi
    sed -e "s#/path/to/spark-rapids-ml\.zip#${SPARK_RAPIDS_ML_ZIP}#g" -e "s#/path/to/benchmark\.zip#${BENCHMARK_ZIP}#g" $init_script > ${init_script}.updated && \
    databricks workspace import --format AUTO --content $(${base64_cmd} ${init_script}.updated) ${INIT_SCRIPT_DIR}/${init_script} --profile ${DB_PROFILE} ${DB_OVERWRITE}
done
