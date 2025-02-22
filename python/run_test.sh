#! /bin/bash -e
# Copyright (c) 2025, NVIDIA CORPORATION.
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

unset SPARK_HOME

python ../ci/lint_python.py --format --type-check || exit 1

total_num_gpus=$(python -c 'import cupy; print(cupy.cuda.runtime.getDeviceCount())')
if [ ${total_num_gpus} -gt 4 ]
then
    echo "Tests use at most 4 GPUs. If failed, try setting CUDA_VISIBLE_DEVICES."
fi

# no package import change tests
# runs on gpu
python -m spark_rapids_ml tests_no_import_change/test_no_import_change.py 0.2
# runs on cpu
python tests_no_import_change/test_no_import_change.py 0.2
# runs on gpu with spark-submit
spark-rapids-submit --master local[1] tests_no_import_change/test_no_import_change.py 0.2
# runs on cpu with spark-submit
spark-submit --master local[1] tests_no_import_change/test_no_import_change.py 0.2


# calculate pytest parallelism by following https://github.com/NVIDIA/spark-rapids/tree/main/integration_tests/run_pyspark_from_build.sh
MAX_PARALLEL=3
NVIDIA_SMI_ARGS="" 
if [ ${CUDA_VISIBLE_DEVICES} ]; then
        NVIDIA_SMI_ARGS="${NVIDIA_SMI_ARGS} -i ${CUDA_VISIBLE_DEVICES}" 
fi
GPU_MEM_PARALLEL=`nvidia-smi ${NVIDIA_SMI_ARGS} --query-gpu=memory.free --format=csv,noheader | awk 'NR == 1 { MIN = $1 } { if ($1 < MIN) { MIN = $1 } } END { print int((MIN - 2 * 1024) / ((3 * 1024) + 750)) }'`
CPU_CORES=`nproc`
TMP_PARALLEL=$(( $GPU_MEM_PARALLEL > $CPU_CORES ? $CPU_CORES : $GPU_MEM_PARALLEL ))
TMP_PARALLEL=$(( $TMP_PARALLEL > $MAX_PARALLEL ? $MAX_PARALLEL : $TMP_PARALLEL ))
if  (( $TMP_PARALLEL <= 1 )); then
        TEST_PARALLEL=1
    else
        TEST_PARALLEL=$TMP_PARALLEL
fi
echo "Test functions in benchmark/test_gen_data.py and tests/ directory will be executed in parallel with ${TEST_PARALLEL} pytest workers" 

echo "use --runslow to run all tests"
pytest "$@" -n ${TEST_PARALLEL} benchmark/test_gen_data.py
PYTHONPATH=`pwd`/benchmark pytest -ra "$@" -n ${TEST_PARALLEL} --durations=10 tests
#PYTHONPATH=`pwd`/benchmark pytest -ra --runslow -n ${TEST_PARALLEL} --durations=10 tests
#PYTHONPATH=`pwd`/benchmark pytest -ra "$@" --durations=10 tests_large
