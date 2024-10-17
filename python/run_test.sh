#! /bin/bash -e
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

# calculate pytest parallelism following https://github.com/NVIDIA/spark-rapids/blob/branch-24.12/integration_tests/run_pyspark_from_build.sh
GPU_MEM_PARALLEL=`nvidia-smi --query-gpu=memory.free --format=csv,noheader | awk '{if (MAX < $1){ MAX = $1}} END {print int((MAX - 2 * 1024) / ((1.5 * 1024) + 750))}'`
CPU_CORES=`nproc`
HOST_MEM_PARALLEL=`cat /proc/meminfo | grep MemAvailable | awk '{print int($2 / (8 * 1024 * 1024))}'`
TMP_PARALLEL=$(( $GPU_MEM_PARALLEL > $CPU_CORES ? $CPU_CORES : $GPU_MEM_PARALLEL ))
TMP_PARALLEL=$(( $TMP_PARALLEL > $HOST_MEM_PARALLEL ? $HOST_MEM_PARALLEL : $TMP_PARALLEL ))
TEST_PARALLEL=${TMP_PARALLEL}
echo "${TEST_PARALLEL} pytest workers will be used to execute test functions in benchmark/test_gen_data.py and tests/ directory in parallel"

echo "use --runslow to run all tests"
pytest "$@" -n ${TEST_PARALLEL} benchmark/test_gen_data.py
PYTHONPATH=`pwd`/benchmark pytest -ra "$@" -n ${TEST_PARALLEL} --durations=10 tests
#PYTHONPATH=`pwd`/benchmark pytest -ra --runslow -n ${TEST_PARALLEL} --durations=10 tests
#PYTHONPATH=`pwd`/benchmark pytest -ra "$@" --durations=10 tests_large
