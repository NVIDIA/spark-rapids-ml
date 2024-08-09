#! /bin/bash
unset SPARK_HOME

python ../ci/lint_python.py --format --type-check || exit 1

total_num_gpus=$(python -c 'import cupy; print(cupy.cuda.runtime.getDeviceCount())')
if [ ${total_num_gpus} -gt 4 ]
then
    echo "Tests use at most 4 GPUs. If failed, try setting CUDA_VISIBLE_DEVICES."
fi
echo "use --runslow to run all tests"
#pytest "$@" benchmark/test_gen_data.py
PYTHONPATH=`pwd`/benchmark pytest -s --runslow tests/test_logistic_regression.py::test_sparse_int64
#PYTHONPATH=`pwd`/benchmark pytest -ra "$@" --durations=10 tests
#PYTHONPATH=`pwd`/benchmark pytest -ra --runslow --durations=10 tests
