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

echo "use --runslow to run all tests"
pytest "$@" benchmark/test_gen_data.py
PYTHONPATH=`pwd`/benchmark pytest -ra "$@" --durations=10 tests
#PYTHONPATH=`pwd`/benchmark pytest -ra --runslow --durations=10 tests
#PYTHONPATH=`pwd`/benchmark pytest -ra "$@" --durations=10 tests_large
