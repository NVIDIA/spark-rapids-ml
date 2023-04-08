#! /bin/bash
unset SPARK_HOME

python ../ci/lint_python.py --format --type-check || exit 1

total_num_gpus=```nvidia-smi --query-gpu=index --format=csv,noheader | wc -l```
if [ ${total_num_gpus} -gt 4 ]
then
        echo "Tests use at most 4 GPUs. If failed, try setting CUDA_VISIBLE_DEVICES."
            echo "Ignore the message if CUDA_VISIBLE_DEVICES has been set properly."
fi
echo "use --runslow to run all tests"
pytest -ra "$@" --durations=10 tests
# pytest -ra --runslow --durations=10 tests
