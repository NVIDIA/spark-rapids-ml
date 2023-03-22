#! /bin/bash
unset SPARK_HOME

python ../ci/lint_python.py --format --type-check || exit 1

echo "use --runslow to run all tests"
pytest -ra "$@" --durations=10 tests
# pytest -ra --runslow --durations=10 tests
