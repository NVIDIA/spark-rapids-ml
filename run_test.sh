#! /bin/bash

unset SPARK_HOME
python ci/lint_python.py --format --type-check || exit 1
pytest -ra --runslow --durations=10 tests
