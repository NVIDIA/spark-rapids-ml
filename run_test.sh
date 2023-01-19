#! /bin/bash

unset SPARK_HOME
python ci/lint_python.py --format --type-check || exit 1
pytest -s sparkcuml
