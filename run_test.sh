#! /bin/bash

##### 1. format checking

python ci/lint_python.py --format=1 --type-check=0 --pylint=0 || exit 1

##### 2. mypy

python ci/lint_python.py --format=0 --type-check=1 --pylint=0 || exit 1

##### 3. run test

pytest sparkcuml
