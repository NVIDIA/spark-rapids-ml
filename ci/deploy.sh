#!/bin/bash
# exit on first fail
set -e

# build pip package
pushd python
pip install -r requirements_dev.txt && pip install -e .
python -m build
popd
