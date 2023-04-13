#!/bin/bash

set -euxo pipefail

function get_metadata_attribute() {
  local -r attribute_name=$1
  local -r default_value=$2
  /usr/share/google/get_metadata_value "attributes/${attribute_name}" || echo -n "${default_value}"
}

RAPIDS_VERSION=$(get_metadata_attribute rapids-version 23.02)

# upgrade pip
pip install --upgrade pip

# patch existing
pip install --ignore-installed "llvmlite<0.40,>=0.39.0dev0" "numba>=0.56.2"

# install cudf and cuml
pip install cudf-cu11~=${RAPIDS_VERSION} \
cuml-cu11~=${RAPIDS_VERSION} \
pylibraft-cu11~=${RAPIDS_VERSION} \
rmm-cu11~=${RAPIDS_VERSION} \
--extra-index-url=https://pypi.nvidia.com

# rapids pip package patch: link ucx libraries in ucx-py to default location searched by raft_dask
ln -s /opt/conda/miniconda3/lib/python3.8/site-packages/ucx_py_cu11.libs/ucx /usr/lib/ucx

# install benchmark files
BENCHMARK_HOME=$(get_metadata_attribute benchmark-home leey-test/benchmark)

gsutil cp gs://${BENCHMARK_HOME}/benchmark_runner.py .
gsutil cp gs://${BENCHMARK_HOME}/spark_rapids_ml.zip .
gsutil cp gs://${BENCHMARK_HOME}/benchmark.zip .

unzip spark_rapids_ml.zip -d /opt/conda/miniconda3/lib/python3.8/site-packages
unzip benchmark.zip -d /opt/conda/miniconda3/lib/python3.8/site-packages
