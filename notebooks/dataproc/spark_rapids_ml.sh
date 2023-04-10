#!/bin/bash

RAPIDS_VERSION=23.2.0

# upgrade pip
pip install --upgrade pip

# patch existing
pip install --ignore-installed "llvmlite<0.40,>=0.39.0dev0" "numba>=0.56.2"

# install cudf and cuml
pip install cudf-cu11==${RAPIDS_VERSION} \
cuml-cu11==${RAPIDS_VERSION} \
pylibraft-cu11==${RAPIDS_VERSION} \
rmm-cu11==${RAPIDS_VERSION} \
--extra-index-url=https://pypi.nvidia.com

# rapids pip package patch: link ucx libraries in ucx-py to default location searched by raft_dask
ln -s /opt/conda/miniconda3/lib/python3.8/site-packages/ucx_py_cu11.libs/ucx /usr/lib/ucx

# install spark-rapids-ml
pip install spark-rapids-ml
