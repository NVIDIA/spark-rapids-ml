#!/bin/bash

RAPIDS_VERSION=23.6.0

# patch existing packages
mamba install "llvmlite<0.40,>=0.39.0dev0" "numba>=0.56.2"

# dataproc 2.1 pyarrow and arrow conda installation is not compatible with cudf
mamba uninstall -y pyarrow arrow

# install cudf and cuml
pip install --upgrade pip
pip install cudf-cu11~=${RAPIDS_VERSION} cuml-cu11~=${RAPIDS_VERSION} \
    pylibraft-cu11~=${RAPIDS_VERSION} \
    rmm-cu11~=${RAPIDS_VERSION} \
    --extra-index-url=https://pypi.nvidia.com

# install spark-rapids-ml
pip install spark-rapids-ml
