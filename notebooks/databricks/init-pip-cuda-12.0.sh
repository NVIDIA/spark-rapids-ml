#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -ex

# IMPORTANT: specify RAPIDS_VERSION fully 23.10.0 and not 23.10
# also in general, RAPIDS_VERSION (python) fields should omit any leading 0 in month/minor field (i.e. 23.8.0 and not 23.08.0)
# while SPARK_RAPIDS_VERSION (jar) should have leading 0 in month/minor (e.g. 23.08.2 and not 23.8.2)
RAPIDS_VERSION=25.12.0
SPARK_RAPIDS_VERSION=25.08.0

curl -L https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/${SPARK_RAPIDS_VERSION}/rapids-4-spark_2.12-${SPARK_RAPIDS_VERSION}-cuda12.jar -o /databricks/jars/rapids-4-spark_2.12-${SPARK_RAPIDS_VERSION}.jar

# install cudatoolkit 12.0 via runfile approach
wget https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda_12.0.1_525.85.12_linux.run
sh cuda_12.0.1_525.85.12_linux.run --silent --toolkit

# reset symlink and update library loading paths
rm /usr/local/cuda
ln -s /usr/local/cuda-12.0 /usr/local/cuda

# upgrade pip
/databricks/python/bin/pip install --upgrade pip

# install cudf, cuml and their rapids dependencies
# using ~= pulls in latest micro version patches
/databricks/python/bin/pip install --no-cache-dir \
    cudf-cu12~=${RAPIDS_VERSION} \
    cuml-cu12~=${RAPIDS_VERSION} \
    cuvs-cu12~=${RAPIDS_VERSION} \
    pylibraft-cu12~=${RAPIDS_VERSION} \
    raft-dask-cu12~=${RAPIDS_VERSION} \
    dask-cuda-cu12~=${RAPIDS_VERSION} \
    numpy~=1.0 \
    --extra-index-url=https://pypi.nvidia.com

# install spark-rapids-ml
/databricks/python/bin/pip install spark-rapids-ml

# set up no-import-change for cluster if enabled
if [[ $SPARK_RAPIDS_ML_NO_IMPORT_ENABLED == 1 ]]; then
    echo "enabling no import change in cluster" 1>&2
    mkdir -p /root/.ipython/profile_default/startup
    echo "import spark_rapids_ml.install" >/root/.ipython/profile_default/startup/00-spark-rapids-ml.py
fi



