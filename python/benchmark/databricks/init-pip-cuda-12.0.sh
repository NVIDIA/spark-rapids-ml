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

# set portion of path below after /dbfs/ to dbfs zip file location
SPARK_RAPIDS_ML_ZIP=/dbfs/path/to/spark-rapids-ml.zip
BENCHMARK_ZIP=/dbfs/path/to/benchmark.zip
# IMPORTANT: specify rapids fully 23.10.0 and not 23.10
# also, in general, RAPIDS_VERSION (python) fields should omit any leading 0 in month/minor field (i.e. 23.8.0 and not 23.08.0)
# while SPARK_RAPIDS_VERSION (jar) should have leading 0 in month/minor (e.g. 23.08.2 and not 23.8.2)
RAPIDS_VERSION=25.10.0
SPARK_RAPIDS_VERSION=25.08.0

curl -L https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/${SPARK_RAPIDS_VERSION}/rapids-4-spark_2.12-${SPARK_RAPIDS_VERSION}-cuda12.jar -o /databricks/jars/rapids-4-spark_2.12-${SPARK_RAPIDS_VERSION}.jar

# install cudatoolkit 12.0 via runfile approach
wget https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda_12.0.1_525.85.12_linux.run
sh cuda_12.0.1_525.85.12_linux.run --silent --toolkit

# reset symlink and update library loading paths
# **** set LD_LIBRARY_PATH as below in env var section of cluster config in DB cluster UI ****
rm /usr/local/cuda
ln -s /usr/local/cuda-12.0 /usr/local/cuda

# upgrade pip
/databricks/python/bin/pip install --upgrade pip

# install cudf and cuml
# using ~= pulls in micro version patches
/databricks/python/bin/pip install cudf-cu12~=${RAPIDS_VERSION} \
    cuml-cu12~=${RAPIDS_VERSION} \
    cuvs-cu12~=${RAPIDS_VERSION} \
    --extra-index-url=https://pypi.nvidia.com

# install spark-rapids-ml
python_ver=`python --version | grep -oP '3\.[0-9]+'`
unzip ${SPARK_RAPIDS_ML_ZIP} -d /databricks/python3/lib/python${python_ver}/site-packages
unzip ${BENCHMARK_ZIP} -d /databricks/python3/lib/python${python_ver}/site-packages

