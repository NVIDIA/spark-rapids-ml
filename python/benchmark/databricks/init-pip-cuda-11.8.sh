#!/bin/bash
# set portion of path below after /dbfs/ to dbfs zip file location
SPARK_RAPIDS_ML_ZIP=/dbfs/path/to/spark-rapids-ml.zip
BENCHMARK_ZIP=/dbfs/path/to/benchmark.zip
# IMPORTANT: specify rapids fully 23.10.0 and not 23.10
# also RAPIDS_VERSION (python) fields should omit any leading 0 in month/minor field (i.e. 23.8.0 and not 23.08.0)
# while SPARK_RAPIDS_VERSION (jar) should have leading 0 in month/minor (e.g. 23.08.2 and not 23.8.1)
RAPIDS_VERSION=23.10.0
SPARK_RAPIDS_VERSION=23.08.2

curl -L https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/${SPARK_RAPIDS_VERSION}/rapids-4-spark_2.12-${SPARK_RAPIDS_VERSION}-cuda11.jar -o /databricks/jars/rapids-4-spark_2.12-${SPARK_RAPIDS_VERSION}.jar

# install cudatoolkit 11.8 via runfile approach
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit

# reset symlink and update library loading paths
# **** set LD_LIBRARY_PATH as below in env var section of cluster config in DB cluster UI ****
rm /usr/local/cuda
ln -s /usr/local/cuda-11.8 /usr/local/cuda

# upgrade pip
/databricks/python/bin/pip install --upgrade pip

# install cudf and cuml
# using ~= pulls in micro version patches
/databricks/python/bin/pip install cudf-cu11~=${RAPIDS_VERSION} \
    cuml-cu11~=${RAPIDS_VERSION} \
    pylibraft-cu11~=${RAPIDS_VERSION} \
    rmm-cu11~=${RAPIDS_VERSION} \
    --extra-index-url=https://pypi.nvidia.com

# install spark-rapids-ml
python_ver=`python --version | grep -oP '3\.[0-9]+'`
unzip ${SPARK_RAPIDS_ML_ZIP} -d /databricks/python3/lib/python${python_ver}/site-packages
unzip ${BENCHMARK_ZIP} -d /databricks/python3/lib/python${python_ver}/site-packages

