#!/bin/bash
# set portion of path below after /dbfs/ to dbfs zip file location
SPARK_RAPIDS_ML_ZIP=/dbfs/path/to/zip/file
RAPIDS_VERSION=23.2.0
SPARK_RAPIDS_VERSION=23.02.0

curl -L https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/${SPARK_RAPIDS_VERSION}/rapids-4-spark_2.12-${SPARK_RAPIDS_VERSION}.jar -o /databricks/jars/rapids-4-spark_2.12-${SPARK_RAPIDS_VERSION}.jar

# install cudatoolkit 11.8 via runfile approach
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit

# install forward compatibility package due to old driver
distro=ubuntu2004
arch=x86_64
apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update
apt-get install -y cuda-compat-11-8


# reset symlink and update library loading paths
# **** set LD_LIBRARY_PATH as below in env var section of cluster config in DB cluster UI ****
rm /usr/local/cuda
ln -s /usr/local/cuda-11.8 /usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64
ldconfig

# upgrade pip
/databricks/python/bin/pip install --upgrade pip

# install cudf
/databricks/python/bin/pip install cudf-cu11==${RAPIDS_VERSION} \
cuml-cu11==${RAPIDS_VERSION} \
pylibraft-cu11==${RAPIDS_VERSION} \
rmm-cu11==${RAPIDS_VERSION} \
--extra-index-url=https://pypi.nvidia.com

# rapids pip package patch: link ucx libraries in ucx-py to default location searched by raft_dask
ln -s /databricks/python/lib/python3.8/site-packages/ucx_py_cu11.libs/ucx /usr/lib/ucx

# install spark-rapids-ml
unzip ${SPARK_RAPIDS_ML_ZIP} -d /databricks/python3/lib/python3.8/site-packages

