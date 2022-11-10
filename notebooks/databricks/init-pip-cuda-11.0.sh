#!/bin/bash
SPARKCUML_ZIP=
RAPIDS_VERSION=22.10.0

curl -L https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/${RAPIDS_VERSION}/rapids-4-spark_2.12-${RAPIDS_VERSION}.jar -o /databricks/jars/rapids-4-spark_2.12-${RAPIDS_VERSION}.jar

# delete symlink to allow cupy to find some .so files in 11-3 directories
rm /usr/local/cuda

# install cudf, cuml
/databricks/python/bin/pip install cudf-cu11 dask-cudf-cu11 --extra-index-url=https://pypi.ngc.nvidia.com
/databricks/python/bin/pip install cuml-cu11 --extra-index-url=https://pypi.ngc.nvidia.com
# downgrade cupy for compatibility with 110 databricks cuda rt
/databricks/python/bin/pip uninstall -y cupy-cuda115
/databricks/python/bin/pip install cupy-cuda110

# force numba upgrade to elimnate cudf import error
/databricks/python/bin/pip install numba==0.56

# needed to avoid cudf import error
/databricks/python/bin/pip install nvidia-ml-py

# install spark-cuml
unzip ${SPARKCUML_ZIP} -d /databricks/python3/lib/python3.8/site-packages

# patches for libucp
ln -s /databricks/python3/lib/python3.8/site-packages/raft_dask_cu11.libs/libucp-26342de7.so.0.0.0 /usr/lib/libucp.so



