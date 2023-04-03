#!/bin/bash
# set portion of path below after /dbfs/ to dbfs zip file location
SPARK_RAPIDS_ML_ZIP=/dbfs/path/to/spark-rapids-ml.zip
BENCHMARK_ZIP=/dbfs/path/to/benchmark.zip

# install spark-rapids-ml
unzip ${SPARK_RAPIDS_ML_ZIP} -d /databricks/python3/lib/python3.8/site-packages
unzip ${BENCHMARK_ZIP} -d /databricks/python3/lib/python3.8/site-packages

