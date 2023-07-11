## Running notebooks on Databricks

If you already have a Databricks account, you can run the example notebooks on a Databricks cluster, as follows:
- Install the [databricks-cli](https://docs.databricks.com/dev-tools/cli/index.html).
- Configure it with your workspace URL and an [access token](https://docs.databricks.com/dev-tools/api/latest/authentication.html).  For demonstration purposes, we will configure a new [connection profile](https://docs.databricks.com/dev-tools/cli/index.html#connection-profiles) named `spark-rapids-ml`.  If you already have a connection profile, just set the `PROFILE` environment variable accordingly and skip the configure step.
  ```
  export PROFILE=spark-rapids-ml
  databricks configure --token --profile ${PROFILE}
  ```
- Create a zip file for the `spark-rapids-ml` package.
  ```
  cd spark-rapids-ml/python/src
  zip -r spark_rapids_ml.zip spark_rapids_ml
  ```
- Copy the zip file to DBFS, setting `SAVE_DIR` to the directory of your choice.
  ```"
  export SAVE_DIR="/path/to/save/artifacts"
  databricks fs cp spark_rapids_ml.zip dbfs:${SAVE_DIR}/spark_rapids_ml.zip --profile ${PROFILE}
  ```
- Edit the [init-pip-cuda-11.8.sh](init-pip-cuda-11.8.sh) init script to set the `SPARK_RAPIDS_ML_ZIP` variable to the DBFS location used above.
  ```
  cd spark-rapids-ml/notebooks/databricks
  sed -i"" -e "s;/path/to/zip/file;${SAVE_DIR}/spark_rapids_ml.zip;" init-pip-cuda-11.8.sh
  ```
  **Note**: the `databricks` CLI requires the `dbfs:` prefix for all DBFS paths, but inside the spark nodes, DBFS will be mounted to a local `/dbfs` volume, so the path prefixes will be slightly different depending on the context.

  **Note**: this init script does the following on each Spark node:
  - updates the CUDA runtime to 11.8 (required for Spark Rapids ML dependencies).
  - downloads and installs the [Spark-Rapids](https://github.com/NVIDIA/spark-rapids) plugin for accelerating data loading and Spark SQL.
  - installs various `cuXX` dependencies via pip.
- Copy the modified `init-pip-cuda-11.8.sh` init script to DBFS.
  ```
  databricks fs cp init-pip-cuda-11.8.sh dbfs:${SAVE_DIR}/init-pip-cuda-11.8.sh --profile ${PROFILE}
  ```
- Create a cluster using **Databricks 11.3 LTS ML GPU Runtime** using at least two single-gpu workers and add the following configurations to the **Advanced options**.
  - **Init Scripts**
    - add the DBFS path to the uploaded init script, e.g. `dbfs:/path/to/save/artifacts/init-pip-cuda-11.8.sh`.
  - **Spark**
    - **Spark config**
      ```
      spark.task.resource.gpu.amount 1
      spark.databricks.delta.preview.enabled true
      spark.python.worker.reuse true
      spark.executorEnv.PYTHONPATH /databricks/jars/rapids-4-spark_2.12-23.06.0.jar:/databricks/spark/python
      spark.sql.execution.arrow.maxRecordsPerBatch 100000
      spark.rapids.memory.gpu.minAllocFraction 0.0001
      spark.plugins com.nvidia.spark.SQLPlugin
      spark.locality.wait 0s
      spark.sql.cache.serializer com.nvidia.spark.ParquetCachedBatchSerializer
      spark.rapids.memory.gpu.pooling.enabled false
      spark.rapids.sql.explain ALL
      spark.rapids.memory.gpu.reserve 20
      spark.sql.execution.sortBeforeRepartition false
      spark.rapids.sql.python.gpu.enabled true
      spark.rapids.memory.pinnedPool.size 2G
      spark.python.daemon.module rapids.daemon_databricks
      spark.rapids.sql.batchSizeBytes 512m
      spark.sql.adaptive.enabled false
      spark.databricks.delta.optimizeWrite.enabled false
      spark.rapids.sql.concurrentGpuTasks 2
      spark.sql.execution.arrow.pyspark.enabled true
      ```
    - **Environment variables**
      ```
      LIBCUDF_CUFILE_POLICY=OFF
      LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64
      NCCL_DEBUG=INFO
      ```
- Start the configured cluster.
- Select your workspace and upload the desired [notebook](../) via `Import` in the drop down menu for your workspace.
