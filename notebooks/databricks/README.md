## Running notebooks on Databricks

If you already have a Databricks account, you can run the example notebooks on a Databricks cluster, as follows:
- Install the latest [databricks-cli](https://docs.databricks.com/dev-tools/cli/index.html).  Note that Databricks has deprecated the legacy python based cli in favor of a self contained executable. Make sure the new version is first on the executables PATH after installation.
- Configure it with your workspace URL and an [access token](https://docs.databricks.com/dev-tools/api/latest/authentication.html).  For demonstration purposes, we will configure a new [connection profile](https://docs.databricks.com/dev-tools/cli/index.html#connection-profiles) named `spark-rapids-ml`.  If you already have a connection profile, just set the `PROFILE` environment variable accordingly and skip the configure step.
  ```bash
  export PROFILE=spark-rapids-ml
  databricks configure --token --profile ${PROFILE}
  ```
- Copy the init scripts to your *workspace* (not DBFS) (ex. workspace directory: /Users/< databricks-user-name >/init_scripts).
  ```bash
  export WS_SAVE_DIR="/path/to/directory/in/workspace"
  databricks workspace mkdirs ${WS_SAVE_DIR} --profile ${PROFILE}
  databricks workspace import --format AUTO --file init-pip-cuda-12.0.sh ${WS_SAVE_DIR}/init-pip-cuda-12.0.sh --profile ${PROFILE}
  ```
  **Note**: the init script does the following on each Spark node:
  - updates the CUDA runtime to 12.0 (required for Spark Rapids ML dependencies).
  - downloads and installs the [Spark-Rapids](https://github.com/NVIDIA/spark-rapids) plugin for accelerating data loading and Spark SQL.
  - installs various `cuXX` dependencies via pip.
  - if the cluster environment variable `SPARK_RAPIDS_ML_NO_IMPORT_ENABLED=1` is define (see below), the init script also modifies a Databricks notebook kernel startup script to enable no-import change UX for the cluster.  See [no-import-change](../README.md#no-import-change).
- Create a cluster using **Databricks 13.3 LTS ML GPU Runtime** using at least two single-gpu workers and add the following configurations to the **Advanced options**.
  - **Init Scripts**
    - add the workspace path to the uploaded init script `${WS_SAVE_DIR}/init-pip-cuda-12.0.sh` as set above (but substitute variables manually in the form).
  - **Spark**
    - **Spark config**
      ```
      spark.task.resource.gpu.amount 0.125
      spark.databricks.delta.preview.enabled true
      spark.python.worker.reuse true
      spark.executorEnv.PYTHONPATH /databricks/jars/rapids-4-spark_2.12-25.08.0.jar:/databricks/spark/python
      spark.sql.execution.arrow.maxRecordsPerBatch 100000
      spark.plugins com.nvidia.spark.SQLPlugin
      spark.locality.wait 0s
      spark.sql.cache.serializer com.nvidia.spark.ParquetCachedBatchSerializer
      spark.rapids.memory.gpu.pool NONE
      spark.rapids.sql.explain ALL
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
      NCCL_DEBUG=INFO
      SPARK_RAPIDS_ML_NO_IMPORT_ENABLED=0
      ```
      If you wish to enable [no-import-change](../README.md#no-import-change) UX for the cluster, set `SPARK_RAPIDS_ML_NO_IMPORT_ENABLED=1` instead.  The init script checks this cluster environment variable and modifies the runtime accordingly.
- Start the configured cluster.
- Select your workspace and upload the desired [notebook](../) via `Import` in the drop down menu for your workspace.  For the no-import-change UX, you can try the example [kmeans-no-import-change.ipynb](../kmeans-no-import-change.ipynb).
