Assuming you already have a Databricks account, to run notebooks on Databricks do the following:
- If you don't already have it, install [databricks cli](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/) and create and save an [access token](https://docs.databricks.com/dev-tools/api/latest/authentication.html) to your workspace using the workspace UI.  You may need to create a new profile, in which case it should be supplied via the `--profile` option to all `databricks` cli commands issued below.
- Inside the [src](../../src/) directory, create a zip file of the `spark_rapids_ml` directory via `zip -r spark_rapids_ml.zip spark_rapids_ml` command at the top level of the repo and copy to a location in dbfs using the databricks cli command `databricks fs cp spark_rapids_ml.zip <dbfs:/dbfs location>` 
- Edit the file [init-pip-cuda-11.8.sh](init-pip-cuda-11.8.sh) to set the `SPARK_RAPIDS_ML_ZIP` variable to the `dbfs` location used above and upload the resulting modifed `.sh` file to some location in `dbfs` using the `databricks` cli.  Make a note of this `dbfs:/` path for use below.  Note that the setting for `SPARK_RAPIDS_ML_ZIP` here starts with `/dbfs/` which is where the `dbfs` filesystem is mounted in the databricks runtime containers.  The init script does the following:
  - configures the nodes with a more recent version of the CUDA runtime (11.8) required for Spark Rapids ML dependencies.
  - downloads and installs the Spark-Rapids SQL Plugin for accelerating the data loading and Spark SQL portions of ML jobs.
  - installs the cuXX dependencies via pip
- Create a cluster using Databricks 10.4LTS runtime using at least 2 single-gpu based workers and add the following configs to the respective Cluster Config fields/tabs/drop downs (e.g., under `Advanced options` in AWS Databricks):
  - add the `init-pip-cuda-11.8.sh` file `dbfs` location used above to the `Init Scripts` field/tab
  - add the following configs to the `Spark config` field that appears when selecting the `Spark` tab.
    ```
    spark.task.resource.gpu.amount 1
    spark.databricks.delta.preview.enabled true
    spark.python.worker.reuse true
    spark.executorEnv.PYTHONPATH /databricks/jars/rapids-4-spark_2.12-22.10.0.jar:/databricks/spark/python
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
  - add the following environment variable settings to the `Environment variables` field of the `Spark` tab.
    ```
    LIBCUDF_CUFILE_POLICY=OFF
    LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64
    NCCL_DEBUG=INFO
    ```
- Start the configured cluster.
- Select your workspace and upload the desired [notebook](../) via `Import` in the drop down menu for your workspace.
