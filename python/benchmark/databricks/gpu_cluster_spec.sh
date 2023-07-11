# needed for bm script arguments
cat <<EOF
{
    "num_workers": $num_gpus,
    "cluster_name": "$cluster_name",
    "spark_version": "11.3.x-gpu-ml-scala2.12",
    "spark_conf": {
        "spark.task.resource.gpu.amount": "0.25",
        "spark.task.cpus": "1",
        "spark.databricks.delta.preview.enabled": "true",
        "spark.python.worker.reuse": "true",
        "spark.executorEnv.PYTHONPATH": "/databricks/jars/rapids-4-spark_2.12-23.06.0.jar:/databricks/spark/python",
        "spark.sql.files.minPartitionNum": "2",
        "spark.sql.execution.arrow.maxRecordsPerBatch": "10000",
        "spark.executor.cores": "8",
        "spark.executor.memory": "5g",
        "spark.rapids.memory.gpu.minAllocFraction": "0.0001",
        "spark.plugins": "com.nvidia.spark.SQLPlugin",
        "spark.locality.wait": "0s",
        "spark.sql.cache.serializer": "com.nvidia.spark.ParquetCachedBatchSerializer",
        "spark.rapids.memory.gpu.pooling.enabled": "false",
        "spark.rapids.sql.explain": "ALL",
        "spark.rapids.memory.gpu.reserve": "20",
        "spark.sql.execution.sortBeforeRepartition": "false",
        "spark.rapids.sql.python.gpu.enabled": "true",
        "spark.rapids.memory.pinnedPool.size": "2G",
        "spark.python.daemon.module": "rapids.daemon_databricks",
        "spark.rapids.sql.batchSizeBytes": "512m",
        "spark.sql.adaptive.enabled": "false",
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.sql.files.maxPartitionBytes": "2000000000000",
        "spark.databricks.delta.optimizeWrite.enabled": "false",
        "spark.rapids.sql.concurrentGpuTasks": "2"
    },
    "aws_attributes": {
        "first_on_demand": 1,
        "availability": "SPOT_WITH_FALLBACK",
        "zone_id": "us-west-2a",
        "spot_bid_price_percent": 100,
        "ebs_volume_count": 0
    },
    "node_type_id": "g5.2xlarge",
    "driver_node_type_id": "g4dn.xlarge",
    "custom_tags": {},
    "cluster_log_conf": {
        "dbfs": {
            "destination": "dbfs:${BENCHMARK_HOME}/cluster_logs/${cluster_name}"
        }
    },
    "spark_env_vars": {
        "LIBCUDF_CUFILE_POLICY": "OFF",
        "LD_LIBRARY_PATH": "/usr/local/cuda/compat:/usr/local/cuda/lib64",
        "NCCL_DEBUG": "INFO"
    },
    "autotermination_minutes": 30,
    "enable_elastic_disk": false,
    "init_scripts": [
        {
            "dbfs": {
                "destination": "dbfs:${BENCHMARK_HOME}/init_script/init-pip-cuda-11.8.sh"
            }
        }
    ],
    "enable_local_disk_encryption": false,
    "runtime_engine": "STANDARD"
}
EOF
