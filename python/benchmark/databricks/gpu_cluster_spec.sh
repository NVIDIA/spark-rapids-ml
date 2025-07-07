# Copyright (c) 2024, NVIDIA CORPORATION.
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

# needed for bm script arguments
cat <<EOF
{
    "num_workers": $num_gpus,
    "cluster_name": "$cluster_name",
    "spark_version": "${db_version}.x-gpu-ml-scala2.12",
    "spark_conf": {
        "spark.task.resource.gpu.amount": "0.25",
        "spark.task.cpus": "1",
        "spark.databricks.delta.preview.enabled": "true",
        "spark.python.worker.reuse": "true",
        "spark.executorEnv.PYTHONPATH": "/databricks/spark/python",
        "spark.sql.files.minPartitionNum": "2",
        "spark.sql.execution.arrow.maxRecordsPerBatch": "10000",
        "spark.executor.cores": "8",
        "spark.executor.memory": "5g",
        "spark.locality.wait": "0s",
        "spark.sql.execution.sortBeforeRepartition": "false",
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.sql.files.maxPartitionBytes": "2000000000000",
        "spark.databricks.delta.optimizeWrite.enabled": "false"
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
    "autotermination_minutes": 30,
    "enable_elastic_disk": false,
    "init_scripts": [
        {
            "workspace": {
                "destination": "${INIT_SCRIPT_DIR}/init-pip-cuda-12.0.sh"
            }
        }
    ],
    "enable_local_disk_encryption": false,
    "runtime_engine": "STANDARD"
}
EOF
