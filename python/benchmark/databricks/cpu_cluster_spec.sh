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
    "num_workers": $(( $num_cpus / 8)),
    "cluster_name": "$cluster_name",
    "spark_version": "${db_version}.x-cpu-ml-scala2.12",
    "spark_conf": {},
    "aws_attributes": {
        "first_on_demand": 1,
        "availability": "SPOT_WITH_FALLBACK",
        "zone_id": "us-west-2a",
        "spot_bid_price_percent": 100,
        "ebs_volume_type": "GENERAL_PURPOSE_SSD",
        "ebs_volume_count": 3,
        "ebs_volume_size": 100
    },
    "node_type_id": "m5.2xlarge",
    "driver_node_type_id": "m5.2xlarge",
    "ssh_public_keys": [],
    "custom_tags": {},
    "cluster_log_conf": {
        "dbfs": {
            "destination": "dbfs:${BENCHMARK_HOME}/cluster_logs/${cluster_name}"
        }
    },
    "spark_env_vars": {},
    "autotermination_minutes": 30,
    "enable_elastic_disk": false,
    "init_scripts": [
        {
            "workspace": {
                "destination": "${INIT_SCRIPT_DIR}/init-cpu.sh"
            }
        }
    ],
    "enable_local_disk_encryption": false,
    "runtime_engine": "STANDARD"
}
EOF
