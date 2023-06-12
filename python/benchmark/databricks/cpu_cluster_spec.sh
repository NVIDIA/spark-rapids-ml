# needed for bm script arguments
cat <<EOF
{
    "num_workers": $(( $num_cpus / 8)),
    "cluster_name": "$cluster_name",
    "spark_version": "11.3.x-cpu-ml-scala2.12",
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
            "dbfs": {
                "destination": "dbfs:${BENCHMARK_HOME}/init_script/init-cpu.sh"
            }
        }
    ],
    "enable_local_disk_encryption": false,
    "runtime_engine": "STANDARD"
}
EOF
