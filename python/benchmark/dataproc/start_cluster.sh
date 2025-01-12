#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.
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

cluster_type=${1:-gpu}

# configure arguments
if [[ -z ${COMPUTE_REGION} ]]; then
    echo "Please export COMPUTE_REGION per README.md"
    exit 1
fi

if [[ -z ${GCS_BUCKET} ]]; then
    echo "Please export GCS_BUCKET per README.md"
    exit 1
fi

BENCHMARK_HOME=${BENCHMARK_HOME:-${GCS_BUCKET}/benchmark}

gpu_args=$(cat <<EOF
--master-accelerator type=nvidia-tesla-t4,count=1
--worker-accelerator type=nvidia-tesla-t4,count=1
--initialization-actions gs://${BENCHMARK_HOME}/spark-rapids.sh,gs://${BENCHMARK_HOME}/init_benchmark.sh
--initialization-action-timeout=20m
--metadata gpu-driver-provider="NVIDIA"
--metadata rapids-runtime=SPARK
--metadata benchmark-home=${BENCHMARK_HOME}
EOF
)

cpu_args=$(cat <<EOF
--initialization-actions gs://${BENCHMARK_HOME}/init_benchmark.sh
EOF
)

if [[ ${cluster_type} == "gpu" ]]; then
    extra_args=${gpu_args}
elif [[ ${cluster_type} == "cpu" ]]; then
    extra_args=${cpu_args}
else
    echo "unknown cluster type ${cluster_type}"
    echo "usage: ./${script_name} cpu|gpu"
    exit 1
fi

# start cluster if not already running
cluster_name=${CLUSTER_NAME:-"${USER}-spark-rapids-ml-${cluster_type}"}

gcloud dataproc clusters list | grep "${cluster_name}"
if [[ $? == 0 ]]; then
    echo "WARNING: Cluster ${cluster_name} is already started."
else
    set -x
    gcloud dataproc clusters create ${cluster_name} \
    --image-version=2.2-ubuntu22 \
    --region ${COMPUTE_REGION} \
    --master-machine-type n1-standard-16 \
    --num-workers 2 \
    --worker-min-cpu-platform="Intel Skylake" \
    --worker-machine-type n1-standard-16 \
    --num-worker-local-ssds 4 \
    --worker-local-ssd-interface=NVME \
    ${extra_args} \
    --optional-components=JUPYTER \
    --bucket ${GCS_BUCKET} \
    --enable-component-gateway \
    --max-idle "30m" \
    --subnet=default \
    --no-shielded-secure-boot
fi
