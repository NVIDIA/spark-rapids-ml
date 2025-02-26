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

# setup arguments
if [[ -z ${COMPUTE_REGION} ]]; then
    echo "Please export COMPUTE_REGION per README.md"
    exit 1
fi

common_args=$(cat <<EOF
--spark_confs spark.executorEnv.CUPY_CACHE_DIR=/tmp/.cupy \
--spark_confs spark.locality.wait=0s \
--spark_confs spark.sql.execution.arrow.pyspark.enabled=true \
--spark_confs spark.sql.execution.arrow.maxRecordsPerBatch=100000 \
--spark_confs spark.sql.execution.sortBeforeRepartition=false
EOF
)

gpu_args=$(cat <<EOF
--num_cpus=0 \
--num_gpus=2 \
--spark_confs spark.executor.resource.gpu.amount=1 \
--spark_confs spark.task.resource.gpu.amount=1 \
--spark_confs spark.rapids.memory.gpu.pooling.enabled=false
EOF
)

cpu_args=$(cat <<EOF
--num_cpus=16 \
--num_gpus=0
EOF
)

num_runs=1
if [[ ${cluster_type} == "gpu" ]]; then
    extra_args=${gpu_args}
    n_streams="--n_streams 4"
    kmeans_runs=1 #3
    rf_runs=1 #3
    rf_cpu_options=""
elif [[ ${cluster_type} == "cpu" ]]; then
    # kmeans and random forest take a long time on cpu cluster, so only do 1 run each
    extra_args=${cpu_args}
    n_streams=""
    kmeans_runs=1
    rf_runs=1
    rf_cpu_options="--subsamplingRate=0.5"
else
    echo "unknown cluster type ${cluster_type}"
    echo "usage: $0 cpu|gpu"
    exit 1
fi

BENCHMARK_DATA_HOME=gs://spark-rapids-ml-benchmarking/datasets

# start benchmark cluster
./start_cluster.sh $cluster_type
if [[ $? != 0 ]]; then
    echo "Failed to start cluster."
    exit 1
fi

cluster_name=${CLUSTER_NAME:-"${USER}-spark-rapids-ml-${cluster_type}"}

# run benchmarks
sep="=================="

echo
echo "$sep algo: kmeans $sep"
for i in `seq $kmeans_runs`; do
    set -x
    gcloud dataproc jobs submit pyspark \
    ../benchmark_runner.py \
    --cluster=${cluster_name} \
    --region=${COMPUTE_REGION} \
    -- \
    kmeans \
    --num_runs 1 \
    --k 1000 \
    --tol 1.0e-20 \
    --maxIter 30 \
    --initMode random \
    --no_cache \
    --train_path "${BENCHMARK_DATA_HOME}/pca/1m_3k_singlecol_float32_50_files.parquet" \
    ${common_args} \
    ${extra_args} 2>&1 | tee kmeans_$i.out
    set +x
    sleep 30
done

# # Note: requires 24GB GPU, i.e. A10 or A100
# echo
# echo "$sep algo: knn $sep"
# for i in `seq $num_runs`; do
#     set -x
#     gcloud dataproc jobs submit pyspark \
#     ../benchmark_runner.py \
#     --cluster=${cluster_name} \
#     --region=${COMPUTE_REGION} \
#     -- \
#     knn \
#     --k 3 \
#     --train_path "${BENCHMARK_DATA_HOME}/pca/1m_3k_singlecol_float32_50_files.parquet" \
#     ${common_args} \
#     ${extra_args} 2>&1 | tee knn_$i.out
#     set +x
#     sleep 30
# done

echo
echo "$sep algo: pca $sep"
for i in `seq $num_runs`; do
    set -x
    gcloud dataproc jobs submit pyspark \
    ../benchmark_runner.py \
    --cluster=${cluster_name} \
    --region=${COMPUTE_REGION} \
    -- \
    pca \
    --num_runs 1 \
    --k 3 \
    --no_cache \
    --train_path "${BENCHMARK_DATA_HOME}/pca/1m_3k_singlecol_float32_50_files.parquet" \
    ${common_args} \
    ${extra_args} 2>&1 | tee pca_$i.out
    set +x
    sleep 30
done

echo
echo "$sep algo: linear regression - no regularization $sep"
for i in `seq $num_runs`; do
    set -x
    gcloud dataproc jobs submit pyspark \
    ../benchmark_runner.py \
    --cluster=${cluster_name} \
    --region=${COMPUTE_REGION} \
    -- \
    linear_regression \
    --num_runs 1 \
    --regParam 0.0 \
    --elasticNetParam 0.0 \
    --standardization False \
    --train_path "${BENCHMARK_DATA_HOME}/linear_regression/1m_3k_standardized_singlecol_float32_50_files.parquet" \
    ${common_args} \
    ${extra_args} 2>&1 | tee linear_regression_noreg_$i.out
    set +x
    sleep 30
done

echo
echo "$sep algo: linear regression - elasticnet regularization $sep"
for i in `seq $num_runs`; do
    set -x
    gcloud dataproc jobs submit pyspark \
    ../benchmark_runner.py \
    --cluster=${cluster_name} \
    --region=${COMPUTE_REGION} \
    -- \
    linear_regression \
    --num_runs 1 \
    --regParam 0.00001 \
    --elasticNetParam 0.5 \
    --tol 1.0e-30 \
    --maxIter 10 \
    --standardization False \
    --train_path "${BENCHMARK_DATA_HOME}/linear_regression/1m_3k_standardized_singlecol_float32_50_files.parquet" \
    ${common_args} \
    ${extra_args} 2>&1 | tee linear_regression_elasticnet_$i.out
    set +x
    sleep 30
done

echo
echo "$sep algo: linear regression - ridge regularization $sep"
for i in `seq $num_runs`; do
    set -x
    gcloud dataproc jobs submit pyspark \
    ../benchmark_runner.py \
    --cluster=${cluster_name} \
    --region=${COMPUTE_REGION} \
    -- \
    linear_regression \
    --num_runs 1 \
    --regParam 0.00001 \
    --elasticNetParam 0.0 \
    --tol 1.0e-30 \
    --maxIter 10 \
    --standardization False \
    --train_path "${BENCHMARK_DATA_HOME}/linear_regression/1m_3k_standardized_singlecol_float32_50_files.parquet" \
    ${common_args} \
    ${extra_args} 2>&1 | tee linear_regression_ridge_$i.out
    set +x
    sleep 30
done

echo
echo "$sep algo: random forest classification $sep"
for i in `seq $rf_runs`; do
    set -x
    gcloud dataproc jobs submit pyspark \
    ../benchmark_runner.py \
    --cluster=${cluster_name} \
    --region=${COMPUTE_REGION} \
    -- \
    random_forest_classifier \
    ${n_streams} \
    ${rf_cpu_options} \
    --num_runs 1 \
    --numTrees 50 \
    --maxBins 128 \
    --maxDepth 13 \
    --train_path "${BENCHMARK_DATA_HOME}/classification/1m_3k_singlecol_float32_50_1_3_inf_red_files.parquet" \
    ${common_args} \
    ${extra_args} 2>&1 | tee random_forest_classifier_$i.out
    set +x
    sleep 30
done

echo
echo "$sep algo: random forest regression $sep"
for i in `seq $rf_runs`; do
    set -x
    gcloud dataproc jobs submit pyspark \
    ../benchmark_runner.py \
    --cluster=${cluster_name} \
    --region=${COMPUTE_REGION} \
    -- \
    random_forest_regressor \
    ${n_streams} \
    ${rf_cpu_options} \
    --num_runs 1 \
    --numTrees 30 \
    --maxBins 128 \
    --maxDepth 6 \
    --train_path "${BENCHMARK_DATA_HOME}/linear_regression/1m_3k_singlecol_float32_50_files.parquet" \
    ${common_args} \
    ${extra_args} 2>&1 | tee random_forest_regressor_$i.out
    set +x
    sleep 30
done

echo
echo "$sep algo: logistic regression $sep"
for i in `seq $num_runs`; do
    set -x
    gcloud dataproc jobs submit pyspark \
    ../benchmark_runner.py \
    --cluster=${cluster_name} \
    --region=${COMPUTE_REGION} \
    -- \
    logistic_regression \
    --num_runs 1 \
    --standardization False \
    --maxIter 200 \
    --tol 1e-30 \
    --regParam 0.00001 \
    --train_path "${BENCHMARK_DATA_HOME}/classification/1m_3k_singlecol_float32_50_1_3_inf_red_files.parquet" \
    ${common_args} \
    ${extra_args} 2>&1 | tee logistic_regression_$i.out
    set +x
    sleep 30
done
