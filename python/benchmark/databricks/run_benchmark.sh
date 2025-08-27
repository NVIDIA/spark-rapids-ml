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


cluster_type=${1:-gpu_etl}
db_version=${2:-13.3}

if [[ $cluster_type == "gpu" || $cluster_type == "gpu_etl" ]]; then
    num_cpus=0
    num_gpus=2
elif [[ $cluster_type == "cpu" ]]; then
    num_cpus=16
    num_gpus=0
else
    echo "unknown cluster type $cluster_type"
    echo "usage: $0 cpu|gpu|gpu_etl [12.2|13.3|14.3|15.4]" 
    exit 1
fi

if [[ $db_version > 13.3 && $cluster_type == "gpu_etl" ]]; then
    echo "spark rapids etl plugin is not supported on databricks ${db_version}"
    echo "please specify db_version 12.2 or 13.3 for cluster type gpu_etl"
    exit 1
fi

source benchmark_utils.sh

BENCHMARK_DATA_HOME=s3a://spark-rapids-ml-bm-datasets-public

# creates cluster and sets CLUSTER_ID equal to created cluster's id
create_cluster $cluster_type

# cpu options. reset below if gpu cluster is used
# kmeans and random forest take a long time on cpu cluster, so do 1 run each
# do 3 runs in all other cases

n_streams=""
kmeans_runs=1
rf_runs=1
rf_cpu_options="--subsamplingRate=0.5"

if [[ $cluster_type =~ "gpu" ]]
then
    n_streams="--n_streams 4"
    kmeans_runs=3
    rf_runs=3
    rf_cpu_options=""
fi
num_runs=3

sep="=================="

echo
echo "$sep algo: kmeans $sep"
for i in `seq $kmeans_runs`; do run_bm kmeans \
--num_runs 1 \
--k 1000 \
--tol 1.0e-20 \
--maxIter 30 \
--initMode random \
--no_cache \
--train_path "${BENCHMARK_DATA_HOME}/pca/1m_3k_singlecol_float32_50_files.parquet" \
&& sleep 90; done

echo
echo "$sep algo: pca $sep"
for i in `seq $num_runs`; do run_bm pca \
--num_runs 1 \
--k 3 \
--no_cache \
--train_path "${BENCHMARK_DATA_HOME}/low_rank_matrix/1m_3k_singlecol_float32_50_files.parquet" \
&& sleep 90; done

echo
echo "$sep algo: linear regression - no regularization $sep"
for i in `seq $num_runs`; do run_bm linear_regression \
--num_runs 1 \
--regParam 0.0 \
--elasticNetParam 0.0 \
--standardization False \
--train_path "${BENCHMARK_DATA_HOME}/linear_regression/1m_3k_standardized_singlecol_float32_50_files.parquet" \
&& sleep 90; done

echo
echo "$sep algo: linear regression - elasticnet regularization $sep"
for i in `seq $num_runs`; do run_bm linear_regression \
--num_runs 1 \
--regParam 0.00001 \
--elasticNetParam 0.5 \
--tol 1.0e-30 \
--maxIter 10 \
--standardization False \
--train_path "${BENCHMARK_DATA_HOME}/linear_regression/1m_3k_standardized_singlecol_float32_50_files.parquet" \
&& sleep 90; done

echo
echo "$sep algo: linear regression - ridge regularization $sep"
for i in `seq $num_runs`; do run_bm linear_regression \
--num_runs 1 \
--regParam 0.00001 \
--elasticNetParam 0.0 \
--tol 1.0e-30 \
--maxIter 10 \
--standardization False \
--train_path "${BENCHMARK_DATA_HOME}/linear_regression/1m_3k_standardized_singlecol_float32_50_files.parquet" \
&& sleep 90; done

echo
echo "$sep algo: random forest classification $sep"
for i in `seq $rf_runs`; do run_bm random_forest_classifier \
$n_streams \
$rf_cpu_options \
--num_runs 1 \
--numTrees 50 \
--maxBins 128 \
--maxDepth 13 \
--train_path "${BENCHMARK_DATA_HOME}/classification/1m_3k_singlecol_float32_50_1_3_inf_red_files.parquet" \
&& sleep 90; done

echo
echo "$sep algo: random forest regression $sep"
for i in `seq $rf_runs`; do run_bm random_forest_regressor \
$n_streams \
$rf_cpu_options \
--num_runs 1 \
--numTrees 30 \
--maxBins 128 \
--maxDepth 6 \
--train_path "${BENCHMARK_DATA_HOME}/linear_regression/1m_3k_singlecol_float32_50_files.parquet" && \
sleep 90; done

echo
echo "$sep algo: logistic regression $sep"
for i in `seq $num_runs`; do run_bm logistic_regression \
--num_runs 1 \
--standardization False \
--maxIter 200 \
--tol 1e-30 \
--regParam 0.00001 \
--train_path "${BENCHMARK_DATA_HOME}/classification/1m_3k_singlecol_float32_50_1_3_inf_red_files.parquet" \
&& sleep 90; done
