#! /bin/bash
#
# Usage: ./run_benchmark.sh cpu|gpu <mode> [<args>]
# where <mode> can be:
#     all
#     kmeans
#     knn
#     linear_regression
#     pca
#     random_forest_classifier
#     random_forest_regressor
#
# By default, the tests should target a single node with a single GPU for CI/CD purposes.
# For more advanced configurations, use extra <args>.

# if multiple gpus are available, set CUDA_VISIBLE_DEVICES to comma separated list of gpu indices to use
export CUDA_VISIBLE_DEVICES=0

cluster_type=${1:-gpu}
shift
local_threads=4
if [[ $cluster_type == "gpu" ]]; then
    num_cpus=0
    num_gpus=$(( `echo $CUDA_VISIBLE_DEVICES | grep -o ',' | wc -l` + 1 ))
elif [[ $cluster_type == "cpu" ]]; then
    num_cpus=$local_threads
    num_gpus=0
else
    echo "unknown cluster type $cluster_type"
    echo "usage: $0 cpu|gpu mode [extra-args] "
    exit 1
fi

MODE=${1:-all}
shift
EXTRA_ARGS=$@

unset SPARK_HOME


# stop on first fail
set -e

# KMeans
if [[ "${MODE}" == "kmeans" ]] || [[ "${MODE}" == "all" ]]; then
    if [[ ! -d "/tmp/blobs/5k_3k_float32.parquet" ]]; then
        python ./benchmark/gen_data.py default \
            --num_rows 5000 \
            --num_cols 3000 \
            --dtype "float32" \
            --feature_type "array" \
            --output_dir "/tmp/default/5k_3k_float32.parquet" \
            --spark_conf "spark.master=local[$local_threads]" \
            --spark_confs "spark.driver.memory=128g"
    fi

    python ./benchmark/benchmark_runner.py kmeans \
        --k 1000 \
        --num_gpus $num_gpus \
        --num_cpus $num_cpus \
        --no_cache \
        --train_path "/tmp/default/5k_3k_float32.parquet" \
        --report_path "report_kmeans.csv" \
        --spark_confs "spark.master=local[$local_threads]" \
        --spark_confs "spark.driver.memory=128g" \
        --spark_confs "spark.sql.execution.arrow.maxRecordsPerBatch=20000" \
        ${EXTRA_ARGS}
fi

# KNearestNeighbors
if [[ "${MODE}" == "knn" ]] || [[ "${MODE}" == "all" ]]; then
    if [[ ! -d "/tmp/blobs/5k_3k_float32.parquet" ]]; then
        python ./benchmark/gen_data.py blobs \
            --num_rows 5000 \
            --num_cols 3000 \
            --dtype "float32" \
            --feature_type "array" \
            --output_dir "/tmp/blobs/5k_3k_float32.parquet" \
            --spark_conf "spark.master=local[$local_threads]" \
            --spark_confs "spark.driver.memory=128g"
    fi

    python ./benchmark/benchmark_runner.py knn \
        --n_neighbors 3 \
        --num_gpus $num_gpus \
        --num_cpus $num_cpus \
        --no_cache \
        --train_path "/tmp/blobs/5k_3k_float32.parquet" \
        --report_path "report_knn.csv" \
        --spark_confs "spark.master=local[$local_threads]" \
        --spark_confs "spark.driver.memory=128g" \
        --spark_confs "spark.sql.execution.arrow.maxRecordsPerBatch=20000" \
        ${EXTRA_ARGS}
fi

# Linear Regression
if [[ "${MODE}" == "linear_regression" ]] || [[ "${MODE}" == "all" ]]; then
    if [[ ! -d "/tmp/regression/5k_3k_float32.parquet" ]]; then
        python ./benchmark/gen_data.py regression \
            --num_rows 5000 \
            --num_cols 3000 \
            --dtype "float32" \
            --feature_type "array" \
            --output_dir "/tmp/regression/5k_3k_float32.parquet" \
            --spark_conf "spark.master=local[$local_threads]" \
            --spark_confs "spark.driver.memory=128g"
    fi

    python ./benchmark/benchmark_runner.py linear_regression \
        --num_gpus $num_gpus \
        --num_cpus $num_cpus \
        --train_path "/tmp/regression/5k_3k_float32.parquet" \
        --transform_path "/tmp/regression/5k_3k_float32.parquet" \
        --report_path "report_linear_regression.csv" \
        --spark_confs "spark.master=local[$local_threads]" \
        --spark_confs "spark.driver.memory=128g" \
        --spark_confs "spark.sql.execution.arrow.maxRecordsPerBatch=20000" \
        ${EXTRA_ARGS}
fi

# PCA
if [[ "${MODE}" == "pca" ]] || [[ "${MODE}" == "all" ]]; then
    if [[ ! -d "/tmp/blobs/5k_3k_float32.parquet" ]]; then
        python ./benchmark/gen_data.py low_rank_matrix \
            --num_rows 5000 \
            --num_cols 3000 \
            --dtype "float32" \
            --feature_type "array" \
            --output_dir "/tmp/low_rank_matrix/5k_3k_float32.parquet" \
            --spark_conf "spark.master=local[$local_threads]" \
            --spark_confs "spark.driver.memory=128g"
    fi

    python ./benchmark/benchmark_runner.py pca \
        --k 3 \
        --num_gpus $num_gpus \
        --num_cpus $num_cpus \
        --no_cache \
        --train_path "/tmp/blobs/5k_3k_float32.parquet" \
        --report_path "report_pca.csv" \
        --spark_confs "spark.master=local[$local_threads]" \
        --spark_confs "spark.driver.memory=128g" \
        --spark_confs "spark.sql.execution.arrow.maxRecordsPerBatch=20000" \
        ${EXTRA_ARGS}

#    # standalone mode
#    SPARK_MASTER=spark://hostname:port
#    tar -czvf spark-rapids-ml.tar.gz -C ./src .
#
#    python ./benchmark/bench_pca.py \
#        --n_components 3 \
#        --num_gpus 2 \
#        --num_cpus 0 \
#        --num_runs 3 \
#        --no_cache \
#        --parquet_path "/tmp/blobs/5k_3k_float32.parquet" \
#        --report_path "./report_standalone.csv" \
#        --spark_confs "spark.master=${SPARK_MASTER}" \
#        --spark_confs "spark.driver.memory=128g" \
#        --spark_confs "spark.sql.execution.arrow.maxRecordsPerBatch=200000"  \
#        --spark_confs "spark.executor.memory=128g" \
#        --spark_confs "spark.rpc.message.maxSize=2000" \
#        --spark_confs "spark.pyspark.python=${PYTHON_ENV_PATH}" \
#        --spark_confs "spark.submit.pyFiles=./spark-rapids-ml.tar.gz" \
#        --spark_confs "spark.task.resource.gpu.amount=1" \
#        --spark_confs "spark.executor.resource.gpu.amount=1"
fi

# Random Forest Classification
if [[ "${MODE}" == "random_forest_classifier" ]] || [[ "${MODE}" == "all" ]]; then
    if [[ ! -d /tmp/classification/5k_3k_float32.parquet ]]; then
        python ./benchmark/gen_data.py classification \
            --num_rows 5000 \
            --num_cols 3000 \
            --dtype "float32" \
            --feature_type "array" \
            --output_dir "/tmp/classification/5k_3k_float32.parquet" \
            --spark_conf "spark.master=local[$local_threads]" \
            --spark_confs "spark.driver.memory=128g"
    fi

    python ./benchmark/benchmark_runner.py random_forest_classifier \
        --num_gpus $num_gpus \
        --num_cpus $num_cpus \
        --train_path "/tmp/classification/5k_3k_float32.parquet" \
        --transform_path "/tmp/classification/5k_3k_float32.parquet" \
        --report_path "report_rf_classifier.csv" \
        --spark_confs "spark.master=local[$local_threads]" \
        --spark_confs "spark.driver.memory=128g" \
        --spark_confs "spark.sql.execution.arrow.maxRecordsPerBatch=20000" \
        ${EXTRA_ARGS}
fi

# Random Forest Regression
if [[ "${MODE}" == "random_forest_regressor" ]] || [[ "${MODE}" == "all" ]]; then
    if [[ ! -d /tmp/regression/5k_3k_float32.parquet ]]; then
        python ./benchmark/gen_data.py regression \
            --num_rows 5000 \
            --num_cols 3000 \
            --dtype "float32" \
            --feature_type "array" \
            --output_dir "/tmp/regression/5k_3k_float32.parquet" \
            --spark_conf "spark.master=local[$local_threads]" \
            --spark_confs "spark.driver.memory=128g"
    fi

    python ./benchmark/benchmark_runner.py random_forest_regressor \
        --num_gpus $num_gpus \
        --num_cpus $num_cpus \
        --train_path "/tmp/regression/5k_3k_float32.parquet" \
        --transform_path "/tmp/regression/5k_3k_float32.parquet" \
        --report_path "report_rf_regressor.csv" \
        --spark_confs "spark.master=local[$local_threads]" \
        --spark_confs "spark.driver.memory=128g" \
        --spark_confs "spark.sql.execution.arrow.maxRecordsPerBatch=20000" \
        ${EXTRA_ARGS}
fi
