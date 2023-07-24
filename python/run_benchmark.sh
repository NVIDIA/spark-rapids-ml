#! /bin/bash
#
# Usage: ./run_benchmark.sh cpu|gpu|gpu_etl <mode> [<args>]
# where <mode> can be:
#     all
#     kmeans
#     knn
#     linear_regression
#     pca
#     random_forest_classifier
#     random_forest_regressor
#
#     and any comma separted list of the above like knn,linear_regression
#
# gpu_etl is gpu ML with Spark RAPIDS plugin for gpu accelerated data loading
# 
# By default, the tests should target a single node with a single GPU for CI/CD purposes.
# For more advanced configurations, use extra <args>.
# if multiple gpus are available, set CUDA_VISIBLE_DEVICES to comma separated list of gpu indices to use
#
# The following environment variables can be set on the command-line to control behavior with the indicated
# defaults:
# cuda_version=${cuda_version:-11}
# cluster_type=${1:-gpu}
# local_threads=${local_threads:-4}
# num_rows=${num_rows:-5000}
# num_cols=${num_cols:-3000}
# rapids_jar=${rapids_jar:-rapids-4-spark_2.12-$SPARK_RAPIDS_VERSION.jar}
#
# ex:  num_rows=1000000 num_cols=300 ./run_benchmark.sh gpu_etl kmeans,pca 
# would run gpu based kmeans and pca on respective synthetic datasets with 1m rows and 300 cols
# and would enable the Spark RAPIDS plugin for gpu accelerated data loading.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cuda_version=${cuda_version:-11}

cluster_type=${1:-gpu}
shift
local_threads=${local_threads:-4}
num_gpus=1
if [[ $cluster_type == "gpu" || $cluster_type == "gpu_etl" ]]; then
    num_cpus=0
    if [ -n $CUDA_VISIBLE_DEVICES ]; then
        num_gpus=$(( `echo $CUDA_VISIBLE_DEVICES | grep -o ',' | wc -l` + 1 ))
    fi
elif [[ $cluster_type == "cpu" ]]; then
    num_cpus=$local_threads
    num_gpus=0
else
    echo "unknown cluster type $cluster_type"
    echo "usage: $0 cpu|gpu|gpu_etl mode [extra-args] "
    exit 1
fi

num_runs=1

MODE=${1:-all}
shift
EXTRA_ARGS=$@

unset SPARK_HOME

# data set params
num_rows=${num_rows:-5000}
knn_num_rows=$num_rows
num_cols=${num_cols:-3000}

# for large num_rows (e.g. > 100k), set below to ./benchmark/gen_data_distributed.py and /tmp/distributed
gen_data_script=./benchmark/gen_data.py
#gen_data_script=./benchmark/gen_data_distributed.py
gen_data_root=/tmp/data

# if num_rows=1m => output_files=50, scale linearly
output_num_files=$(( ( $num_rows * $num_cols + 3000 * 20000 - 1 ) / ( 3000 * 20000 ) ))

# if num_cols=3000 => arrow_batch_size=20000, scale linearly for smaller number of columns
arrow_batch_size=$(( 20000 * ( ( $num_cols + 3000 - 1 ) / $num_cols ) ))


# stop on first fail
set -e

sep="=================="

common_confs=$( 
cat <<EOF 
--spark_confs spark.sql.execution.arrow.pyspark.enabled=true \
--spark_confs spark.sql.execution.arrow.maxRecordsPerBatch=$arrow_batch_size \
--spark_confs spark.python.worker.reuse=true \
--spark_confs spark.master=local[$local_threads] \
--spark_confs spark.driver.memory=128g \
--spark_confs spark.rapids.ml.uvm.enabled=true
EOF
)



if [[ $cluster_type == "gpu_etl" ]]
then
SPARK_RAPIDS_VERSION=23.06.0
rapids_jar=${rapids_jar:-rapids-4-spark_2.12-$SPARK_RAPIDS_VERSION.jar}
if [ ! -f $rapids_jar ]; then
    echo "downloading spark rapids jar"
    curl -L https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/${SPARK_RAPIDS_VERSION}/rapids-4-spark_2.12-${SPARK_RAPIDS_VERSION}-cuda${cuda_version}.jar \
    -o $rapids_jar
fi


spark_rapids_confs=$( 
cat <<EOF 
--spark_confs spark.executorEnv.PYTHONPATH=${rapids_jar} \
--spark_confs spark.sql.files.minPartitionNum=${num_gpus} \
--spark_confs spark.rapids.memory.gpu.minAllocFraction=0.0001 \
--spark_confs spark.plugins=com.nvidia.spark.SQLPlugin \
--spark_confs spark.locality.wait=0s \
--spark_confs spark.sql.cache.serializer=com.nvidia.spark.ParquetCachedBatchSerializer \
--spark_confs spark.rapids.memory.gpu.pooling.enabled=false \
--spark_confs spark.rapids.sql.explain=ALL \
--spark_confs spark.rapids.memory.gpu.reserve=20 \
--spark_confs spark.sql.execution.sortBeforeRepartition=false \
--spark_confs spark.rapids.sql.format.parquet.reader.type=MULTITHREADED \
--spark_confs spark.rapids.sql.format.parquet.multiThreadedRead.maxNumFilesParallel=20 \
--spark_confs spark.rapids.sql.multiThreadedRead.numThreads=20 \
--spark_confs spark.rapids.sql.python.gpu.enabled=true \
--spark_confs spark.rapids.memory.pinnedPool.size=2G \
--spark_confs spark.python.daemon.module=rapids.daemon \
--spark_confs spark.rapids.sql.batchSizeBytes=512m \
--spark_confs spark.sql.adaptive.enabled=false \
--spark_confs spark.sql.files.maxPartitionBytes=2000000000000 \
--spark_confs spark.rapids.sql.concurrentGpuTasks=2 \
--spark_confs spark.jars=${rapids_jar}
EOF
)
fi
    
# KMeans
if [[ "${MODE}" =~ "kmeans" ]] || [[ "${MODE}" == "all" ]]; then
    if [[ ! -d "${gen_data_root}/default/r${num_rows}_c${num_cols}_float32.parquet" ]]; then
        python $gen_data_script default \
            --num_rows $num_rows \
            --num_cols $num_cols \
            --output_num_files $output_num_files \
            --numPartitions $output_num_files \
            --dtype "float32" \
            --feature_type "array" \
            --output_dir "${gen_data_root}/default/r${num_rows}_c${num_cols}_float32.parquet" \
            $common_confs
           
    fi

    echo "$sep algo: kmeans $sep"
    python ./benchmark/benchmark_runner.py kmeans \
        --k 1000 \
        --tol 1.0e-20 \
        --maxIter 30 \
        --initMode random \
        --num_gpus $num_gpus \
        --num_cpus $num_cpus \
        --no_cache \
        --num_runs $num_runs \
        --train_path "${gen_data_root}/default/r${num_rows}_c${num_cols}_float32.parquet" \
        --report_path "report_kmeans_${cluster_type}.csv" \
        $common_confs $spark_rapids_confs \
        ${EXTRA_ARGS}
fi

# KNearestNeighbors
if [[ "${MODE}" =~ "knn" ]] || [[ "${MODE}" == "all" ]]; then
    if [[ ! -d "${gen_data_root}/blobs/r${knn_num_rows}_c${num_cols}_float32.parquet" ]]; then
        python $gen_data_script blobs \
            --num_rows $knn_num_rows \
            --num_cols $num_cols \
            --output_num_files $output_num_files \
            --dtype "float32" \
            --feature_type "array" \
            --output_dir "${gen_data_root}/blobs/r${knn_num_rows}_c${num_cols}_float32.parquet" \
            $common_confs
    fi

    echo "$sep algo: knn $sep"
    python ./benchmark/benchmark_runner.py knn \
        --n_neighbors 3 \
        --num_gpus $num_gpus \
        --num_cpus $num_cpus \
        --no_cache \
        --num_runs $num_runs \
        --train_path "${gen_data_root}/blobs/r${knn_num_rows}_c${num_cols}_float32.parquet" \
        --report_path "report_knn_${cluster_type}.csv" \
        $common_confs $spark_rapids_confs \
        ${EXTRA_ARGS}
fi

# Linear Regression
# TBD standardize datasets to allow better cpu to gpu training accuracy comparison:
# https://github.com/NVIDIA/spark-rapids-ml/blob/branch-23.08/python/src/spark_rapids_ml/regression.py#L519-L520
if [[ "${MODE}" =~ "linear_regression" ]] || [[ "${MODE}" == "all" ]]; then
    if [[ ! -d "${gen_data_root}/regression/r${num_rows}_c${num_cols}_float32.parquet" ]]; then
        python $gen_data_script regression \
            --num_rows $num_rows \
            --num_cols $num_cols \
            --output_num_files $output_num_files \
            --noise 10 \
            --dtype "float32" \
            --feature_type "array" \
            --output_dir "${gen_data_root}/regression/r${num_rows}_c${num_cols}_float32.parquet" \
            $common_confs
    fi

    echo "$sep algo: linear regression - no regularization $sep"
    python ./benchmark/benchmark_runner.py linear_regression \
        --regParam 0.0 \
        --elasticNetParam 0.0 \
        --standardization False \
        --num_gpus $num_gpus \
        --num_cpus $num_cpus \
        --num_runs $num_runs \
        --train_path "${gen_data_root}/regression/r${num_rows}_c${num_cols}_float32.parquet" \
        --transform_path "${gen_data_root}/regression/r${num_rows}_c${num_cols}_float32.parquet" \
        --report_path "report_linear_regression_noreg_${cluster_type}.csv" \
        $common_confs $spark_rapids_confs \
        ${EXTRA_ARGS}
    
    echo "$sep algo: linear regression - elasticnet regularization $sep"
    python ./benchmark/benchmark_runner.py linear_regression \
        --regParam 0.00001 \
        --elasticNetParam 0.5 \
        --tol 1.0e-30 \
        --maxIter 10 \
        --standardization False \
        --num_gpus $num_gpus \
        --num_cpus $num_cpus \
        --num_runs $num_runs \
        --train_path "${gen_data_root}/regression/r${num_rows}_c${num_cols}_float32.parquet" \
        --transform_path "${gen_data_root}/regression/r${num_rows}_c${num_cols}_float32.parquet" \
        --report_path "report_linear_regression_elastic_net_${cluster_type}.csv" \
        $common_confs $spark_rapids_confs \
        ${EXTRA_ARGS}
    
    echo "$sep algo: linear regression - ridge regularization $sep"
    python ./benchmark/benchmark_runner.py linear_regression \
        --regParam 0.00001 \
        --elasticNetParam 0.0 \
        --tol 1.0e-30 \
        --maxIter 10 \
        --standardization False \
        --num_gpus $num_gpus \
        --num_cpus $num_cpus \
        --num_runs $num_runs \
        --train_path "${gen_data_root}/regression/r${num_rows}_c${num_cols}_float32.parquet" \
        --transform_path "${gen_data_root}/regression/r${num_rows}_c${num_cols}_float32.parquet" \
        --report_path "report_linear_regression_ridge_${cluster_type}.csv" \
        $common_confs $spark_rapids_confs \
        ${EXTRA_ARGS}
fi

# PCA
if [[ "${MODE}" =~ "pca" ]] || [[ "${MODE}" == "all" ]]; then
    if [[ ! -d "${gen_data_root}/low_rank_matrix/r${num_rows}_c${num_cols}_float32.parquet" ]]; then
        python $gen_data_script low_rank_matrix \
            --num_rows $num_rows \
            --num_cols $num_cols \
            --output_num_files $output_num_files \
            --dtype "float32" \
            --feature_type "array" \
            --output_dir "${gen_data_root}/low_rank_matrix/r${num_rows}_c${num_cols}_float32.parquet" \
            $common_confs
    fi

    echo "$sep algo: pca $sep"
    python ./benchmark/benchmark_runner.py pca \
        --k 3 \
        --num_gpus $num_gpus \
        --num_cpus $num_cpus \
        --no_cache \
        --num_runs $num_runs \
        --train_path "${gen_data_root}/low_rank_matrix/r${num_rows}_c${num_cols}_float32.parquet" \
        --report_path "report_pca_${cluster_type}.csv" \
        $common_confs $spark_rapids_confs \
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
#        --parquet_path "${gen_data_root}/blobs/r${num_rows}_c${num_cols}_float32.parquet" \
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
if [[ "${MODE}" =~ "random_forest_classifier" ]] || [[ "${MODE}" == "all" ]]; then
    if [[ ! -d ${gen_data_root}/classification/r${num_rows}_c${num_cols}_float32.parquet ]]; then
        python $gen_data_script classification \
            --n_informative $( expr $num_cols / 3 )  \
            --n_redundant $( expr $num_cols / 3 ) \
            --num_rows $num_rows \
            --num_cols $num_cols \
            --output_num_files $output_num_files \
            --dtype "float32" \
            --feature_type "array" \
            --output_dir "${gen_data_root}/classification/r${num_rows}_c${num_cols}_float32.parquet" \
            $common_confs
    fi

    echo "$sep algo: random forest classification $sep"
    python ./benchmark/benchmark_runner.py random_forest_classifier \
        --numTrees 50 \
        --maxBins 128 \
        --maxDepth 13 \
        --num_gpus $num_gpus \
        --num_cpus $num_cpus \
        --num_runs $num_runs \
        --train_path "${gen_data_root}/classification/r${num_rows}_c${num_cols}_float32.parquet" \
        --transform_path "${gen_data_root}/classification/r${num_rows}_c${num_cols}_float32.parquet" \
        --report_path "report_rf_classifier_${cluster_type}.csv" \
        $common_confs $spark_rapids_confs \
        ${EXTRA_ARGS}
fi

# Random Forest Regression
if [[ "${MODE}" =~ "random_forest_regressor" ]] || [[ "${MODE}" == "all" ]]; then
    if [[ ! -d ${gen_data_root}/regression/r${num_rows}_c${num_cols}_float32.parquet ]]; then
        python $gen_data_script regression \
            --num_rows $num_rows \
            --num_cols $num_cols \
            --output_num_files $output_num_files \
            --dtype "float32" \
            --feature_type "array" \
            --output_dir "${gen_data_root}/regression/r${num_rows}_c${num_cols}_float32.parquet" \
            $common_confs
    fi

    echo "$sep algo: random forest regression $sep"
    python ./benchmark/benchmark_runner.py random_forest_regressor \
        --numTrees 30 \
        --maxBins 128 \
        --maxDepth 6 \
        --num_gpus $num_gpus \
        --num_cpus $num_cpus \
        --num_runs $num_runs \
        --train_path "${gen_data_root}/regression/r${num_rows}_c${num_cols}_float32.parquet" \
        --transform_path "${gen_data_root}/regression/r${num_rows}_c${num_cols}_float32.parquet" \
        --report_path "report_rf_regressor_${cluster_type}.csv" \
        $common_confs $spark_rapids_confs \
        ${EXTRA_ARGS}
fi
