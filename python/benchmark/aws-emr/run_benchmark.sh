#!/bin/bash
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

cluster_type=${1:-gpu}
BENCHMARK_DATA_HOME="s3://spark-rapids-ml-bm-datasets-public"

if [[ -z ${BENCHMARK_HOME} ]]; then
    echo "Please export BENCHMARK_HOME per README.md"
    exit 1
fi

gpu_submit_args=$(cat <<EOF
--deploy-mode,client,\
--master,yarn,\
--conf,spark.yarn.submit.waitAppCompletion=true,\
--conf,spark.sql.files.minPartitionNum=2,\
--conf,spark.sql.execution.arrow.maxRecordsPerBatch=10000,\
--conf,spark.executor.cores=8,\
--conf,spark.executor.memory=5g,\
--conf,spark.sql.files.maxPartitionBytes=2000000000000,\
--conf,spark.driver.memory=5g,\
--py-files,"s3://${BENCHMARK_HOME}/spark_rapids_ml.zip,s3://${BENCHMARK_HOME}/benchmark.zip",\
s3://${BENCHMARK_HOME}/benchmark_runner.py
EOF
)

cpu_submit_args=$(cat <<EOF
--deploy-mode,client,\
--master,yarn,\
--conf,spark.yarn.submit.waitAppCompletion=true,\
--conf,spark.locality.wait=0s,\
--conf,spark.sql.execution.arrow.pyspark.enabled=true,\
--conf,spark.sql.execution.arrow.maxRecordsPerBatch=10000,\
--conf,spark.sql.execution.sortBeforeRepartition=false,\
--conf,spark.executor.cores=8,\
--conf,spark.driver.memory=10g,\
--conf,spark.executor.memory=20g,\
--py-files,"s3://${BENCHMARK_HOME}/spark_rapids_ml.zip,s3://${BENCHMARK_HOME}/benchmark.zip",\
s3://${BENCHMARK_HOME}/benchmark_runner.py
EOF
)

gpu_args=$(cat << EOF
--num_cpus,0,\
--num_gpus,2
EOF
)

cpu_args=$(cat << EOF
--num_cpus,16,\
--num_gpu,0
EOF
)

rf_gpu_extra_args=$(cat << EOF
--n_streams,4
EOF
)

rf_cpu_extra_args=$(cat << EOF
--subsamplingRate,0.5
EOF
)

num_runs=1
if [[ ${cluster_type} == "gpu" ]]; then
    spark_submit_args=${gpu_submit_args}
    extra_args=${gpu_args}
    kmeans_runs=1 #3
    rf_runs=1 #3
    rf_extra_args=${rf_gpu_extra_args}
    device="GPU"
elif [[ ${cluster_type} == "cpu" ]]; then
    spark_submit_args=${cpu_submit_args}
    # kmeans and random forest take a long time on cpu cluster, so only do 1 run each
    extra_args=${cpu_args}
    kmeans_runs=1
    rf_runs=1
    rf_extra_args=${rf_cpu_extra_args}
    device="CPU"
else
    echo "unknown cluster type ${cluster_type}"
    echo "usage: $0 cpu|gpu"
    exit 1
fi

# start benchmark cluster
CLUSTER_ID=$(./start_cluster.sh ${cluster_type})
if [[ $? != 0 ]]; then
    echo "Failed to start cluster."
    exit 1
fi

ssh_command () {
    aws emr wait cluster-running --cluster-id $CLUSTER_ID
    if [[ $? != 0 ]]; then
        echo "cluster terminated, exiting"
        exit 1
    fi
    ssh -i $KEYPAIR -o StrictHostKeyChecking=no ec2-user@$masternode $1
}

get_masternode () {
    aws emr list-instances --cluster-id $CLUSTER_ID --instance-group-type MASTER | grep PublicDnsName | grep -oP 'ec2[^"]*'
}

get_appid () {
    ssh_command "hdfs dfs -text $stderr_path" | grep -oP "application_[0-9]*_[0-9]*" | head -n 1
}

get_appstatus () {
    ssh_command "yarn application -status $app_id" | grep -P "\tState :" | grep -oP FINISHED
}

poll_stdout () {
    stdout_path=s3://${BENCHMARK_HOME}/logs/$1/steps/$2/stdout.gz
    stderr_path=s3://${BENCHMARK_HOME}/logs/$1/steps/$2/stderr.gz
    masternode=$( get_masternode )

    while [[ -z $masternode ]]; do
    sleep 30
    masternode=$( get_masternode )
    done

    echo masternode: $masternode
    app_id=""

    app_id=$( get_appid )
    echo app_id: $app_id
    while [[ -z $app_id ]]
    do
        sleep 30
        app_id=$( get_appid )
        echo app_id: $app_id
    done

    res=$( get_appstatus )
    echo res: $res
    while [[ ${res} != FINISHED ]]
    do
        sleep 30
        res=$( get_appstatus )
        echo res: ${res}
    done

    aws emr cancel-steps --cluster-id $1 --step-ids $2 --step-cancellation-option SEND_INTERRUPT

    res=$( ssh_command "yarn application -status $app_id" | grep -P "\tFinal-State :" | sed -e 's/.*: *//g' )

    if [[ $res != SUCCEEDED ]]; then
        echo "benchmark step failed"
        exit 1
    fi

    # check if EMR stdout.gz is complete
    res=""
    while [[ ${res} != *"datetime"* ]]
    do
        sleep 30
        aws s3 cp ${stdout_path} ./.stdout.gz
        res=$(gunzip -c .stdout.gz)
    done
    gunzip -c .stdout.gz | tee $3
    rm .stdout.gz
}

# run benchmarks
sep="=================="

echo
echo "$sep algo: kmeans $sep"
kmeans_args=$(cat << EOF
${spark_submit_args},\
kmeans,\
${extra_args},\
--num_runs,1,\
--k,1000,\
--tol,1.0e-20,\
--maxIter,30,\
--initMode,random,\
--no_cache,\
--train_path,${BENCHMARK_DATA_HOME}/pca/1m_3k_singlecol_float32_50_files.parquet
EOF
)

for i in `seq $kmeans_runs`; do
    set -x
    STEP_ID=$(aws emr add-steps --cluster-id ${CLUSTER_ID} \
        --steps Type=Spark,Name="${device} KMeans",ActionOnFailure=CONTINUE,Args=[${kmeans_args}] | tee /dev/tty | grep -o 's-[0-9|A-Z]*')
    set +x
    poll_stdout $CLUSTER_ID $STEP_ID ./kmeans_$i.out
done

echo
echo "$sep algo: pca $sep"
pca_args=$(cat << EOF
${spark_submit_args},\
pca,\
${extra_args},\
--num_runs,1,\
--k,3,\
--no_cache,\
--train_path,${BENCHMARK_DATA_HOME}/low_rank_matrix/1m_3k_singlecol_float32_50_files.parquet
EOF
)
for i in `seq $num_runs`; do
    set -x
    STEP_ID=$(aws emr add-steps --cluster-id ${CLUSTER_ID} \
        --steps Type=Spark,Name="${device} PCA",ActionOnFailure=CONTINUE,Args=[${pca_args}] | tee /dev/tty | grep -o 's-[0-9|A-Z]*')
    set +x
    poll_stdout $CLUSTER_ID $STEP_ID ./pca_$i.out
done

echo
echo "$sep algo: linear regression - no regularization $sep"
linear_regression_args=$(cat << EOF
${spark_submit_args},\
linear_regression,\
${extra_args},\
--num_runs,1,\
--regParam,0.0,\
--elasticNetParam,0.0,\
--standardization,False,\
--train_path,"${BENCHMARK_DATA_HOME}/linear_regression/1m_3k_standardized_singlecol_float32_50_files.parquet"
EOF
)
for i in `seq $num_runs`; do
    set -x
    STEP_ID=$(aws emr add-steps --cluster-id ${CLUSTER_ID} \
        --steps Type=Spark,Name="${device} Linear Regression",ActionOnFailure=CONTINUE,Args=[${linear_regression_args}] | tee /dev/tty | grep -o 's-[0-9|A-Z]*')
    set +x
    poll_stdout $CLUSTER_ID $STEP_ID ./linear_regression_noreg_$i.out
done

echo
echo "$sep algo: linear regression - elasticnet regularization $sep"
elasticnet_args=$(cat << EOF
${spark_submit_args},\
linear_regression,\
${extra_args},\
--num_runs,1,\
--regParam,0.00001,\
--elasticNetParam,0.5,\
--tol,1.0e-30,\
--maxIter,10,\
--standardization,False,\
--train_path,"${BENCHMARK_DATA_HOME}/linear_regression/1m_3k_standardized_singlecol_float32_50_files.parquet"
EOF
)
for i in `seq $num_runs`; do
    set -x
    STEP_ID=$(aws emr add-steps --cluster-id ${CLUSTER_ID} \
        --steps Type=Spark,Name="${device} Linear Regression - Elasticnet",ActionOnFailure=CONTINUE,Args=[${elasticnet_args}] | tee /dev/tty | grep -o 's-[0-9|A-Z]*')
    set +x
    poll_stdout $CLUSTER_ID $STEP_ID ./linear_regression_elasticnet_$i.out
done

echo
echo "$sep algo: linear regression - ridge regularization $sep"
ridge_args=$(cat << EOF
${spark_submit_args},\
linear_regression,\
${extra_args},\
--num_runs,1,\
--regParam,0.00001,\
--elasticNetParam,0.0,\
--tol,1.0e-30,\
--maxIter,10,\
--standardization,False,\
--train_path,"${BENCHMARK_DATA_HOME}/linear_regression/1m_3k_standardized_singlecol_float32_50_files.parquet"
EOF
)
for i in `seq $num_runs`; do
    set -x
    STEP_ID=$(aws emr add-steps --cluster-id ${CLUSTER_ID} \
        --steps Type=Spark,Name="${device} Linear Regression - Ridge",ActionOnFailure=CONTINUE,Args=[${ridge_args}] | tee /dev/tty | grep -o 's-[0-9|A-Z]*')
    set +x
    poll_stdout $CLUSTER_ID $STEP_ID ./linear_regression_ridge_$i.out
done

echo ${extra_args}
echo ${rf_extra_args}
echo
echo "$sep algo: random forest classification $sep"
rf_classification_args=$(cat << EOF
${spark_submit_args},\
random_forest_classifier,\
${extra_args},\
${rf_extra_args},\
--num_runs,1,\
--numTrees,50,\
--maxBins,128,\
--maxDepth,13,\
--train_path,"${BENCHMARK_DATA_HOME}/classification/1m_3k_singlecol_float32_50_1_3_inf_red_files.parquet"
EOF
)
for i in `seq $rf_runs`; do
    set -x
    STEP_ID=$(aws emr add-steps --cluster-id ${CLUSTER_ID} \
        --steps Type=Spark,Name="${device} Random Forest Classification",ActionOnFailure=CONTINUE,Args=[${rf_classification_args}] | tee /dev/tty | grep -o 's-[0-9|A-Z]*')
    set +x
    poll_stdout $CLUSTER_ID $STEP_ID ./random_forest_classifier_$i.out
done

echo
echo "$sep algo: random forest regression $sep"
rf_regressor_args=$(cat << EOF
${spark_submit_args},\
random_forest_regressor,\
${extra_args},\
${rf_extra_args},\
--num_runs,1,\
--numTrees,30,\
--maxBins,128,\
--maxDepth,6,\
--train_path,"${BENCHMARK_DATA_HOME}/linear_regression/1m_3k_singlecol_float32_50_files.parquet"
EOF
)
for i in `seq $rf_runs`; do
    set -x
    STEP_ID=$(aws emr add-steps --cluster-id ${CLUSTER_ID} \
        --steps Type=Spark,Name="${device} Random Forest Regressor",ActionOnFailure=CONTINUE,Args=[${rf_regressor_args}] | tee /dev/tty | grep -o 's-[0-9|A-Z]*')
    set +x
    poll_stdout $CLUSTER_ID $STEP_ID ./random_forest_regressor_$i.out
done

echo
echo "$sep algo: logistic regression $sep"
lr_classification_args=$(cat << EOF
${spark_submit_args},\
logistic_regression,\
${extra_args},\
--num_runs,1,\
--standardization,False,\
--maxIter,200,\
--tol,1e-30,\
--regParam,0.00001,\
--train_path,"${BENCHMARK_DATA_HOME}/classification/1m_3k_singlecol_float32_50_1_3_inf_red_files.parquet"
EOF
)
for i in `seq $num_runs`; do
    set -x
    STEP_ID=$(aws emr add-steps --cluster-id ${CLUSTER_ID} \
        --steps Type=Spark,Name="${device} Logistic Regression",ActionOnFailure=CONTINUE,Args=[${lr_classification_args}] | tee /dev/tty | grep -o 's-[0-9|A-Z]*')
    set +x
    poll_stdout $CLUSTER_ID $STEP_ID ./logistic_regression_$i.out
done
