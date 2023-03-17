#!/bin/bash -xe

# copies files to dbfs 

if [[ -z $DB_PROFILE ]]; then
    echo "please export DB_PROFILE per README.md"
    exit 1
fi

if [[ -z $BENCHMARK_HOME ]]; then
    echo "please export BENCHMARK_HOME_ROOT per README.md"
    exit 1
fi

#BENCHMARK_HOME_ROOT=/${USER}/spark-rapids-ml/benchmarking
SPARK_RAPIDS_ML_HOME='../..'

echo "**** copying benchmarking related files to ${BENCHMARK_HOME} ****"

INIT_SCRIPT_DIR="${BENCHMARK_HOME}/init_scripts"
SPARK_RAPIDS_ML_ZIP="${BENCHMARK_HOME}/zips/spark-rapids-ml.zip"
BENCHMARK_ZIP="${BENCHMARK_HOME}/zips/benchmark.zip"
BENCHMARK_SCRIPTS="${BENCHMARK_HOME}/scripts"

pushd ${SPARK_RAPIDS_ML_HOME}/benchmark && rm -f benchmark.zip && \
zip -r benchmark.zip benchmark && \
databricks fs cp benchmark.zip dbfs:${BENCHMARK_ZIP} --profile ${DB_PROFILE} ${DB_OVERWRITE} && \
popd

pushd ${SPARK_RAPIDS_ML_HOME} && \
ls benchmark
databricks fs cp benchmark/benchmark_runner.py dbfs:${BENCHMARK_SCRIPTS}/benchmark_runner.py --profile  ${DB_PROFILE} ${DB_OVERWRITE} && \
popd

pushd ${SPARK_RAPIDS_ML_HOME}/src && rm -f spark-rapids-ml.zip && \
zip -r spark-rapids-ml.zip spark_rapids_ml && \
databricks fs cp spark-rapids-ml.zip dbfs:${SPARK_RAPIDS_ML_ZIP} --profile ${DB_PROFILE} ${DB_OVERWRITE} && \
popd

# point cpu and gpu cluster init scripts to new files and upload
for init_script in init-pip-cuda-11.8.sh init-cpu.sh
do
# NOTE: on linux delete the .bu after -i
    sed -e "s#/path/to/spark-rapids-ml\.zip#${SPARK_RAPIDS_ML_ZIP}#g" -e "s#/path/to/benchmark\.zip#${BENCHMARK_ZIP}#g" $init_script > ${init_script}.updated && \
    databricks fs cp ${init_script}.updated dbfs:${BENCHMARK_HOME}/init_script/${init_script} --profile ${DB_PROFILE} ${DB_OVERWRITE}
done











