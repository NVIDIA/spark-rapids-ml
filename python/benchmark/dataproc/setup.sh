#!/bin/bash -xe

# copies files to GCS bucket

if [[ -z $BENCHMARK_HOME ]]; then
    echo "please export BENCHMARK_HOME per README.md"
    exit 1
fi

SPARK_RAPIDS_ML_HOME='../..'

echo "**** copying benchmarking related files to ${BENCHMARK_HOME} ****"

gsutil cp init_benchmark.sh gs://${BENCHMARK_HOME}/init_benchmark.sh
curl -LO https://raw.githubusercontent.com/GoogleCloudDataproc/initialization-actions/master/spark-rapids/spark-rapids.sh
gsutil cp spark-rapids.sh gs://${BENCHMARK_HOME}/spark-rapids.sh

pushd ${SPARK_RAPIDS_ML_HOME}/benchmark
zip -r - benchmark >benchmark.zip
gsutil cp benchmark.zip gs://${BENCHMARK_HOME}/benchmark.zip
popd

pushd ${SPARK_RAPIDS_ML_HOME}
gsutil cp benchmark/benchmark_runner.py gs://${BENCHMARK_HOME}/benchmark_runner.py
popd

pushd ${SPARK_RAPIDS_ML_HOME}/src
zip -r - spark_rapids_ml >spark_rapids_ml.zip
gsutil cp spark_rapids_ml.zip gs://${BENCHMARK_HOME}/spark_rapids_ml.zip
popd
