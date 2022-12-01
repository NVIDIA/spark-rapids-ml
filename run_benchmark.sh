#! /bin/bash
# local mode
unset SPARK_HOME
SPARKCUML_HOME=`pwd`
export PYTHONPATH="$SPARKCUML_HOME:$PYTHONPATH"

CUDA_VISIBLE_DEICES=0,1 python ./benchmark/bench_pca.py \
    --num_vecs 5000 \
    --dim 3000 \
    --n_components 3 \
    --num_gpus 2 \
    --num_cpus 0 \
    --dtype "float64" \
    --num_runs 3 \
    --report_path "./report.csv" \
    --spark_confs "spark.master=local[12]" \
    --spark_confs "spark.driver.memory=128g" \
    --spark_confs "spark.sql.execution.arrow.maxRecordsPerBatch=200000" 

### standalone mode
#SPARK_MASTER=spark://hostname:port
#unset SPARK_HOME
#SPARKCUML_HOME=`pwd`
#export PYTHONPATH="$SPARKCUML_HOME:$PYTHONPATH"
#
#python ./benchmark/bench_pca.py \
#    --num_vecs 5000 \
#    --dim 3000 \
#    --n_components 3 \
#    --num_gpus 2 \
#    --num_cpus 0 \
#    --dtype "float64" \
#    --num_runs 3 \
#    --report_path "./report_standalone.csv" \
#    --spark_confs "spark.master=${SPARK_MASTER}" \
#    --spark_confs "spark.driver.memory=128g" \
#    --spark_confs "spark.sql.execution.arrow.maxRecordsPerBatch=200000"  \
#    --spark_confs "spark.executor.memory=128g" \
#    --spark_confs "spark.rpc.message.maxSize=2000" \
#    --spark_confs "spark.pyspark.python=${PYTHON_ENV_PATH}" \
#    --spark_confs "spark.submit.pyFiles=./sparkcuml.tar.gz" \
#    --spark_confs "spark.task.resource.gpu.amount=1" \
#    --spark_confs "spark.executor.resource.gpu.amount=1" 
