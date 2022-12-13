# SparkCuml

SparkCuml is a project to make cuml python package distibuted run on spark.

## Run tests

We strongly suggest installing the python dependencies in conda environment

``` bash
conda create -n rapids-22.10 -c rapidsai -c nvidia -c conda-forge \
    cuml=22.10 python=3.9 cudatoolkit=11.5
```

You can choose the latest version from [rapids.ai](https://rapids.ai/start.html#get-rapids).

Once you have the `rapids-22.10`, you still need to install below packages.

``` bash
conda activate rapids-22.10
pip install pylint pytest pyspark black mypy scikit-learn
```

Run test

``` bash
./run_test.sh
```

Run benchmark

``` bash
./run_benchmark.sh
```


## To run GPU-accelerated pyspark PCA (multi-node multi-gpu)
```bash
SPARK_MASTER=spark://hostname:port
PYTHON_ENV_PATH=~/miniconda3/envs/cuspark/bin/python

${SPARK_HOME}/bin/spark-submit --master ${SPARK_MASTER} \
  --conf spark.task.resource.gpu.amount=1 \
  --conf spark.executor.resource.gpu.amount=2 \
  --conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
  --conf spark.files=${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh \
  --conf spark.pyspark.python=${PYTHON_ENV_PATH} \
  cuspark_pca.py
```

## To run GPU-accelerated pyspark Kmeans (single-gpu) 
```bash
    python cuspark_kmeans.py
```

## To reproduce a bug 
```bash
    python bug.py
```

## Contact
- [Jinfeng Li](jinfengl@nvidia.com) 
- [Bobby Wang](bobwang@nvidia.com)
- [Erik Ordentlich](eordentlich@nvidia.com) 
