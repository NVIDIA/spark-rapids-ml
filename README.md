# SparkCuML
SparkCuML is a project to make the GPU-accelerated [RAPIDS cuML](https://docs.rapids.ai/api/cuml/stable/) python package run distibuted on [Apache Spark](https://spark.apache.org/).

## Installation
We strongly suggest installing the python dependencies in conda environment
```bash
conda create -n rapids-22.10 -c rapidsai -c nvidia -c conda-forge \
    cuml=22.10 python=3.9 cudatoolkit=11.5
```

You can choose the latest version from [rapids.ai](https://rapids.ai/start.html#get-rapids).

Once you have the `rapids-22.10` conda environment, install the required packages.
```bash
conda activate rapids-22.10
pip install -r requirements.txt
```
## Usage
### Run PCA (multi-node multi-gpu)
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

### Run K-Means (single-gpu)
```bash
python cuspark_kmeans.py
```

## Development
### Run tests
```bash
pip install -e .
./run_test.sh --runslow
```

### Run benchmarks
```bash
pip install -e .
./run_benchmark.sh
```

### Build package
```bash
python -m build
```

### Build docs
```bash
# sphinx-apidoc -f -o docs/source src/sparkcuml
cd docs
make html
```

### Preview docs
```
cd docs/build/html
python -m http.server 8080
```
