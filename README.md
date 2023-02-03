# Spark Rapids ML
Spark Rapids ML is a python package enabling GPU accelerated distributed machine learning on [Apache Spark](https://spark.apache.org/).  It provides a pySpark ML compatible API and is powered by the GPU-accelerated [RAPIDS cuML](https://docs.rapids.ai/api/cuml/stable/) library.

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
TBD

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
# sphinx-apidoc -f -o docs/source src/spark_rapids_ml
cd docs
make html
```

### Preview docs
```
cd docs/build/html
python -m http.server 8080
```
