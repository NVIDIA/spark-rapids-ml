# Spark Rapids ML (Python)

This PySpark-compatible API leverages the RAPIDS cuML python API to provide GPU-accelerated implementations of many common ML algorithms.  These implementations are adapted to use PySpark for distributed training and inference.

## Installation

For simplicity, the following instructions just use Spark local mode, assuming a server with at least one GPU.

First, install RAPIDS cuML per [these instructions](https://rapids.ai/start.html).
```bash
conda create -n rapids-23.06 \
    -c rapidsai -c nvidia -c conda-forge \
    cuml=23.06 python=3.9 cudatoolkit=11.5
```

**Note**: while testing, we recommend using conda or docker to simplify installation and isolate your environment while experimenting.  Once you have a working environment, you can then try installing directly, if necessary.

**Note**: you can select the latest version compatible with your environment from [rapids.ai](https://rapids.ai/start.html#get-rapids).

Once you have the conda environment, activate it and install the required packages.
```bash
conda activate rapids-23.06

# for development access to notebooks, tests, and benchmarks
git clone --branch main https://github.com/NVIDIA/spark-rapids-ml.git
cd spark-rapids-ml/python
pip install -r requirements.txt
pip install -e .

# OPTIONAL: for package installation only
pip install -r https://raw.githubusercontent.com/NVIDIA/spark-rapids-ml/main/python/requirements.txt
pip install spark-rapids-ml
```

## Examples

These examples demonstrate the API using toy datasets.  However, GPUs are more effective when using larger datasets that require extensive compute.  So once you are confident in your environment setup, use a more representative dataset for your specific use case to gauge how GPUs can improve performance.

### PySpark shell

#### Linear Regression
```python
## pyspark --master local[*]
# from pyspark.ml.regression import LinearRegression
from spark_rapids_ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors

df = spark.createDataFrame([
     (1.0, 2.0, Vectors.dense(1.0, 0.0)),
     (0.0, 2.0, Vectors.dense(0.0, 1.0))], ["label", "weight", "features"])

# number of partitions should match number of GPUs in Spark cluster
df = df.repartition(1)

lr = LinearRegression(regParam=0.0, solver="normal")
lr.setMaxIter(5)
lr.setRegParam(0.0)
lr.setFeaturesCol("features")
lr.setLabelCol("label")

model = lr.fit(df)

model.coefficients
# DenseVector([0.5, -0.5])
```

#### K-Means
```python
## pyspark --master local[*]
# from pyspark.ml.clustering import KMeans
from spark_rapids_ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
data = [(Vectors.dense([0.0, 0.0]), 2.0), (Vectors.dense([1.0, 1.0]), 2.0),
        (Vectors.dense([9.0, 8.0]), 2.0), (Vectors.dense([8.0, 9.0]), 2.0)]
df = spark.createDataFrame(data, ["features", "weighCol"])

# number of partitions should match number of GPUs in Spark cluster
df = df.repartition(1)

kmeans = KMeans(k=2)
kmeans.setSeed(1)
kmeans.setMaxIter(20)
kmeans.setFeaturesCol("features")

model = kmeans.fit(df)

centers = model.clusterCenters()
print(centers)
# [array([0.5, 0.5]), array([8.5, 8.5])]

model.setPredictionCol("newPrediction")
transformed = model.transform(df)
transformed.show()
# +--------+----------+-------------+
# |weighCol|  features|newPrediction|
# +--------+----------+-------------+
# |     2.0|[0.0, 0.0]|            1|
# |     2.0|[1.0, 1.0]|            1|
# |     2.0|[9.0, 8.0]|            0|
# |     2.0|[8.0, 9.0]|            0|
# +--------+----------+-------------+
rows[0].newPrediction == rows[1].newPrediction
# True
rows[2].newPrediction == rows[3].newPrediction
# True
```

#### PCA
```python
## pyspark --master local[*]
# from pyspark.ml.feature import PCA
from spark_rapids_ml.feature import PCA

data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
        (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
        (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
df = spark.createDataFrame(data,["features"])

# number of partitions should match number of GPUs in Spark cluster
df = df.repartition(1)

pca = PCA(k=2, inputCol="features")
pca.setOutputCol("pca_features")

model = pca.fit(df)

model.setOutputCol("output")
model.transform(df).collect()[0].output
# [-1.6485728230896184, -4.013282697765595]
model.explainedVariance
# DenseVector([0.7944, 0.2056])
model.pc
# DenseMatrix(5, 2, [0.4486, -0.133, 0.1252, -0.2165, 0.8477, -0.2842, -0.0562, 0.7636, -0.5653, -0.1156], False)
```

### Jupyter Notebooks
To run the example notebooks locally, see [these instructions](../notebooks/README.md).

To run the example notebooks in Databricks (assuming you already have a Databricks account), follow [these instructions](../notebooks/databricks/README.md).

## API Compatibility

While the Spark Rapids ML API attempts to mirror the PySpark ML API to minimize end-user code changes, the underlying implementations are entirely different, so there are some differences.
- **Unsupported ML Params** - some PySpark ML algorithms have ML Params which do not map directly to their respective cuML implementations.  For these cases, the ML Param default values will be ignored, and if explicitly set by end-user code:
    - a warning will be printed (for non-critical cases that should have minimal impact, e.g. `initSteps`).
    - an exception will be raised (for critical cases that can greatly affect results, e.g. `weightCol`).
- **Unsupported methods** - some PySpark ML methods may not map to the underlying cuML implementations, or may not be meaningful for GPUs.  In these cases, an error will be raised if the method is invoked.
- **cuML parameters** - there may be additional cuML-specific parameters which might be useful for optimizing GPU performance.  These can be supplied to the various class constructors, but they are _not_ exposed via getters and setters to avoid any confusion with the PySpark ML Params.  If needed, they can be observed via the `cuml_params` attribute.
- **Algorithmic Results** - again, since the GPU implementations are entirely different from their PySpark ML CPU counterparts, there may be slight differences in the produced results.  This can be due to various reasons, including different optimizations, randomized initializations, or algorithm design.  While these differences should be relatively minor, they should still be reviewed in the context of your specific use case to see if they are within acceptable limits.

**Example**
```python
# from pyspark.ml.clustering import KMeans
from spark_rapids_ml.clustering import KMeans
from pyspark.ml.linalg import Vectors

data = [(Vectors.dense([0.0, 0.0]), 2.0), (Vectors.dense([1.0, 1.0]), 2.0),
        (Vectors.dense([9.0, 8.0]), 2.0), (Vectors.dense([8.0, 9.0]), 2.0)]
df = spark.createDataFrame(data, ["features", "weighCol"]).repartition(1)

# `k` is a Spark ML Param, `max_samples_per_batch` is a cuML parameter
kmeans = KMeans(k=3, max_samples_per_batch=16384)
kmeans.setK(2)  # setter is available for `k`, but not for `max_samples_per_batch`
kmeans.setInitSteps(10)  # non-critical unsupported param, prints a warning
# kmeans.setWeightCol("weight")  # critical unsupported param, raises an error

# show cuML-specific parameters
print(kmeans.cuml_params)
# {'n_clusters': 2, 'max_iter': 20, 'tol': 0.0001, 'verbose': False, 'random_state': 1909113551, 'init': 'scalable-k-means++', 'n_init': 1, 'oversampling_factor': 2.0, 'max_samples_per_batch': 16384}

model = kmeans.fit(df)

sample = df.head().features  # single example
# unsupported method, raises an error, since not optimal use-case for GPUs
# model.predict(sample)

centers = model.clusterCenters()
print(centers)  # slightly different results
# [[8.5, 8.5], [0.5, 0.5]]
# PySpark: [array([0.5, 0.5]), array([8.5, 8.5])]
```

## API Documentation

- [Spark Rapids ML](https://nvidia.github.io/spark-rapids-ml/)
- [PySpark ML](https://spark.apache.org/docs/latest/api/python/reference/pyspark.ml.html)
- [RAPIDS cuML](https://docs.rapids.ai/api/cuml/stable/api.html)
