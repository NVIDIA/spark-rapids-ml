# Spark Rapids ML

Spark Rapids ML enables GPU accelerated distributed machine learning on [Apache Spark](https://spark.apache.org/).  It provides several PySpark ML compatible algorithms powered by the [RAPIDS cuML](https://docs.rapids.ai/api/cuml/stable/) library.

These APIs seek to minimize any code changes to end user Spark code.  After your environment is configured to support GPUs (with drivers, CUDA toolkit, and RAPIDS dependencies), you should be able to just change an import statement or class name to take advantage of GPU acceleration.   See [here](./python/README.md#clis-enabling-no-package-import-change) for experimental CLIs that enable GPU acceleration without the need for changing the `pyspark.ml` package names in an existing pyspark ml application.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/spark-rapids-ml)

**Python**
```python
# from pyspark.ml.feature import PCA
from spark_rapids_ml.feature import PCA

pca = (
    PCA()
    .setK(3)
    .setInputCol("features")
    .setOutputCol("pca_features")
)
pca.fit(df)
```

## Supported Algorithms

The following table shows the currently supported algorithms.  The goal is to expand this over time with support from the underlying RAPIDS cuML libraries.  If you would like support for a specific algorithm, please file a [git issue](https://github.com/NVIDIA/spark-rapids-ml/issues) to help us prioritize.

| Supported Algorithms   | Python |
| :--------------------- | :----: |
| CrossValidator         |   √    |
| DBSCAN (*)             |   √    |
| KMeans                 |   √    |
| approx/exact k-NN (*)  |   √    |
| LinearRegression       |   √    |
| LogisticRegression     |   √    |
| PCA                    |   √    |
| RandomForestClassifier |   √    |
| RandomForestRegressor  |   √    |
| UMAP (*)               |   √    |

(*) Notes: 
- As an alternative to KMeans, we also provide a Spark API for GPU accelerated Density-Based Spatial Clustering of Applications with Noise (DBSCAN), a density based clustering algorithm in the RAPIDS cuML library.
- Spark does not provide a k-Nearest Neighbors (k-NN) implementation, but it does have an [LSH-based Approximate Nearest Neighbor](https://spark.apache.org/docs/latest/ml-features.html#approximate-nearest-neighbor-search) implementation. 
- As an alternative to PCA, we also provide a Spark API for GPU accelerated Uniform Manifold Approximation and Projection (UMAP), a non-linear dimensionality reduction algorithm in the RAPIDS cuML library. 

## Getting started

For PySpark (Python) users, see [this guide](python/README.md).

## Performance

GPU acceleration can provide significant performance and cost benefits.  Benchmarking instructions and results can be found [here](python/benchmark/README.md).

## Contributing

We welcome community contributions!  Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) to get started.