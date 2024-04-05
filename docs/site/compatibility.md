---
title: Compatibility
nav_order: 3
---
# Compatibility with Apache Spark

## Supported Algorithms

The following table shows the currently supported algorithms.  The goal is to expand this over time with support from the underlying RAPIDS cuML libraries.  If you would like support for a specific algorithm, please file a [git issue](https://github.com/NVIDIA/spark-rapids-ml/issues) to help us prioritize.

| Supported Algorithms   | Python | Scala |
| :--------------------- | :----: | :---: |
| DBSCAN (*)             |   √    |       |
| CrossValidator         |   √    |       |
| KMeans                 |   √    |       |
| k-NN (*)               |   √    |       |
| LinearRegression       |   √    |       |
| LogisticRegression     |   √    |       | 
| PCA                    |   √    |   √   |
| RandomForestClassifier |   √    |       |
| RandomForestRegressor  |   √    |       |
| UMAP (*)               |   √    |       |

Note: Though they have no direct counterparts in Spark MLlib, we also provide Spark APIs for RAPIDS cuML's distributed GPU accelerated implementations of: Density-Based Spatial Clustering of Applications with Noise (DBSCAN), Exact k-Nearest Neighbors (k-NN), and Uniform Manifold Approximation and Projection (UMAP - a non-linear dimensionality reduction algorithm).  The closest counterpart to k-NN in Spark MLlib is [LSH-based Approximate Nearest Neighbor](https://spark.apache.org/docs/latest/ml-features.html#approximate-nearest-neighbor-search) implementation.


## Supported Versions

| Spark Rapids ML | CUDA  | Spark  | Python |
| :-------------- | :---- | :----- | :----- |
| 1.0.0           | 11.5+ | 3.2.1+ | 3.9+   |


## Single vs Double precision inputs
The underlying cuML implementations all accept single precision (e.g. Float or float32) input types and offer the best performance in this case.  As a result, by default, Spark RAPIDs ML converts Spark DataFrames supplied to `fit` and `transform` methods having double precision data types (i.e. `VectorUDT`, `ArrayType(DoubleType())`, `DoubleType()` columns) to single precision before passing them down to the cuML layer.  Most of the cuML algorithm implementations also support double precision inputs.   The Estimator (for all algorithms) constructor parameter `float32_inputs` can be used to control this behavior.  The default value is `True` which forces the conversion to single precision for all algorithms, but it can be set to `False` in which case double precision input data is passed to those cuML algorithms which support it.

Currently all algorithms *except* the following support double precision:  LogisticRegression, k-NN, UMAP.
