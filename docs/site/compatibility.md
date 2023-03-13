---
title: Compatibility
nav_order: 3
---
# Compatibility with Apache Spark

## Supported Algorithms

The following table shows the currently supported algorithms.  The goal is to expand this over time with support from the underlying RAPIDS cuML libraries.  If you would like support for a specific algorithm, please file a [git issue](https://github.com/NVIDIA/spark-rapids-ml/issues) to help us prioritize.

| Spark ML Algorithm     | Python | Scala |
| :--------------------- | :----: | :---: |
| K-Means                |   √    |       |
| k-NN (*)               |   √    |       |
| LinearRegression       |   √    |       |
| PCA                    |   √    |   √   |
| RandomForestClassifier |   √    |       |
| RandomForestRegressor  |   √    |       |

Note: Spark does not provide a k-NN implementation, but it does have an [LSH-based Approximate Nearest Neighbor](https://spark.apache.org/docs/latest/ml-features.html#approximate-nearest-neighbor-search) implementation.

## Supported Versions

| Spark Rapids ML | CUDA  | Spark  | Python |
| :-------------- | :---- | :----- | :----- |
| 1.0.0           | 11.5+ | 3.2.1+ | 3.8+   |
