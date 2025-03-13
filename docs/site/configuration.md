---
title: Configuration
nav_order: 6
---
# Configuration

The following configurations can be supplied as Spark properties.

| Property name   | Default | Meaning  |
| :-------------- | :------ | :------- |
| spark.rapids.ml.uvm.enabled | false | if set to true, enables [CUDA unified virtual memory](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/) (aka managed memory) during estimator.fit() operations to allow processing of larger datasets than would fit in GPU memory|

Since the algorithms rely heavily on Pandas UDFs, we also require `spark.sql.execution.arrow.pyspark.enabled=true` to ensure efficient data transfer between the JVM and Python processes. 