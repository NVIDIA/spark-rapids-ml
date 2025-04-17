---
title: Configuration
nav_order: 6
---
# Configuration

The following configurations can be supplied as Spark properties.

| Property name   | Default | Meaning  |
| :-------------- | :------ | :------- |
| spark.rapids.ml.uvm.enabled | false | if set to true, enables [CUDA unified virtual memory](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/) (aka managed memory) during estimator.fit() operations to allow processing of larger datasets than would fit in GPU memory|
| spark.rapids.ml.gpuMemRatioForData | None |  If set to a float value between 0 and 1, Spark Rapids ML will reserve a portion of GPU memory on each GPU and incrementally append PySpark data batches into this reserved space. This setting is recommended for large datasets, as it prevents duplicating the entire dataset in GPU memory and reduces the risk of out-of-memory errors. |

Since the algorithms rely heavily on Pandas UDFs, we also require `spark.sql.execution.arrow.pyspark.enabled=true` to ensure efficient data transfer between the JVM and Python processes. 