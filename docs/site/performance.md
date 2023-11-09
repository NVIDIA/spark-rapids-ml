---
title: Performance Tuning
nav_order: 6
---
# Performance Tuning

* TOC
{:toc}

## Stage-level scheduling

Starting from spark-rapids-ml `23.10.0`, stage-level scheduling is automatically enabled.
Therefore, if you are using Spark **standalone** cluster version **`3.4.0`** or higher, we strongly recommend
configuring the `"spark.task.resource.gpu.amount"` as a fractional value. This will
enable running multiple tasks in parallel during the ETL phase to help the performance. An example configuration
would be `"spark.task.resource.gpu.amount=1/spark.executor.cores"`. For example,

``` bash
spark-submit \
  --master spark://<master-ip>:7077 \
  --conf spark.executor.cores=12 \
  --conf spark.task.cpus=1 \
  --conf spark.executor.resource.gpu.amount=1 \
  --conf spark.task.resource.gpu.amount=0.08 \
  ...
```

The above spark-submit command specifies a request for 1 GPU and 12 CPUs per executor. So you can see,
a total of 12 tasks per executor will be executed concurrently during the ETL phase. And the stage-level scheduling
is then used internally to the library to automatically carry out the ML training phases using the required 1 gpu per task.

However, if you are using a spark-rapids-ml version earlier than 23.10.0 or a Spark
standalone cluster version below 3.4.0, you need to make sure there will be only 1 task running at any time per executor.
You can set `spark.task.cpus` equal to `spark.executor.cores`, or `"spark.task.resource.gpu.amount"=1`. For example,

``` bash
spark-submit \
  --master spark://<master-ip>:7077 \
  --conf spark.executor.cores=12 \
  --conf spark.task.cpus=1 \
  --conf spark.executor.resource.gpu.amount=1 \
  --conf spark.task.resource.gpu.amount=1 \
  ...
```
