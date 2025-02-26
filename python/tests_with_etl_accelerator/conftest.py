#
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
import os

from pyspark.sql import SparkSession

_gpu_number = 1
_default_conf = {
    "spark.master": f"local[{_gpu_number}]",
    "spark.python.worker.reuse": "false",
    "spark.driver.host": "127.0.0.1",
    "spark.task.maxFailures": "1",
    "spark.driver.memory": "128g",
    "spark.sql.execution.pyspark.udf.simplifiedTraceback.enabled": "false",
    "spark.sql.pyspark.jvmStacktrace.enabled": "true",
    "spark.sql.execution.arrow.pyspark.enabled": "true",
    "spark.rapids.ml.uvm.enabled": True,
}

import os
jar_path = os.environ["SPARK_RAPIDS_PLUGIN_JAR"]

accelerator_conf = {
    "spark.jars": jar_path,
    "spark.executorEnv.PYTHONPATH" : jar_path,
    "spark.rapids.sql.concurrentGpuTasks": "1",
    "spark.rapids.memory.pinnedPool.size": "2G",
    "spark.sql.files.maxPartitionBytes": "512m",
    "spark.plugins" : "com.nvidia.spark.SQLPlugin",
    "spark.rapids.sql.python.gpu.enabled": "true",
    "spark.rapids.sql.explain": "NOT_ON_GPU",
    "spark.rapids.sql.rowBasedUDF.enabled": "true",
}

def _get_spark(confs) -> SparkSession:
    builder = SparkSession.builder.appName(
        name="spark-rapids-ml with tests on large datasets"
    )
    for k, v in confs.items():
        builder.config(k, v)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    logging.getLogger("pyspark").setLevel(logging.WARN)
    return spark


_spark = _get_spark({**_default_conf, **accelerator_conf})
