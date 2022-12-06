from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from typing import List
def prepare_spark_session(spark_confs: List[str]) -> SparkSession:
    builder = SparkSession.builder
    for sconf in spark_confs:
        key, value = sconf.split("=")
        builder = builder.config(key, value)
    spark = builder.getOrCreate()
    return spark
