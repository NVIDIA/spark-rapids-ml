from pyspark.sql import SparkSession
from typing import List, Any


class WithSparkSession(object):
    def __init__(self, confs: List[str]) -> None:
        builder = SparkSession.builder
        for conf in confs:
            key, value = conf.split("=")
            builder = builder.config(key, value)
        self.spark = builder.getOrCreate()

    def __enter__(self) -> SparkSession:
        return self.spark

    def __exit__(self, *args: Any) -> None:
        self.spark.stop()
