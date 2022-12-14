from time import time

from pyspark.sql import SparkSession
from typing import List, Any, Callable


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


def with_benchmark(phrase: str, action: Callable) -> Any:
    start = time()
    result = action()
    end = time()
    print('-' * 100)
    print('{} takes {} seconds'.format(phrase, round(end - start, 2)))
    print('-' * 100)
    return result
