import os

from pyspark.ml.classification import (LogisticRegression,
                                       LogisticRegressionModel)
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

spark = (SparkSession.builder.remote("sc://localhost")
         .config("spark.connect.ml.backend.classes", "com.nvidia.rapids.ml.Plugin")
         .getOrCreate())
spark.addArtifact("/home/bobwang/work.d/spark-rapids-ml/java/target/com.nvidia.rapids.ml-1.0-SNAPSHOT.jar")

def run_test():
    df = spark.createDataFrame([
        (Vectors.dense([1.0, 2.0]), 1),
        (Vectors.dense([2.0, -1.0]), 1),
        (Vectors.dense([-3.0, -2.0]), 0),
        (Vectors.dense([-1.0, -2.0]), 0),
    ], schema=['features', 'label'])
    lr = LogisticRegression()
    lr.fit(df)

# Train with com.nvidia.rapids.ml.LogisticRegression
print("first test")
run_test()
# spark.conf.unset("spark.connect.ml.backend.classes")
# run_test()

