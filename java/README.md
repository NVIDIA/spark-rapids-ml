# Spark Rapids ML Java

Spark-rapids-ml java project is the java wrapper of spark-rapids-ml python package,
which could accelerate the Spark ML library on the connect environment without changing
users' code.

## Compile

``` shell
mvn clean package
```

After compiling, you can get the latest `com.nvidia.rapids.ml-1.0-SNAPSHOT.jar` under target

## Deploy

### Installation

Follow up [this guide](https://github.com/NVIDIA/spark-rapids-ml/blob/branch-25.02/python/README.md#installation) to
install spark-rapids-ml

### Start connect server

``` shell
export PYSPARK_PYTHON=YOUR_PATH_PATH_WITH_SPARK_RAPIDS_ML
start-connect-server.sh --master local[*] \
  --jars ${SPARK_HOME}/jars/spark-connect_2.13-4.0.0-SNAPSHOT.jar,com.nvidia.rapids.ml-1.0-SNAPSHOT.jar \
  --conf spark.driver.memory=20G
```

### Test

Then you could play around the following code,

```shell
from pyspark.ml.classification import (LogisticRegression,
                                       LogisticRegressionModel)
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

spark = (SparkSession.builder.remote("sc://localhost")
         .config("spark.connect.ml.backend.classes", "com.nvidia.rapids.ml.Plugin")
         .getOrCreate())

df = spark.createDataFrame([
        (Vectors.dense([1.0, 2.0]), 1),
        (Vectors.dense([2.0, -1.0]), 1),
        (Vectors.dense([-3.0, -2.0]), 0),
        (Vectors.dense([-1.0, -2.0]), 0),
        ], schema=['features', 'label'])
lr = LogisticRegression(maxIter=19, tol=0.0023)
model = lr.fit(df)
print(f"======== model.intercept: {model.intercept}")
print(f"======== model.coefficients: {model.coefficients}")
model.transform(df).show()
```
