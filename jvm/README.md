# Spark Rapids ML Connect Plugin

The Spark Rapids ML Connect Plugin is a project designed to accelerate Spark ML algorithms
in a Spark Connect environment using the spark-rapids-ml Python package. It enables GPU
acceleration for machine learning workloads without requiring changes to the user's existing code.

## Environment Setup

- Install spark-rapids-ml

  Follow
  the [installation guide](https://github.com/NVIDIA/spark-rapids-ml/blob/branch-25.06/python/README.md#installation)
  to install the spark-rapids-ml package on the server side.

- Setup Spark

  Download the latest Spark snapshot archive
  from [this site](https://urm.nvidia.com/artifactory/sw-spark-maven-local/org/apache/spark/4.1.0-SNAPSHOT/)

  Extract the archive and set the `SPARK_HOME` environment variable to point to the Spark directory.

- Compile the Spark Rapids ML Connect Plugin

  To compile the plugin, run the following command:

    ``` shell
    mvn clean package -DskipTests
    ```

  if you would like to compile the plugin and run the unit tests, run the following command:

    ``` shell
    export PYSPARK_PYTHON=YOUR_PYTHON_PATH_WITH_SPARK_RAPIDS_ML
    mvn clean package
    ```

  After compilation, the latest JAR file, `com.nvidia.rapids.ml-<LATEST_VERSION>.jar`, will be
  available in the `target` directory.

- Install PySpark Connect Client

  To install the PySpark Connect client on the client side, follow these steps:

    ``` shell
    cd $SPARK_HOME/python
    python packaging/client/setup.py sdist

    # Create a new conda environment for the client
    conda create -n pyspark-client python==3.12
    conda activate pyspark-client

    # Install the PySpark client package
    pip install $SPARK_HOME/dist/pyspark-client-4.1.0.dev0.tar.gz
    ```

  This will set up the PySpark client in the pyspark-client conda environment.

## Testing

This section outlines the steps to test Spark Connect with the RAPIDS ML plugin,
including setting up the server and running client-side tests.

### Start connect server (server side)

To start the Spark Connect server with RAPIDS ML support, follow these steps:

``` shell
export PYSPARK_PYTHON=YOUR_PYTHON_PATH_WITH_SPARK_RAPIDS_ML
start-connect-server.sh --master local[*] \
  --jars ${SPARK_HOME}/jars/spark-connect_2.13-4.1.0-SNAPSHOT.jar,com.nvidia.rapids.ml-1.0-SNAPSHOT.jar \
  --conf spark.driver.memory=20G
```

### Run the tests (client side)

Once the server is running, you can connect to it from a client under the `pyspark-client` environment
with Spark Connect support. Run below testing code to test it:

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

print(f"model.intercept: {model.intercept}")
print(f"model.coefficients: {model.coefficients}")

model.transform(df).show()
```
