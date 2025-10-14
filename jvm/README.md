# Spark Rapids ML Connect Plugin

The Spark Rapids ML Connect Plugin is designed to accelerate Spark ML algorithms
in a Spark Connect environment using the spark-rapids-ml Python package. It enables GPU
acceleration for machine learning workloads without requiring changes to the user's existing code.
The plugin requires Spark 4.0.  See [https://spark.apache.org/docs/latest/spark-connect-overview.html](https://spark.apache.org/docs/latest/spark-connect-overview.html) for an overview of Spark Connect, which allows running Spark SQL, DataFrame, and MLlib based applications in thin GRPC protocol clients.

## Getting Started
### Requirements
JDK 17, Spark 4.0

### Environment Setup

- Install spark-rapids-ml and dependencies

  Follow
  the [installation guide](../python/README.md#installation)
  to install the spark-rapids-ml package and dependencies on the server side.    Follow the `## OPTIONAL: for package installation` part of the instructions.

- Setup Spark

  Download the Spark 4.0 tgz file with the Spark Connect enabled 'package type'
  from [spark.apache.org](https://spark.apache.org/downloads.html)

  Extract the archive and note the directory for reference below.


- Install PySpark Connect Client

  To install the PySpark Connect client on the client side, follow these steps:

    ```shell
    # Create a new conda environment for the client
    conda create -n pyspark-client python==3.10
    conda activate pyspark-client

    # Install the PySpark client package
    pip install pyspark-client
    ```

  This will set up the PySpark client in the pyspark-client conda environment.

### Testing

This section outlines the steps to test Spark Connect with the Spark Rapids ML plugin,
including setting up the server and running client-side tests.

#### Start connect server (server side)

To start the Spark Connect server with Spark Rapids ML support, follow these steps:

```shell
conda activate rapids-25.10  # from spark-rapids-ml installation
export SPARK_HOME=<directory where spark was installed above>
export PYSPARK_PYTHON=$(which python)
export PLUGIN_JAR=$(pip show spark-rapids-ml | grep Location: | cut -d ' ' -f 2 )/spark_rapids_ml/jars/com.nvidia.rapids.ml-25.10.0.jar
$SPARK_HOME/sbin/start-connect-server.sh --master local[*] \
  --jars $PLUGIN_JAR \
  --conf spark.driver.memory=20G
```
Notice that the plugin jar is bundled in the pip installed `spark-rapids-ml` python package.

#### Run the tests (client side)

Once the server is running, you can connect to it from a client under the `pyspark-client` environment
with Spark Connect support.  

Run below to test it:

```shell
conda activate pyspark-client
cat <<EOF >test_connect.py
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
EOF
python test_connect.py
```

## Building the Spark Rapids ML Connect Plugin

To compile the plugin, clone this repo and run the following command in the `jvm` directory:

``` shell
mvn clean package -DskipTests
```

if you would like to compile the plugin and run the unit tests, install `spark-rapids-ml` python package and its dependencies per the above instructions and run the following command:

``` shell
conda activate rapids-25.10
export PYSPARK_PYTHON=$(which python)
mvn clean package
```

After compilation, the latest JAR file, `com.nvidia.rapids.ml-<LATEST_VERSION>.jar`, will be
available in the `target` directory.
