# RAPIDS Accelerator for Apache Spark ML

The RAPIDS Accelerator for Apache Spark ML provides a set of GPU accelerated Spark ML algorithms.


## API change

We describe main API changes for GPU accelerated algorithms:



### 1. PCA

Comparing to the original PCA training API:

```scala
val pca = new org.apache.spark.ml.feature.PCA()
  .setInputCol("feature_vector_type")
  .setOutputCol("feature_value_3d")
  .setK(3)
  .fit(vectorDf)
```

We used a customized class and user will need to do `no code change` to enjoy the GPU acceleration:

```scala
val pca = new com.nvidia.spark.ml.feature.PCA()
  .setInputCol("feature_array_type") // accept ArrayType column, no need to convert it to Vector type
  .setOutputCol("feature_value_3d")
  .setK(3)
  .fit(vectorDf)
...
```

Note: The `setInputCol` is targeting the input column of `Vector` type for training process in `CPU`
version. But in GPU version, user doesn't need to do the extra preprocess step to convert column of
`ArrayType` to `Vector` type, the `setInputCol` will accept the raw `ArrayType` column.

## Build

### Prerequisites:
1. essential build tools: 
    - [cmake(>=3.20)](https://cmake.org/download/), 
    - [ninja(>=1.10)](https://github.com/ninja-build/ninja/releases),
    - [gcc(>=9.3)](https://gcc.gnu.org/releases.html)
2. [CUDA Toolkit(>=11.0)](https://developer.nvidia.com/cuda-toolkit)
3. conda: use [miniconda](https://docs.conda.io/en/latest/miniconda.html) to maintain header files
and cmake dependecies
4. [cuDF](https://github.com/rapidsai/cudf):
    - install cuDF shared library via conda:
      ```bash
      conda install -c rapidsai-nightly -c nvidia -c conda-forge cudf=22.04 python=3.8 -y
      ```
5. [RAFT(22.06)](https://github.com/rapidsai/raft):
    - raft provides only header files, so no build instructions for it.
      ```bash
      $ git clone -b branch-22.06 https://github.com/rapidsai/raft.git
      ```
6. export RAFT_PATH:
    ```bash
    export RAFT_PATH=PATH_TO_YOUR_RAFT_FOLDER
    ```
### Build target jar
User can build it directly in the _project root path_ with:
```
mvn clean package
```
Then `rapids-4-spark-ml_2.12-22.06.0-SNAPSHOT.jar` will be generated under `target` folder.

_Note_: This module contains both native and Java/Scala code. The native library build instructions
has been added to the pom.xml file so that maven build command will help build native library all
the way. Make sure the prerequisites are all met, or the build will fail with error messages
accordingly such as "cmake not found" or "ninja not found" etc.

## How to use
When building the jar, spark-rapids plugin jar will be downloaded to your local maven
repository, usually in your `~/.m2/repository`.

Add the artifact jar to the Spark, for example:
```bash
ML_JAR="target/rapids-4-spark-ml_2.12-22.06.0-SNAPSHOT.jar"
PLUGIN_JAR="~/.m2/repository/com/nvidia/rapids-4-spark_2.12/22.06.0-SNAPSHOT/rapids-4-spark_2.12-22.06.0-SNAPSHOT.jar"

$SPARK_HOME/bin/spark-shell --master $SPARK_MASTER \
 --driver-memory 20G \
 --executor-memory 30G \
 --conf spark.driver.maxResultSize=8G \
 --jars ${ML_JAR},${PLUGIN_JAR} \
 --conf spark.plugins=com.nvidia.spark.SQLPlugin \
 --conf spark.rapids.sql.enabled=true \
 --conf spark.task.resource.gpu.amount=0.08 \
 --conf spark.executor.resource.gpu.amount=1 \
 --conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
 --files ${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh
```
### PCA examples

Please refer to
[PCA examples](https://github.com/NVIDIA/spark-rapids-examples/blob/branch-22.06/examples/Spark-cuML/pca/) for
more details about example code. We provide both
[Notebook](https://github.com/NVIDIA/spark-rapids-examples/blob/branch-22.06/examples/Spark-cuML/pca/PCA-example-notebook.ipynb)
and [jar](https://github.com/NVIDIA/spark-rapids-examples/blob/branch-22.06/examples/Spark-cuML/pca/scala/src/com/nvidia/spark/examples/pca/Main.scala)
 versions there. Instructions to run these examples are described in the
 [README](https://github.com/NVIDIA/spark-rapids-examples/blob/branch-22.06/examples/Spark-cuML/pca/README.md).
