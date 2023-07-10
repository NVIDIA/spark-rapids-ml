# Spark Rapids ML (Scala)

### PCA

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

### Build in Docker:

We provide a Dockerfile to build the project in a container. See [docker](../docker/README.md) for more instructions.

### Prerequisites:

1. essential build tools:
    - [cmake(>=3.23.1)](https://cmake.org/download/),
    - [ninja(>=1.10)](https://github.com/ninja-build/ninja/releases),
    - [gcc(>=9.3)](https://gcc.gnu.org/releases.html)
2. [CUDA Toolkit(>=11.5)](https://developer.nvidia.com/cuda-toolkit)
3. conda: use [miniconda](https://docs.conda.io/en/latest/miniconda.html) to maintain header files
and cmake dependecies
4. [cuDF](https://github.com/rapidsai/cudf):
    - install cuDF shared library via conda:
      ```bash
      conda install -c rapidsai -c conda-forge cudf=22.04 python=3.8 -y
      ```
5. [RAFT(22.12)](https://github.com/rapidsai/raft):
    - raft provides only header files, so no build instructions for it. Note we fix the version to
      22.12 to avoid potential API compatibility issues in the future.
      ```bash
      $ git clone -b branch-22.12 https://github.com/rapidsai/raft.git
      ```
6. export RAFT_PATH:
    ```bash
    export RAFT_PATH=ABSOLUTE_PATH_TO_YOUR_RAFT_FOLDER
    ```
Note: For those using other types of GPUs which do not have CUDA forward compatibility (for example, GeForce), CUDA 11.5 or later is required.

### Build target jar

Spark-rapids-ml uses [spark-rapids](https://github.com/NVIDIA/spark-rapids) plugin as a dependency.
To build the _SNAPSHOT_ jar, user needs to build and install the denpendency jar _rapids-4-spark_ first
because there's no snapshot jar for spark-rapids plugin in public maven repositories.
See [build instructions](https://github.com/NVIDIA/spark-rapids/blob/branch-23.04/CONTRIBUTING.md#building-a-distribution-for-multiple-versions-of-spark) to get the dependency jar installed.

User can also modify the pom file to use the _release_ version spark-rapids plugin as the dependency. In this case user doesn't need to manually build and install spark-rapids plugin jar by themselves.

Make sure the _rapids-4-spark_ is installed in your local maven then user can build it directly in
the _project root path_ with:
```
cd jvm
mvn clean package
```
Then `rapids-4-spark-ml_2.12-23.06.0-SNAPSHOT.jar` will be generated under `target` folder.

Users can also use the _release_ version spark-rapids plugin as the dependency if it's already been
released in public maven repositories, see [rapids-4-spark maven repository](https://mvnrepository.com/artifact/com.nvidia/rapids-4-spark)
for release versions. In this case, users don't need to manually build and install spark-rapids
plugin jar by themselves. Remember to replace the [dependency](https://github.com/NVIDIA/spark-rapids-ml/blob/branch-23.04/pom.xml#L94-L96)
in pom file.

_Note_: This module contains both native and Java/Scala code. The native library build instructions
has been added to the pom.xml file so that maven build command will help build native library all
the way. Make sure the prerequisites are all met, or the build will fail with error messages
accordingly such as "cmake not found" or "ninja not found" etc.

## How to use

After the building processes, spark-rapids plugin jar will be installed to your local maven
repository, usually in your `~/.m2/repository`.

Add the artifact jar to the Spark, for example:
```bash
ML_JAR="target/rapids-4-spark-ml_2.12-23.06.0-SNAPSHOT.jar"
PLUGIN_JAR="~/.m2/repository/com/nvidia/rapids-4-spark_2.12/23.06.0-SNAPSHOT/rapids-4-spark_2.12-23.06.0-SNAPSHOT.jar"

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
[PCA examples](https://github.com/NVIDIA/spark-rapids-examples/blob/branch-23.04/examples/ML+DL-Examples/Spark-cuML/pca/) for
more details about example code. We provide both
[Notebook](https://github.com/NVIDIA/spark-rapids-examples/blob/branch-23.04/examples/ML+DL-Examples/Spark-cuML/pca/notebooks/Spark_PCA_End_to_End.ipynb)
and [jar](https://github.com/NVIDIA/spark-rapids-examples/blob/branch-23.04/examples/ML+DL-Examples/Spark-cuML/pca/scala/src/com/nvidia/spark/examples/pca/Main.scala)
 versions there. Instructions to run these examples are described in the
[README](https://github.com/NVIDIA/spark-rapids-examples/blob/branch-23.04/examples/ML+DL-Examples/Spark-cuML/pca/README.md).
