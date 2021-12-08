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
...
```

Besides, we provide some switch APIs to allow users to highly customize their training process:

```scala
  .useGemm(true) // or false, default: true. Switch to use original BLAS bsr or cuBLAS gemm to compute covariance matrix
  .useCuSolverSVD(true) // or false, default: true. Switch to use original LAPack solver or cuSolver to compute SVD
  .meanCentering(true) // or false, default: true. Switch to do mean centering or not before computing covariance matrix
```

To speedup the transform process, it's required to add an extra setting:
```scala
  .setTransformInputCol("feature_array_type")
```
Note: The `setInputCol` is targeting the input column of `Vector` type for training process, while
 the `setTransformInputCol` is for column of ArrayType.

## Build

### Prerequisites:
1. essential build tools: 
    - [cmake(>=3.20)](https://cmake.org/download/), 
    - [ninja(>=1.10)](https://github.com/ninja-build/ninja/releases),
    - [gcc(>=9.3)](https://gcc.gnu.org/releases.html)
2. [CUDA Toolkit(>=11.5)](https://developer.nvidia.com/cuda-toolkit)
3. conda: use [miniconda](https://docs.conda.io/en/latest/miniconda.html) to maintain header files
and cmake dependecies
4. [cuDF](https://github.com/rapidsai/cudf):
    - install cuDF shared library via conda:
      ```bash
      conda install -c rapidsai-nightly -c nvidia -c conda-forge cudf=21.12 python=3.8 -y
      ```
5. [RAFT(21.12)](https://github.com/rapidsai/raft):
    - raft provides only header files, so no build instructions for it.
      ```bash
      $ git clone -b branch-21.12 https://github.com/rapidsai/raft.git
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
Then `rapids-4-spark-ml_2.12-21.12.0-SNAPSHOT.jar` will be generated under `target` folder.

_Note_: This module contains both native and Java/Scala code. The native library build instructions
has been added to the pom.xml file so that maven build command will help build native library all
the way. Make sure the prerequisites are all met, or the build will fail with error messages
accordingly such as "cmake not found" or "ninja not found" etc.

## How to use
When building the jar, cudf jar and spark-rapids plugin jar will be downloaded to your local maven
repository, usually in your `$HOME/.m2/repository`.

Add the artifact jar to the Spark, for example:
```bash
ML_JAR="target/rapids-4-spark-ml_2.12-21.12.0-SNAPSHOT.jar"
CUDF_JAR="$HOME/.m2/repository/ai/rapids/cudf/21.12.0-SNAPSHOT/cudf-21.12.0-SNAPSHOT.jar"
PLUGIN_JAR="$HOME/.m2/repository/com/nvidia/rapids-4-spark_2.12/21.12.0-SNAPSHOT/rapids-4-spark_2.12-21.12.0-SNAPSHOT.jar"

$SPARK_HOME/bin/spark-shell --master $SPARK_MASTER \
 --driver-memory 20G \
 --executor-memory 30G \
 --conf spark.driver.maxResultSize=8G \
 --jars ${ML_JAR},${CUDF_JAR},${PLUGIN_JAR} \
 --conf spark.plugins=com.nvidia.spark.SQLPlugin \
 --conf spark.rapids.sql.enabled=true \
 --conf spark.task.resource.gpu.amount=0.08 \
 --conf spark.executor.resource.gpu.amount=1 \
 --conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
 --files ${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh
```
### PCA examples

Please refer to
[PCA examples](https://github.com/NVIDIA/spark-rapids-examples/blob/branch-21.12/examples/pca/main.scala) for
more details about example code.
