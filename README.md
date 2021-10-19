# RAPIDS Accelerator for Apache Spark ML

The RAPIDS Accelerator for Apache Spark ML provides a set of GPU accelerated Spark ML algorithms.


## API change

We describe main API changes for GPU accelerated algorithms:



### 1. PCA

Comparing to the original PCA training API:

```scala
val pca = new org.apache.spark.ml.feature.PCA()
  .setInputCol("feature")
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
  .meanCentering(true) // or false, default: true. Wwitch to do mean centering or not before computing covariance matrix
```
## Build

### Prerequisites:
1. essential build tools: 
    - [cmake(>=3.20)](https://cmake.org/download/), 
    - [ninja(>=1.10)](https://github.com/ninja-build/ninja/releases),
    - [gcc(>=9.3)](https://gcc.gnu.org/releases.html)
2. [CUDA Toolkit(>=11.0)](https://developer.nvidia.com/cuda-toolkit)
3. conda: use [miniconda](https://docs.conda.io/en/latest/miniconda.html) to maintain header files and cmake dependecies
4. [RMM(21.12))](https://github.com/rapidsai/rmm):
    - we need all header files and some extra cmake dependencies, build instructions:
    ```bash
    $ git clone --recurse-submodules -b branch-21.12 https://github.com/rapidsai/rmm.git
    $ cd rmm
    $ mkdir build                                       # make a build directory
    $ cd build                                          # enter the build directory
    $ cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX     # configure cmake ... use $CONDA_PREFIX if you're using Anaconda
    $ make -j                                           # compile the library librmm.so ... '-j' will start a parallel job using the number of physical cores available on your system
    $ make install                                      # install the library librmm.so to '/install/path'
    ```
5. [RAFT(21.12)](https://github.com/rapidsai/raft):
    - raft provides only header files, so no build instructions for it.
    ```bash
    $ git clone -b branch-21.12 https://github.com/rapidsai/raft.git
    ```
6. export RMM_PATH and RAFT_PATH:
    ```bash
    export RAFT_PATH=PATH_TO_YOUR_RAFT_FOLDER
    export RMM_PATH=PATH_TO_YOUR_RMM_FOLDER
    ```
### Build target jar
User can build it directly in the _project root path_ with:
```
mvn clean package
```
Then `rapids-4-spark-ml_2.12-21.10.0-SNAPSHOT.jar` will be generated under `target` folder.

_Note_: This module contains both native and Java/Scala code. The native library build instructions has been added to the pom.xml file so that maven build command will help build native library all the way. Make sure the prerequisites are all met, or the build will fail with error messages accordingly such as "cmake not found" or "ninja not found" etc. 

## How to use

Add the artifact jar to the Spark, for example:
```bash
$SPARK_HOME/bin/spark-shell --master $SPARK_MASTER \
 --driver-memory 20G \
 --executor-memory 30G \
 --conf spark.driver.maxResultSize=8G \
 --jars target/rapids-4-spark-ml_2.12-21.10.0-SNAPSHOT.jar \
 --conf spark.task.resource.gpu.amount=0.08 \
 --conf spark.executor.resource.gpu.amount=1 \
 --conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
 --files ${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh
```
### PCA examples

Please refer to [PCA examples](https://github.com/NVIDIA/spark-rapids-examples/tree/branch-21.10/examples/pca) for more details about example code.
