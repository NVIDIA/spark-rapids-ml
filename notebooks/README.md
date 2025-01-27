# Running notebooks locally

To run notebooks using Spark local mode on a server with one or more NVIDIA GPUs:
1. Follow the [installation instructions](../python/README.md#installation) to setup your environment.
2. Install `jupyter` into the conda environment.
    ```bash
    pip install jupyter
    ```
3. Set `SPARK_HOME`.
    ```bash
    export SPARK_HOME=$( pip show pyspark | grep Location | grep -o '/.*' )/pyspark
    ls $SPARK_HOME/bin/pyspark
    ```
4. In the notebooks directory, start PySpark in local mode with the Jupyter UI.
    ```bash
    cd spark-rapids-ml/notebooks

    PYSPARK_DRIVER_PYTHON=jupyter \
    PYSPARK_DRIVER_PYTHON_OPTS='notebook --ip=0.0.0.0' \
    CUDA_VISIBLE_DEVICES=0 \
    $SPARK_HOME/bin/pyspark --master local[12] \
    --driver-memory 128g \
    --conf spark.sql.execution.arrow.pyspark.enabled=true
    ```
5. Follow the instructions printed by the above command to browse to the Jupyter notebook server.
6. In the Jupyter file browser, open and run any of the notebooks.
7. **OPTIONAL**: If your server is remote with no direct `http` access, but you have `ssh` access, you can connect via an `ssh` tunnel, as follows:
    ```bash
    export REMOTE_USER=<your_remote_username>
    export REMOTE_HOST=<your_remote_hostname>
    ssh -A -L 8888:127.0.0.1:8888 -L 4040:127.0.0.1:4040 ${REMOTE_USER}@${REMOTE_HOST}
    ```
    Then, browse to the `127.0.0.1` URL printed by the command in step 4.   Note that a tunnel is also opened to the Spark UI server on port 4040.  Once a notebook is opened, you can view it by browsing to http://127.0.0.1:4040 in another tab or window.
8. **OPTIONAL**: If you have multiple GPUs in your server, replace the `CUDA_VISIBLE_DEVICES` setting in step 4 with a comma-separated list of the corresponding indices.  For example, for two GPUs use `CUDA_VISIBLE_DEVICES=0,1`.

## No import change
In the default notebooks, the GPU accelerated implementations of algorithms in Spark MLlib are enabled via import statements from the `spark_rapids_ml` package.   

Alternatively, acceleration can also be enabled by executing the following import statement at the start of a notebook:
```
import spark_rapids_ml.install
```
or by modifying the PySpark/Jupyter launch command above to use a CLI `pyspark-rapids` installed by our `pip` package to start Jupyter with pyspark as follows: 
```bash
cd spark-rapids-ml/notebooks

PYSPARK_DRIVER_PYTHON=jupyter \
PYSPARK_DRIVER_PYTHON_OPTS='notebook --ip=0.0.0.0' \
CUDA_VISIBLE_DEVICES=0 \
pyspark-rapids --master local[12] \
--driver-memory 128g \
--conf spark.sql.execution.arrow.pyspark.enabled=true
``` 

After executing either of the above, all subsequent imports and accesses of supported accelerated classes from `pyspark.ml` will automatically redirect and return their counterparts in `spark_rapids_ml`.  Unaccelerated classes will import from `pyspark.ml` as usual.  Thus, all supported acceleration in an existing `pyspark` notebook is enabled with no additional import statement or code changes.  Directly importing from `spark_rapids_ml` also still works (needed for non-MLlib algorithms like UMAP).

For an example notebook, see the notebook [kmeans-no-import-change.ipynb](kmeans-no-import-change.ipynb).

*Note*: As of this release, in this mode, the remaining unsupported methods and attributes on accelerated classes and objects will still raise exceptions.

## Running notebooks on Databricks
See [these instructions](databricks/README.md) for running the notebooks in a Databricks Spark cluster.

## Running notebooks on Google Dataproc
See [these instructions](dataproc/README.md) for running the notebooks in a Dataproc Spark cluster.

## Running notebooks on AWS EMR
See [these instructions](aws-emr/README.md) for running the notebooks in an AWS-EMR cluster.

