# Running notebooks locally

To run notebooks using Spark local mode on a server with one or more NVIDIA GPUs:
1. Follow the [installation instructions](../README_python.md#installation) to setup your environment.
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
    export USER=<your_username>
    export HOST=<your_hostname>
    ssh -A -L 8888:127.0.0.1:8888 -L 4040:127.0.0.1:4040 -L $USER@$HOST
    ```
    Then, browse to the `127.0.0.1` URL printed by the command in step 4.
8. **OPTIONAL**: If you have multiple GPUs in your server, replace the `CUDA_VISIBLE_DEVICES` setting in step 4 with a comma-separated list of the corresponding indices.  For example, for two GPUs use `CUDA_VISIBLE_DEVICES=0,1`.

## Running notebooks on Databricks
See [these instructions](databricks/README.md) for running the notebooks in a Databricks Spark cluster.

