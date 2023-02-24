## Running notebooks locally
To run notebooks locally using Spark local mode on a server with 1 or more NVIDIA GPUs do the following:
- Follow the installation and usage instructions in the top-level [README](../../README.md) 
- Install `jupyter`:
    ```bash
    pip install jupyter
    ```
- Set `SPARK_HOME` by running the command 
    ```bash
    export SPARK_HOME=$( pip show pyspark | grep Location | grep -o '/.*' )/pyspark
    ls $SPARK_HOME/bin/pyspark
    ```
- In the directory [notebooks](../notebooks) directory, run the following command to start pyspark in local mode with a jupyter notebook serving as the shell input:
    ```bash
    PYSPARK_DRIVER_PYTHON=jupyter PYSPARK_DRIVER_PYTHON_OPTS='notebook --ip=0.0.0.0' CUDA_VISIBLE_DEVICES=0 \
    $SPARK_HOME/bin/pyspark --master local[12] --driver-memory 128g --conf spark.sql.execution.arrow.pyspark.enabled=true
    ```
- Follow the instructions printed by the above command to connect a browser to the jupyter notebook server.
- From the `jupyter` file browser open and run any of the notebooks in [notebooks](../notebooks).
- If your server is remote with no direct `http` access, but you have `ssh` access, you can connect via an `ssh` tunnel:
    ```bash
    ssh -A -L 8888:127.0.0.1:8888 -L 4040:127.0.0.1:4040 -L user_name@host
    ```
    and then navigate your local browser to the URL printed by the above command (use the one with `127.0.0.1`).
- If you have multiple GPUs in your server, replace the `CUDA_VISIBLE_DEVICES` setting with a comma separated list of the corresponding indices.  For example, for 2 GPUs use `CUDA_VISIBLE_DEVICES=0,1` in the above command.


## Running in the cloud:
See [databricks](databricks/README.md) for instructions on how to run the notebooks in a Databricks Spark cluster.

