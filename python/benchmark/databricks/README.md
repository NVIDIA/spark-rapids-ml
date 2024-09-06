# Benchmarking on Databricks

This directory contains shell scripts for running larger scale benchmarks on Databricks AWS hosted Spark service using the Databricks CLI.  You will need a Databricks AWS account to run them.  The benchmarks use datasets synthetically generated using [gen_data.py](../gen_data.py).  For convenience, these have been precomputed and currently stored in the public S3 bucket `spark-rapids-ml-bm-datasets-public`.  The benchmark scripts are currently configured to read the data from there.

## Setup

1. Install latest [databricks-cli](https://docs.databricks.com/dev-tools/cli/index.html) on your local workstation.   Note that Databricks has deprecated the legacy python based cli in favor of a self contained executable.  Make sure the new version is first on the executables PATH.
    ```bash
    curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
    ```

2. Generate an [access token](https://docs.databricks.com/dev-tools/api/latest/authentication.html) for your Databricks workspace in the `User Settings` section of the workspace UI.

3. Configure the access token for the Databricks CLI.  If you have multiple workspaces, you should use a distinct `profile` name for this one, e.g. `aws`, or else it will overwrite your current `DEFAULT` profile.  This profile needs to be supplied on all invocations of the `databricks` cli via the `--profile` option.  For safety, the instructions below assume you are using a new `aws` profile.
    ```bash
    export DB_PROFILE=aws
    databricks configure --token --profile $DB_PROFILE
    # Host: <copy-and-paste databricks workspace url>
    # Token: <copy-and-paste access token from UI>
    ```
4. Next, in [this](./) directory, run the following to upload the files required to run the benchmarks:
    ```bash
    # change below to desired dbfs location WITHOUT DBFS URI for uploading benchmarking related files
    export BENCHMARK_HOME=/path/to/benchmark/files/in/dbfs

    # need separate directory for cluster init script as databricks requires these to be stored in the workspace and not dbfs
    # ex: /Users/<databricks-user-name>/benchmark
    export WS_BENCHMARK_HOME=/path/to/benchmark/files/in/workspace

    ./setup.sh
    ```
    This will create and copy the files into a DBFS directory at the path specified by `BENCHMARK_HOME` and a cluster init script to the workspace directory specified by `WS_BENCHMARK_HOME`.   The script will not overwrite existing files and instead simply print the error message returned from databricks.  If overwrite is desired, first deleted the files and/or directories using `databricks fs rm [-r] <dbfs path>` for the dbfs files and `databricks workspace delete [--recursive] <workspace path>` for the workspace files.
    Note: Export `BENCHMARK_HOME`, `WS_BENCHMARK_HOME` and `DB_PROFILE` in any new/different shell in which subsequent steps may be run.

## Running the benchmarks

1. The running time of each individual benchmark run can be limited by the `TIME_LIMIT` environment variable.  The cpu kmeans benchmark takes over 9000 seconds (ie., > 2 hours) to complete.  If not set, the default is `3600` seconds.
    ```bash
    export TIME_LIMIT=3600
    ```

2. The benchmarks can be run as
    ```bash
    ./run_benchmark.sh [cpu|gpu|gpu_etl] [[12.2|13.3|14.3]] >> benchmark_log
    ```

    The script creates a cpu or gpu cluster, respectively using the cluster specs in [cpu_cluster_spec](./cpu_cluster_spec.sh), [gpu_cluster_spec](./gpu_cluster_spec.sh), [gpu_etl_cluster_spec](./gpu_etl_cluster_spec.sh), depending on the supplied argument.  In gpu and gpu_etl mode each algorithm benchmark is run 3 times, and similarly in cpu mode, except for kmeans and random forest classifier and regressor which are each run 1 time due to their long running times.  gpu_etl mode also uses the [spark-rapids](https://github.com/NVIDIA/spark-rapids) gpu accelerated plugin.

    An optional databricks runtime version can be supplied as a second argument, with 13.3 being the default if not specified.   Runtimes higher than 13.3 are only compatible with cpu and gpu modes (i.e. not gpu_etl) as they are not yet supported by the spark-rapids plugin.  

3. The file `benchmark_log` will have the fit/train/transform running times and accuracy scores.  A simple convenience script has been provided to extract timing information for each run:
    ```bash
    ./process_bm_log.sh benchmark_log
    ```

4. **Cancelling** a run:  Hit `Ctrl-C` and then cancel the run with the last printed `runid` (check using `tail benchmark_log`) by executing:
  ```bash
  databricks jobs cancel-run <runid> --profile $DB_PROFILE
  ```

5. The created clusters are configured to terminate after 30 min, but can be manually terminated or deleted via the Databricks UI.

6. Monitor progress periodically in case of a possible hang, to avoid incurring cloud costs in such cases.
