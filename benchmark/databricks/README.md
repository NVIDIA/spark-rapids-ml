# Benchmarking on Databricks

This directory contains shell scripts for running larger scale benchmarks on Databricks AWS hosted Spark service using the Databricks CLI.  You will need a Databricks AWS account to run them.  The benchmarks use datasets synthetically generated using [gen_data.py](../gen_data.py).  For convenience, these have been precomputed and currently stored in the public S3 bucket `spark-rapids-ml-bm-datasets-public`.  The benchmark scripts are currently configured to read the data from there.

## Setup

1. Install [databricks-cli](https://docs.databricks.com/dev-tools/cli/index.html) on your local workstation.
    ```bash
    pip install databricks-cli --upgrade
    ```

2. Generate an [access token](https://docs.databricks.com/dev-tools/api/latest/authentication.html) for your Databricks workspace in the `User Settings` section of the workspace UI.

3. Configure the access token for the Databricks CLI.  If you have multiple workspaces, you should use a distinct `profile` name for this one, e.g. `aws`, or else it will overwrite your current `DEFAULT` profile.  This profile needs to be supplied on all invocations of the `databricks` cli via the `--profile` option.  For safety, the instructions below assume you are using a new `aws` profile.
    ```bash
    export DB_PROFILE=aws
    databricks configure --token --profile $DB_PROFILE
    # 
    # Token: <copy-and-paste access token from UI>
    ```

4. Configure the Databricks cli to use the jobs 2.0 api.
    ```bash
    databricks jobs configure --version=2.0 --profile $DB_PROFILE
    ```

5. Next, in [this](./) directory, run the following to upload the files required to run the benchmarks:
    ```bash
    #change below to desired dbfs location WITHOUT DBFS URI for uploading benchmarking related files
    export BENCHMARK_HOME=/path/to/benchmark/files/in/dbfs
    ./setup.sh
    ```
    This will create and copy the files into a DBFS folder at the path specified by `BENCHMARK_HOME`.
    Note: Export `BENCHMARK_HOME` and `DB_PROFILE` in any new/different shell in which subsequent steps may be run.

## Running the benchmarks

1. The running time of each individual benchmark run can be limited by the `TIME_LIMIT` environment variable.  The cpu kmeans benchmark takes over 9000 seconds (ie., > 2 hours) to complete.  If not set, the default is `3600` seconds.
    ```bash
    export TIME_LIMIT=3600
    ```

2. The benchmarks can be run as
    ```bash
    ./run_benchmarks.sh [cpu|gpu] >> benchmark_log
    ```

    The script creates a cpu or gpu cluster, respectively using the cluster specs in [cpu_cluster_spec](./cpu_cluster_spec.sh) and [gpu_cluster_spec](./gpu_cluster_spec.sh), depending on the supplied argument.  In gpu mode each algorithm benchmark is run 3 times, and similarly in cpu mode, except for kmeans and random forest classifier and regressor which are each run 1 time due to their long running times.

3. The file `benchmark_log` will have the fit/train/transform running times and accuracy scores.  A simple convenience script has been provided to extract timing information for each run:
    ```bash
    ./process_bm_log.sh benchmark_log
    ```

4. **Cancelling** a run:  Hit `Ctrl-C` and then cancel the run with the last printed `runid` (check using `tail benchmark_log`) by executing:
  ```bash
  databricks runs cancel --run-id <runid> --profile $DB_PROFILE
  ```

5. The created clusters are configured to terminate after 30 min, but can be manually terminated or deleted via the Databricks UI.

6. Monitor progress periodically in case of a possible hang, to avoid incurring cloud costs in such cases.
