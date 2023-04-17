# Benchmarking on Dataproc

This directory contains shell scripts for running larger scale benchmarks on a Google Dataproc cluster.  You will need a Dataproc account to run them.  The benchmarks use datasets synthetically generated using [gen_data.py](../gen_data.py).  For convenience, these have been precomputed and currently stored in the public GCS bucket `gs://spark-rapids-ml-benchmarking/datasets`.  The benchmark scripts are currently configured to read the data from there.

## Setup

- Install the [gcloud CLI](https://cloud.google.com/sdk/docs/install) and initialize it via `gcloud init`.

- Configure the following settings:
  ```
  export PROJECT=<your_project>
  export DATAPROC_REGION=<your_dataproc_region>
  export COMPUTE_REGION=<your_compute_region>
  export COMPUTE_ZONE=<your_compute_zone>

  gcloud config set project ${PROJECT}
  gcloud config set dataproc/region ${DATAPROC_REGION}
  gcloud config set compute/region ${COMPUTE_REGION}
  gcloud config set compute/zone ${COMPUTE_ZONE}
  ```

- Create a GCS bucket if you don't already have one:
  ```
  export GCS_BUCKET=<your_gcs_bucket_name>

  gcloud storage buckets create gs://${GCS_BUCKET}
  ```

- Upload the benchmarking files to your GCS bucket:
  ```
  # path to store benchmarking files inside your GCS bucket
  export BENCHMARK_HOME=${GCS_BUCKET}/benchmark

  ./setup.sh
  ```
  **Note**: this step should be repeated for each new version of the spark-rapids-ml package that you want to test.

## Run Benchmarks

- Start the cpu or gpu cluster and run all benchmarks:
  ```
  ./run_benchmark.sh [cpu|gpu] 2>&1 | tee benchmark.log
  ```
  **Note**: the created cluster is configured to automatically terminate after 30 minutes of idle time, but it can still be manually terminated or deleted via the Dataproc UI.

  **Note**: monitor benchmark progress periodically in case of a possible hang, to avoid incurring cloud costs in such cases.

- Extract timing information:
  ```
  egrep -e "[0-9.]* seconds" *.out
  ```

- Stop the cluster via the Dataproc UI, or via this command line:
  ```
  gcloud dataproc clusters delete ${USER}-spark-rapids-ml-[cpu|gpu] --region ${COMPUTE_REGION}
  ```

- **OPTIONAL**: To run a single benchmark manually, search the `benchmark.log` for the `gcloud` command line associated with the target benchmark.  If needed, start the cluster first.  Then, just copy-and-paste that command line into your shell.
  ```
  ./start_cluster.sh [cpu|gpu]
  <paste gcloud command line for benchmark>
  ```
