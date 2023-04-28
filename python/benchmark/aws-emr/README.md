# Benchmarking on Dataproc

This directory contains shell scripts for running larger scale benchmarks on a AWS EMR cluster. You will need a AWS EMR account to run them.  The benchmarks use datasets synthetically generated using [gen_data.py](../gen_data.py). For convenience, these have been precomputed and currently stored in the public S3 bucket `spark-rapids-ml-bm-datasets-public`.  The benchmark scripts are currently configured to read the data from there.

## Setup

- Install the [AWS CLI](https://docs.aws.amazon.com/emr/latest/EMR-on-EKS-DevelopmentGuide/setting-up-cli.html) and initialize it via `aws configure`. You may need to obtain your [access keys and region code](../../../notebooks/aws-emr/README.md).  

- Create an S3 bucket if you don't already have one.
  ```
  export S3_BUCKET=<your_s3_bucket_name>
  aws s3 mb s3://${S3_BUCKET}
  ```

- Upload the benchmarking files to your S3 bucket:
  ```
  # path to store benchmarking files inside your S3 bucket
  export BENCHMARK_HOME=${S3_BUCKET}/benchmark

  ./setup.sh
  ```
  **Note**: this step should be repeated for each new version of the spark-rapids-ml package that you want to test.

## Prepare Subnet 
- Print out available subnets in CLI then pick a SubnetId of your region (e.g. subnet-0744566f of AvailabilityZone us-east-2a in region Ohio). Subnet is required to start a EMR cluster.

  ```
  aws ec2 describe-subnets
  export SUBNET_ID=<your_SubnetId>
  ```
## Run Benchmarks

- Start the cpu or gpu cluster and run all benchmarks.
  ```
  ./run_benchmark.sh [cpu|gpu] 2>&1 | tee benchmark.log
  ```
  **Note**: monitor benchmark progress periodically in case of a possible hang, to avoid incurring cloud costs in such cases.

- Extract timing information.
  ```
  egrep -e "[0-9.]* seconds" benchmark.log
  ```

- Stop the cluster via the AWS EMR Console, or via command line. cluster\_id is available in benchmark.log. 
  ```
  cluster_id=$(grep "cluster-id" benchmark.log | grep -o 'j-[0-9|A-Z]*')
  aws emr terminate-clusters --cluster_ids ${cluster_id}
  ```
- **OPTIONAL**: To run a single benchmark manually, search the `benchmark.log` for the `aws emr add-steps` command line associated with the target benchmark.  If needed, start the cluster first and obtain a cluster_id.  Then, just copy-and-paste that command line into your shell and replace the cluster_id.
  ```
  ./start_cluster.sh [cpu|gpu]
  <paste aws emr command line and replace the cluster_id for benchmark>
  ```
