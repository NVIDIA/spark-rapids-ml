# Benchmarking on AWS EMR

This directory contains shell scripts for running larger-scale benchmarks on an AWS EMR cluster. You will need an AWS account to run them.  The benchmarks use datasets synthetically generated using [gen_data.py](../gen_data.py). For convenience, these have been precomputed and are available in the public S3 bucket `spark-rapids-ml-bm-datasets-public`.  The benchmark scripts are currently configured to read the data from there.

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

## Create an ssh key pair
- The benchmark script needs ssh access to the EMR cluster and this requires creating an [EC2 key pair](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/create-key-pairs.html).  Choose the **pem** format.  After saving the private key locally with `.pem` as the file extension, set the following environment variable to point to its location.
  ```
  export KEYPAIR=/path/to/private/key.pem
  ```

## Prepare Subnet 
- Print out available subnets in CLI then pick a SubnetId of your region (e.g. subnet-0744566f of AvailabilityZone us-east-2a in region Ohio). A subnet is required to start an EMR cluster.  Make sure that your selected subnet allows SSH access (port 22) from your local host where you will be invoking the benchmarking script.  The public subnet in the default VPC in your account might be a suitable choice.   See AWS EMR documentation for more info on [VPCs for EMR](https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-vpc-host-job-flows.html) and related info on SSH access in [managed security groups used by EMR](https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-man-sec-groups.html).

  ```
  aws ec2 describe-subnets
  export SUBNET_ID=<your_SubnetId>
  ```
## Run Benchmarks

- Start the cpu or gpu cluster and run all benchmarks.
  ```
  ./run_benchmark.sh [cpu|gpu] 2>&1 | tee benchmark.log
  ```
  **Note**: the created cluster is configured to automatically terminate after 30 minutes of idle time, but it can still be manually terminated or deleted via the AWS EMR Console.

  **Note**: monitor benchmark progress periodically in case of a possible hang, to avoid incurring cloud costs in such cases.

- Extract timing information. To view the original EMR log files, please log in [AWS EMR console](https://console.aws.amazon.com/emr/). Click "Clusters", choose the created cluster, click "Steps", then click "stdout" of each spark submit application.  
  ```
  egrep -e "[0-9.]* seconds" *.log
  ```

- Stop the cluster via the AWS EMR Console, or via command line. 
  ```
  cluster_id=$(grep "cluster-id" benchmark.log | grep -o 'j-[0-9|A-Z]*' | head -n 1)
  aws emr terminate-clusters --cluster-ids ${cluster_id}
  ```
- **OPTIONAL**: To run a single benchmark manually, search the `benchmark.log` for the `aws emr add-steps` command line associated with the target benchmark. If needed, start the cluster first and obtain its cluster_id. Then, just copy-and-paste that command line into your shell with the correct cluster_id.
  ```
  ./start_cluster.sh [cpu|gpu]
  <paste aws emr command line with the correct cluster_id for benchmark>
  ```
