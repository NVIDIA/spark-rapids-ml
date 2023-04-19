## Running notebooks on AWS EMR 

If you already have a Dataproc account, you can run the example notebooks on a EMR cluster, as follows:
- Install the [AWS CLI](https://docs.aws.amazon.com/emr/latest/EMR-on-EKS-DevelopmentGuide/setting-up-cli.html).
- initialize the CLI via `aws configure`. You may need to create access keys by following [Authenticating using IAM user credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-authentication-user.html). You can find your default region name (e.g. Ohio) on the right of the top navigation bar. Clicking the region name will show the region code (e.g. us-east-2 for Ohio). 
  ```
  aws configure
  AWS Access Key ID [None]: <your_access_key>
  AWS Secret Access Key [None]: <your_secret_access_key>
  Default region name [None]: <region-code>
  Default output format [None]: <json>
  ```
- Create a S3 bucket if you don't already have one.
  ```
  export S3_BUCKET=<your_gcs_bucket_name>

  aws s3 mb s3://${S3_BUCKET}
  ```
- Create a zip file for the `spark-rapids-ml` package
  ```
  cd spark-rapids-ml/python/src
  zip -r spark_rapids_ml.zip spark_rapids_ml
  ```
- Upload the zip file and the initialization script to S3.
  ```
  aws s3 cp init-bootstrap-action.sh s3://${S3_BUCKET}/init-bootstrap-action.sh
  aws s3 cp spark_rapids_ml.zip s3://${S3_BUCKET}/spark_rapids_ml.zip
  ```
- Print out available subnets in CLI then pick a SubnetId (e.g. subnet-0744566f of AvailabilityZone us-east-2a) 

  ```
  aws ec2 describe-subnets
  export SUBNET_ID=<subnet-0744566f>
  ```

- Create a cluster with at least two single-gpu workers.

  ```

  export CLUSTER_NAME="spark_rapids_ml"
  export CUR_DIR=$(pwd)

  aws emr create-cluster \
  --name ${CLUSTER_NAME} \
  --release-label emr-6.10.0 \
  --applications Name=Hadoop Name=Livy Name=Spark Name=JupyterEnterpriseGateway \
  --service-role EMR_DefaultRole \
  --ec2-attributes SubnetId=${SUBNET_ID},InstanceProfile=EMR_EC2_DefaultRole \
  --instance-groups InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m4.10xlarge \
                    InstanceGroupType=CORE,InstanceCount=1,InstanceType=g4dn.2xlarge \
                    InstanceGroupType=TASK,InstanceCount=1,InstanceType=g4dn.2xlarge \
                    InstanceGroupType=TASK,InstanceCount=1,InstanceType=g4dn.2xlarge \
  --configurations file://${CUR_DIR}/init-configurations.json \
  --bootstrap-actions Name='Spark Rapids ML Bootstrap action',Path=s3://${S3_BUCKET}/init-bootstrap-action.sh
  ```
- In the [AWS EMR console](https://console.aws.amazon.com/emr/), click "Clusters", you can find the cluster id of the new created cluster. 
  ```
  export CLUSTER_ID=<cluster id>
  ```
- In the [AWS EMR console](https://console.aws.amazon.com/emr/), click "Workspace(Notebooks)", then creaet a workspace. Wait until the status becomes ready. 

- Click to enter the created workspace. Click the "Cluster" button (usually the top second button of the left navigation bar). Attach the workspace to the newly created EC2 cluster.     

- Use the default notebook or create/upload a new notebook. Set the notebook kernel to "PySpark".  

- Add the following to a new cell at the beginning of the notebook. Replace "s3 path to spark\_rapids\_ml.zip" with the actual s3 path.  
  ```
  %%configure -f
  {
      "conf":{
            "spark.pyspark.python": "python3.9",
            "spark.dynamicAllocation.enabled":false,
            "spark.submit.pyFiles": "s3 path to spark_rapids_ml.zip"
      }
  }
  
  ```
  **Note**: these settings are for demonstration purposes only.  Additional tuning may be required for optimal performance.
- Run the notebook cells.  
