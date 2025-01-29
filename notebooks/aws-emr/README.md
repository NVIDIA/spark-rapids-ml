## Running notebooks on AWS EMR 

If you already have a AWS EMR account, you can run the example notebooks on an EMR cluster, as follows:
- Install the [AWS CLI](https://docs.aws.amazon.com/emr/latest/EMR-on-EKS-DevelopmentGuide/setting-up-cli.html).
- Initialize the CLI via `aws configure`. You may need to create access keys by following [Authenticating using IAM user credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-authentication-user.html). You can find your default region name (e.g. Ohio) on the right of the top navigation bar. Clicking the region name will show the region code (e.g. us-east-2 for Ohio). 
  ```
  aws configure
  AWS Access Key ID [None]: <your_access_key>
  AWS Secret Access Key [None]: <your_secret_access_key>
  Default region name [None]: <region-code>
  Default output format [None]: <json>
  ```
- Create an S3 bucket if you don't already have one.
  ```
  export S3_BUCKET=<your_s3_bucket_name>
  aws s3 mb s3://${S3_BUCKET}
  ```
- Upload the initialization script to S3.
  ```
  aws s3 cp init-bootstrap-action.sh s3://${S3_BUCKET}/
  ```
- Print out available subnets in CLI then pick a SubnetId (e.g. subnet-0744566f of AvailabilityZone us-east-2a).

  ```
  aws ec2 describe-subnets
  export SUBNET_ID=<your_SubnetId>
  ```
  
  If this is your first time using EMR notebooks via EMR Studio and EMR Workspaces, we recommend creating a fresh VPC and subnets with internet access (the initialization script downloads artifacts) meeting the EMR requirements, per EMR documentation, and then specifying one of the new subnets in the above.

- Create a cluster with at least two single-gpu workers. You will obtain a ClusterId in terminal. Noted three GPU nodes are requested here, because EMR cherry picks one node (either CORE or TASK) to run JupyterLab service for notebooks and will not use the node for compute.
  
  If you wish to also enable [no-import-change](../README.md#no-import-change) UX for the cluster, change the init script argument `Args=[--no-import-enabled,0]` to `Args=[--no-import-enabled,1]` below.   The init script `init-bootstrap-action.sh` checks this argument and modifies the runtime accordingly.

  ```
  export CLUSTER_NAME="spark_rapids_ml"
  export CUR_DIR=$(pwd)

  aws emr create-cluster \
  --name ${CLUSTER_NAME} \
  --release-label emr-7.3.0 \
  --ebs-root-volume-size=32 \
  --applications Name=Hadoop Name=Livy Name=Spark Name=JupyterEnterpriseGateway \
  --service-role EMR_DefaultRole \
  --log-uri s3://${S3_BUCKET}/logs \
  --ec2-attributes SubnetId=${SUBNET_ID},InstanceProfile=EMR_EC2_DefaultRole \
  --instance-groups InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m4.2xlarge \
                    InstanceGroupType=CORE,InstanceCount=3,InstanceType=g4dn.2xlarge \
  --configurations file://${CUR_DIR}/init-configurations.json \
  --bootstrap-actions Name='Spark Rapids ML Bootstrap action',Path=s3://${S3_BUCKET}/init-bootstrap-action.sh,Args=[--no-import-enabled,0]
  ```
- In the [AWS EMR console](https://console.aws.amazon.com/emr/), click "Clusters", you can find the cluster id of the created cluster. Wait until the cluster has the "Waiting" status. 
- To use notebooks on EMR you will need an EMR Studio and an associated Workspace.   If you don't already have these, in the [AWS EMR console](https://console.aws.amazon.com/emr/), on the left, in the "EMR Studio" section, click the respective "Studio" and "Workspace (Notebooks)" links and follow instructions to create them.  When creating a Studio, select the `Custom` setup option to allow for configuring a VPC and a Subnet.  These should match the VPC and Subnet used for the cluster.  Select "\*Default\*" for all security group prompts and drop downs for Studio and Workspace setting.  Please check/search EMR documentation for further instructions. 

- In the "Workspace (Notebooks)" list of workspaces, select the created workspace, make sure it has the "Idle" status (select "Stop" otherwise in the "Actions" drop down) and click "Attach" to attach the newly created cluster through cluster id to the workspace.

- Use the default notebook or create/upload a new notebook. Set the notebook kernel to "PySpark".  For the no-import-change UX, you can try the example [kmeans-no-import-change.ipynb](../kmeans-no-import-change.ipynb).

- Run the notebook cells.  
  **Note**: these settings are for demonstration purposes only.  Additional tuning may be required for optimal performance.
