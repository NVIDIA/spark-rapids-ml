#!/bin/bash
cluster_type=${1:-gpu}

# configure arguments
if [[ -z ${SUBNET_ID} ]]; then
    echo "Please export SUBNET_ID per README.md"
    exit 1
fi

if [[ -z ${BENCHMARK_HOME} ]]; then
    echo "Please export BENCHMARK_HOME per README.md"
    exit 1
fi

cluster_name=spark-rapids-ml-${cluster_type}
cur_dir=$(pwd)

if [[ ${cluster_type} == "gpu" ]]; then
    core_type=g4dn.2xlarge
    config_json="file://${cur_dir}/../../../notebooks/aws-emr/init-configurations.json"
elif [[ ${cluster_type} == "cpu" ]]; then
    core_type=m4.2xlarge
    config_json="file://${cur_dir}/cpu-init-configurations.json"
else
    echo "unknown cluster type ${cluster_type}"
    echo "usage: ./${script_name} cpu|gpu"
    exit 1
fi

start_cmd="aws emr create-cluster \
--name ${cluster_name} \
--release-label emr-6.10.0 \
--applications Name=Hadoop Name=Spark \
--service-role EMR_DefaultRole \
--log-uri s3://${BENCHMARK_HOME}/logs \
--ec2-attributes SubnetId=${SUBNET_ID},InstanceProfile=EMR_EC2_DefaultRole \
--instance-groups InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m4.2xlarge \
                  InstanceGroupType=CORE,InstanceCount=3,InstanceType=${core_type} \
--configurations ${config_json} \
--bootstrap-actions Name='Spark Rapids ML Bootstrap action',Path=s3://${BENCHMARK_HOME}/init-bootstrap-action.sh
"

CLUSTER_ID=$(eval ${start_cmd} | tee /dev/tty | grep "ClusterId" | grep -o 'j-[0-9|A-Z]*')
aws emr put-auto-termination-policy --cluster-id ${CLUSTER_ID} --auto-termination-policy IdleTimeout=1800
echo "${CLUSTER_ID}"
