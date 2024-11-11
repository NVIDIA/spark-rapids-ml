#!/bin/bash

set -ex

sudo mkdir -p /spark-rapids-cgroup/devices
sudo mount -t cgroup -o devices cgroupv1-devices /spark-rapids-cgroup/devices
sudo chmod a+rwx -R /spark-rapids-cgroup

sudo yum update -y
sudo yum install -y gcc bzip2-devel libffi-devel tar gzip wget make 
sudo yum install -y mysql-devel --skip-broken
sudo bash -c "wget https://www.python.org/ftp/python/3.10.9/Python-3.10.9.tgz && \
tar xzf Python-3.10.9.tgz && cd Python-3.10.9 && \
./configure --enable-optimizations && make altinstall"

RAPIDS_VERSION=24.10.0

sudo /usr/local/bin/pip3.10 install --upgrade pip

# install scikit-learn 
sudo /usr/local/bin/pip3.10 install scikit-learn

# install cudf and cuml
sudo /usr/local/bin/pip3.10 install --no-cache-dir cudf-cu12 --extra-index-url=https://pypi.nvidia.com --verbose
sudo /usr/local/bin/pip3.10 install --no-cache-dir cuml-cu12 cuvs-cu12 --extra-index-url=https://pypi.nvidia.com --verbose

sudo /usr/local/bin/pip3.10 list

