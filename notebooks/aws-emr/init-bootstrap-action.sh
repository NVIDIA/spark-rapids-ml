#!/bin/bash

set -ex
 
sudo chmod a+rwx -R /sys/fs/cgroup/cpu,cpuacct
sudo chmod a+rwx -R /sys/fs/cgroup/devices

sudo yum install -y gcc openssl-devel bzip2-devel libffi-devel tar gzip wget make mysql-devel
sudo bash -c "wget https://www.python.org/ftp/python/3.9.9/Python-3.9.9.tgz && tar xzf Python-3.9.9.tgz && cd Python-3.9.9 && ./configure --enable-optimizations && make altinstall"

RAPIDS_VERSION=23.6.0

# install scikit-learn 
sudo /usr/local/bin/pip3.9 install scikit-learn

# install cudf and cuml
sudo /usr/local/bin/pip3.9 install --no-cache-dir cudf-cu11==${RAPIDS_VERSION} \
    cuml-cu11==${RAPIDS_VERSION} \
    pylibraft-cu11==${RAPIDS_VERSION} \
    rmm-cu11==${RAPIDS_VERSION} \
    --extra-index-url=https://pypi.nvidia.com

