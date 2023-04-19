#!/bin/bash

set -ex
 
sudo chmod a+rwx -R /sys/fs/cgroup/cpu,cpuacct
sudo chmod a+rwx -R /sys/fs/cgroup/devices

sudo yum install -y gcc openssl-devel bzip2-devel libffi-devel tar gzip wget make mysql-devel
sudo bash -c "wget https://www.python.org/ftp/python/3.9.9/Python-3.9.9.tgz && tar xzf Python-3.9.9.tgz && cd Python-3.9.9 && ./configure --enable-optimizations && make altinstall"

RAPIDS_VERSION=23.2.0

# upgrade pip
sudo /usr/local/bin/pip3.9 install --upgrade pip

# patch existing
#python3 -m pip install --ignore-installed "llvmlite<0.40,>=0.39.0dev0" "numba>=0.56.2"

# install cudf and cuml
sudo /usr/local/bin/pip3.9 install --no-cache-dir cudf-cu11==${RAPIDS_VERSION} cuml-cu11==${RAPIDS_VERSION} --extra-index-url=https://pypi.nvidia.com

# install spark-rapids-ml
#sudo /usr/local/bin/pip3.9 install spark-rapids-ml

#export LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/compat/lib:/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/lib/hadoop/lib/native:/usr/lib/hadoop-lzo/lib/native:/docker/usr/lib/hadoop/lib/native:/docker/usr/lib/hadoop-lzo/lib/native"

#export CUPY_CACHE_DIR="~/.cupy/kernel_cache"
#sudo mkdir /home/.cupy
#sudo chmod -R 666 /home/.cupy
