#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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

RAPIDS_VERSION=25.12.0

sudo /usr/local/bin/pip3.10 install --upgrade pip

# install scikit-learn 
sudo /usr/local/bin/pip3.10 install scikit-learn

# install cudf and cuml
sudo /usr/local/bin/pip3.10 install --no-cache-dir \
         cudf-cu12~=${RAPIDS_VERSION} \
         cuml-cu12~=${RAPIDS_VERSION} \
         cuvs-cu12~=${RAPIDS_VERSION} \
         pylibraft-cu12~=${RAPIDS_VERSION} \
         raft-dask-cu12~=${RAPIDS_VERSION} \
         dask-cuda-cu12~=${RAPIDS_VERSION} \
         --extra-index-url=https://pypi.nvidia.com --verbose
sudo /usr/local/bin/pip3.10 install spark-rapids-ml
sudo /usr/local/bin/pip3.10 list

# set up no-import-change for cluster if enabled
if [[ $1 == "--no-import-enabled" && $2 == 1 ]]; then
    echo "enabling no import change in cluster" 1>&2
    cd /usr/lib/livy/repl_2.12-jars
    sudo jar xf livy-repl_2.12*.jar fake_shell.py
    sudo sed -i fake_shell.py -e '/from __future__/ s/\(.*\)/\1\ntry:\n    import spark_rapids_ml.install\nexcept:\n    pass\n/g'
    sudo jar uf livy-repl_2.12*.jar fake_shell.py
    sudo rm fake_shell.py
fi 

# ensure notebook comes up in python 3.10 by using a background script that waits for an 
# application file to be installed before modifying.
cat <<EOF >/tmp/mod_start_kernel.sh
#!/bin/bash
set -ex
while [ ! -f /mnt/notebook-env/bin/start_kernel_as_emr_notebook.sh ]; do
echo "waiting for /mnt/notebook-env/bin/start_kernel_as_emr_notebook.sh"
sleep 10
done
echo "done waiting"
sleep 10
sudo sed -i /mnt/notebook-env/bin/start_kernel_as_emr_notebook.sh -e 's#"spark.pyspark.python": "python3"#"spark.pyspark.python": "/usr/local/bin/python3.10"#g'
sudo sed -i /mnt/notebook-env/bin/start_kernel_as_emr_notebook.sh -e 's#"spark.pyspark.virtualenv.enabled": "true"#"spark.pyspark.virtualenv.enabled": "false"#g'
exit 0
EOF
sudo bash /tmp/mod_start_kernel.sh &
exit 0

