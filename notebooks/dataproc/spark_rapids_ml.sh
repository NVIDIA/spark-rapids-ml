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

RAPIDS_VERSION=25.10.0


# install cudf and cuml
pip install --upgrade pip
pip install cudf-cu12~=${RAPIDS_VERSION} cuml-cu12~=${RAPIDS_VERSION} cuvs-cu12~=${RAPIDS_VERSION} \
    --extra-index-url=https://pypi.nvidia.com

# install spark-rapids-ml
pip install spark-rapids-ml

# set up no-import-change for cluster if enabled
no_import_change=$(/usr/share/google/get_metadata_value attributes/spark-rapids-ml-no-import-enabled)
if [[ $no_import_change == 1 ]]; then
    echo "enabling no import change in cluster" 1>&2
    mkdir -p /root/.ipython/profile_default/startup
    echo "import spark_rapids_ml.install" >/root/.ipython/profile_default/startup/00-spark-rapids-ml.py
fi
