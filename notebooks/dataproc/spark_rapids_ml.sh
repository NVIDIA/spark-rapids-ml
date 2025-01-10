#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.
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


RAPIDS_VERSION=24.12.0

# patch existing packages
mamba install "llvmlite<0.40,>=0.39.0dev0" "numba>=0.56.2"

# dataproc 2.1 pyarrow and arrow conda installation is not compatible with cudf
mamba uninstall -y pyarrow arrow

# install cudf and cuml
pip install --upgrade pip
pip install cudf-cu12~=${RAPIDS_VERSION} cuml-cu12~=${RAPIDS_VERSION} cuvs-cu12~=${RAPIDS_VERSION} \
    pylibraft-cu12~=${RAPIDS_VERSION} \
    rmm-cu12~=${RAPIDS_VERSION} \
    --extra-index-url=https://pypi.nvidia.com

# install spark-rapids-ml
pip install spark-rapids-ml
