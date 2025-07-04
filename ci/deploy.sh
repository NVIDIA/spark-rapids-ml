#!/bin/bash
#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
#

set -ex

# build plugin jar
pushd jvm
mvn clean package -DskipTests
popd

# copy plugin jar to python package
JARS_DIR=python/src/spark_rapids_ml/jars
mkdir -p $JARS_DIR
rm -f $JARS_DIR/*.jar
cp jvm/target/*.jar $JARS_DIR

# build whl package
pushd python
pip install -r requirements_dev.txt && pip install -e .
python -m build
popd
