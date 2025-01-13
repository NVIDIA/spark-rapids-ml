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


if ! command -v nvidia-smi &> /dev/null
then
    # default to the first GPU
    echo "{\"name\":\"gpu\",\"addresses\":[\"0\"]}"
    exit
else
    # https://github.com/apache/spark/blob/master/examples/src/main/scripts/getGpusResources.sh
    ADDRS=`nvidia-smi --query-gpu=index --format=csv,noheader | sed -e ':a' -e 'N' -e'$!ba' -e 's/\n/","/g'`
    echo {\"name\": \"gpu\", \"addresses\":[\"$ADDRS\"]}
fi