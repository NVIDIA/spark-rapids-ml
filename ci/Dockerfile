#
# Copyright (c) 2023, NVIDIA CORPORATION.
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

ARG CUDA_VERSION=11.5.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

# Install packages to build spark-rapids-ml
RUN apt update -y \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y openjdk-8-jdk \
    && apt install -y git numactl software-properties-common wget zip \
    && rm -rf /var/lib/apt/lists

# Config JAVA_HOME
ENV JAVA_HOME /usr/lib/jvm/java-1.8.0-openjdk-amd64

# Install conda
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py38_4.10.3-Linux-x86_64.sh \
    && conda init

# install cuML
ARG CUML_VER=23.06
RUN conda install -c conda-forge mamba && \
    mamba install -y -c rapidsai -c nvidia -c conda-forge cuml=$CUML_VER python=3.9 cuda-toolkit=11.5 \
    && mamba clean --all -f -y
