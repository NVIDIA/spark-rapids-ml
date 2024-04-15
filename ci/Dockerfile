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

ARG CUDA_VERSION=11.8.0
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
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda init && conda update -n base conda \
    && conda install -n base conda-libmamba-solver \
    && conda config --set solver libmamba

# install cuML
ARG CUML_VER=24.04
RUN conda install -y -c rapidsai -c conda-forge -c nvidia cuml=$CUML_VER python=3.9 cuda-version=11.8 \
    && conda clean --all -f -y
