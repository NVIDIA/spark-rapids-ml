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


###
#
# The image to mvn build jars of GitHub/nvidia/spark-rapids-ml
#
# Arguments: CUDA_VERSION=11.5.x (CUDA Toolkit(>=11.5))
# Arguments: GCC_VERSION=9 (gcc(>=9.3))
# Arguments: CMAKE_VERSION=3.20.x (cmake(>=3.20))
# Arguments: NINJA_VERSION=1.10.x (ninja(>=1.10))
# Arguments: MAVEN_VERSION=3.8.6 (maven(>=3.3.9))
#
# Example to build & upload image :
#    docker build -t spark-rapids-ml:cuda11.5.2 . -f Dockerfile \
#        --build-arg CUDA_VERSION=11.5.2 \
#        --build-arg GCC_VERSION=9 \
#        --build-arg CMAKE_VERSION=3.23.3 \
#        --build-arg NINJA_VERSION=1.10.2 \
#        --build-arg MAVEN_VERSION=3.8.6
#
###

ARG CUDA_VERSION=11.5.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

# Install packages to build spark-rapids-ml jars
RUN apt update -y && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y openjdk-8-jdk wget git zip

# Install gcc & g++
ARG GCC_VERSION=9
RUN apt install -y software-properties-common \
    && add-apt-repository -y ppa:git-core/ppa \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt update -y \
    && apt install -y gcc-${GCC_VERSION} g++-${GCC_VERSION} \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} 100

# Install cmake
ARG CMAKE_VERSION=3.23.3
RUN cd /usr/local/ && wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && \
   tar zxf cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && \
   rm -rf cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz
ENV PATH /usr/local/cmake-${CMAKE_VERSION}-linux-x86_64/bin:$PATH

# Install ninja
ARG NINJA_VERSION=1.10.2
RUN cd /usr/local/ && wget -q https://github.com/ninja-build/ninja/releases/download/v${NINJA_VERSION}/ninja-linux.zip && \
    mkdir -p /usr/local/ninja && unzip -d /usr/local/ninja ninja-linux.zip && rm -rf ninja-linux.zip
ENV PATH /usr/local/ninja:$PATH

# Config JAVA_HOME
ENV JAVA_HOME /usr/lib/jvm/java-1.8.0-openjdk-amd64

# Donwload maven
ARG MAVEN_VERSION=3.8.8
RUN cd /usr/local/ && wget -q https://dlcdn.apache.org/maven/maven-3/$MAVEN_VERSION/binaries/apache-maven-$MAVEN_VERSION-bin.zip && \
     unzip apache-maven-$MAVEN_VERSION-bin.zip && rm -rf apache-maven-$MAVEN_VERSION-bin.zip
ENV PATH /usr/local/apache-maven-$MAVEN_VERSION/bin:$PATH

# Install conda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget --quiet \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py38_4.10.3-Linux-x86_64.sh \
    && conda init

# install cuDF dependency, Fall back to use cudf 22.04 due to issue:
# https://github.com/NVIDIA/spark-rapids-ml/issues/73
ARG CONDA_CUDF_VER=22.04
RUN conda install -c rapidsai -c conda-forge cudf=$CONDA_CUDF_VER python=3.8 -y

# Note: the raft verion is fixed to 22.12, do not modify it when updating the spark-rapids-ml version.
#       newer versions may fail the build process due to API incompatibility.
ARG RAFT_VER=22.12
RUN git clone -b branch-$RAFT_VER https://github.com/rapidsai/raft.git
ENV RAFT_PATH=/raft

### END OF CACHE ###

#ARG RAPIDS_ML_VER=23.04
#RUN git clone -b branch-$RAPIDS_ML_VER https://github.com/NVIDIA/spark-rapids-ml.git
COPY . /spark-rapids-ml
WORKDIR /spark-rapids-ml/jvm

SHELL ["conda", "run", "--no-capture-output", "-n", "base", "/bin/bash", "-c"]
