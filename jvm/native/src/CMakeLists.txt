#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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
#=============================================================================


# Install cuDF nightly via Conda, only for local development, will remove in CI.
find_package(cudf)

set (CMAKE_CUDA_FLAGS "--extended-lambda")

if(DEFINED ENV{RAFT_PATH})
  message(STATUS "RAFT_PATH environment variable detected.")
  message(STATUS "RAFT_DIR set to $ENV{RAFT_PATH}")
  set(RAFT_DIR ENV{RAFT_PATH})

else(DEFINED ENV{RAFT_PATH})
  message(STATUS "RAFT_PATH environment variable NOT detected, cloning RAFT")
  set(RAFT_GIT_DIR ${CMAKE_CURRENT_BINARY_DIR}/raft CACHE STRING "Path to RAFT repo")

  ExternalProject_Add(raft
    GIT_REPOSITORY    git@github.com:rapidsai/raft.git
    GIT_TAG           pinned_commit/git_tag/branch
    PREFIX            ${RAFT_GIT_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   "")

  set(RAFT_INCLUDE_DIR ${RAFT_GIT_DIR}/src/raft/cpp/include CACHE STRING "RAFT include variable")
endif(DEFINED ENV{RAFT_PATH})



#################################################################################################
# - CPM -----------------------------------------------------------------------------------------

set(CPM_DOWNLOAD_VERSION 0.27.2)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake")
    file(DOWNLOAD https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()

include(${CPM_DOWNLOAD_LOCATION})
#################################################################################################

# pull cuDF sources, to use jni_utils.hpp
# cmake options should be added here for CI build.
CPMAddPackage(NAME cudf
        VERSION         "22.04.00"
        GIT_REPOSITORY  https://github.com/rapidsai/cudf.git
        GIT_TAG         branch-22.04
)

add_library(rapidsml_jni SHARED rapidsml_jni.cpp
                                rapidsml_jni.cu
                                )

target_link_options(rapidsml_jni PRIVATE "-Wl,-as-needed")

target_include_directories(rapidsml_jni PRIVATE
                                    "${JNI_INCLUDE_DIRS}"
                                    "${CUDAToolkit_INCLUDE_DIRS}"
                                    "$ENV{RAFT_PATH}/cpp/include"
                                    "${CMAKE_SOURCE_DIR}/src/utils"
                                    "${cudf_SOURCE_DIR}/java/src/main/native/include"
                                    )

target_link_libraries(rapidsml_jni PRIVATE
                      libcudart_static.a
                      libcusparse_static.a
                      libcusolver_static.a
                      libculibos.a
                      liblapack_static.a
                      CUDA::cublas
                      cudf::cudf)

if(PER_THREAD_DEFAULT_STREAM)
  target_compile_definitions(rapidsml_jni PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)
endif()

