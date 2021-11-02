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
find_package(rmm 21.12.00)
find_package(Thrust)

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

###################################################################################################

####################################################################################################
# - cudf -------------------------------------------------------------------------------------------

# Ensure CUDA runtime is dynamic despite statically linking Arrow in libcudf
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

CPMAddPackage(NAME  cudf
        VERSION         "21.12.00"
        GIT_REPOSITORY  https://github.com/rapidsai/cudf.git
        GIT_TAG         branch-21.12
        GIT_SHALLOW     TRUE
        SOURCE_SUBDIR   cpp
        OPTIONS         "BUILD_TESTS OFF"
                        "BUILD_BENCHMARKS OFF"
                        "CUDF_USE_ARROW_STATIC ON"
                        "JITIFY_USE_CACHE ON"
                        "CUDA_STATIC_RUNTIME OFF"
                        "DISABLE_DEPRECATION_WARNING ON"
                        "AUTO_DETECT_CUDA_ARCHITECTURES OFF"
                        "CMAKE_CUDA_ARCHITECTURES ALL"
                        "CUDF_ENABLE_ARROW_S3 OFF"
    )

###################################################################################################


add_library(rapidsml_jni SHARED rapidsml_jni.cu)
target_include_directories(rapidsml_jni PRIVATE "$ENV{RMM_PATH}/include/rmm"
                                    "${JNI_INCLUDE_DIRS}"
                                    "${CUDAToolkit_INCLUDE_DIRS}"
                                    "$ENV{RAFT_PATH}/cpp/include"
                                    "${CMAKE_SOURCE_DIR}/src/utils"
                                    )

target_link_libraries(rapidsml_jni PRIVATE CUDA::cusolver CUDA::cudart CUDA::cublas rmm::rmm cudf::cudf)
if(PER_THREAD_DEFAULT_STREAM)
  target_compile_definitions(rapidsml_jni PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)
endif()

