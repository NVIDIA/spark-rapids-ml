/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>
#include <jni.h>
#include <cusolverDn.h>
#include <assert.h>
#include <iostream>
#include <raft/linalg/cublas_wrappers.h>
#include <raft/linalg/eig.cuh>
#include <raft/linalg/svd.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/matrix/math.cuh>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <rmm/exec_policy.hpp>
#include <nvtx3.hpp>

struct java_domain {
  static constexpr char const *name{"Java"};
};

namespace {
void signFlip(
  double* input, int n_rows, int n_cols, double* components, int n_cols_comp, cudaStream_t stream)
{
  auto counting = thrust::make_counting_iterator(0);
  auto m        = n_rows;

  thrust::for_each(rmm::exec_policy(stream), counting, counting + n_cols, [=] __device__(int idx) {
    int d_i = idx * m;
    int end = d_i + m;

    double max    = 0.0;
    int max_index = 0;
    for (int i = d_i; i < end; i++) {
      double val = input[i];
      if (val < 0.0) { val = -val; }
      if (val > max) {
        max       = val;
        max_index = i;
      }
    }

    if (input[max_index] < 0.0) {
      for (int i = d_i; i < end; i++) {
        input[i] = -input[i];
      }
    }
  });
}

cublasOperation_t convertToCublasOpEnum(int int_type)
{
  if (int_type == 0) {
    return CUBLAS_OP_N;
  } else if (int_type == 1) {
    return CUBLAS_OP_T;
  } else if (int_type == 2) {
    return CUBLAS_OP_C;
  } else if (int_type == 3) {
    return CUBLAS_OP_CONJG;
  } else {
    throw "Invalid type enum: " + std::to_string(int_type);
  }
}
} // anonymous namespace

extern "C" {

  JNIEXPORT void JNICALL Java_com_nvidia_spark_ml_linalg_NvtxRange_push(JNIEnv *env, jclass clazz, jstring name,
                                                          jint color_bits) {
  jclass jlexception = env->FindClass("java/lang/Exception");
  try {
    // cudf::jni::native_jstring range_name(env, name);
    const char *range_name = env->GetStringUTFChars(name, 0);
    nvtx3::color range_color(static_cast<nvtx3::color::value_type>(color_bits));
    nvtx3::event_attributes attr{range_color, range_name};
    nvtxDomainRangePushEx(nvtx3::domain::get<java_domain>(), attr.get());
  } catch (const std::bad_alloc &e) {
    env->ThrowNew(jlexception, "Error nvtx push");
  }
  // CATCH_STD(env, );
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_ml_linalg_NvtxRange_pop(JNIEnv *env, jclass clazz) {
  jclass jlexception = env->FindClass("java/lang/Exception");
  try {
    nvtxDomainRangePop(nvtx3::domain::get<java_domain>());
  } catch (const std::bad_alloc &e) {
    env->ThrowNew(jlexception, "Error nvtx pop");
  }
  // CATCH_STD(env, );
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_ml_linalg_JniRAPIDSML_dspr(JNIEnv* env, jclass, jint n, jdoubleArray x,
                                                                      jdoubleArray A) {
  jclass jlexception = env->FindClass("java/lang/Exception");
  auto size_A = env->GetArrayLength(A);

  double* dev_x;
  auto cuda_error = cudaMalloc((void**)&dev_x, n * sizeof(double));
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error allocating device memory for x");
  }

  double* dev_A;
  cuda_error = cudaMalloc((void**)&dev_A, size_A * sizeof(double));
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error allocating device memory for A");
  }

  auto* host_x = env->GetDoubleArrayElements(x, nullptr);
  cuda_error = cudaMemcpyAsync(dev_x, host_x, n * sizeof(double), cudaMemcpyDefault);
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error copying x to device");
  }

  auto* host_A = env->GetDoubleArrayElements(A, nullptr);
  cuda_error = cudaMemcpyAsync(dev_A, host_A, size_A * sizeof(double), cudaMemcpyDefault);
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error copying A to device");
  }

  cublasHandle_t handle;
  auto status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    env->ThrowNew(jlexception, "Error creating cuBLAS handle");
  }

  double alpha = 1.0;
  status = cublasDspr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha, dev_x, 1, dev_A);
  if (status != CUBLAS_STATUS_SUCCESS) {
    env->ThrowNew(jlexception, "Error calling cublasDspr");
  }

  cuda_error = cudaMemcpyAsync(host_A, dev_A, size_A * sizeof(double), cudaMemcpyDefault);
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error copying A from device");
  }

  cuda_error = cudaFree(dev_x);
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error freeing x from device");
  }

  cuda_error = cudaFree(dev_A);
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error freeing A from device");
  }

  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    env->ThrowNew(jlexception, "Error destroying cuBLAS handle");
  }

  env->ReleaseDoubleArrayElements(x, host_x, JNI_ABORT);
  env->ReleaseDoubleArrayElements(A, host_A, 0);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_ml_linalg_JniRAPIDSML_dgemm(JNIEnv* env, jclass, jint transa, jint transb,
                                                                        jint m, jint n, jint k, jdouble alpha,
                                                                        jdoubleArray A, jint lda, jdoubleArray B,
                                                                        jint ldb, jdouble beta, jdoubleArray C, jint ldc, jint deviceID) {
  cudaSetDevice(deviceID);
  jclass jlexception = env->FindClass("java/lang/Exception");

  raft::handle_t raft_handle;
  cudaStream_t stream = raft_handle.get_stream();

  auto size_A = env->GetArrayLength(A);
  auto size_B = env->GetArrayLength(B);
  auto size_C = env->GetArrayLength(C);

  double* dev_A;
  auto cuda_error = cudaMalloc((void**)&dev_A, size_A * sizeof(double));
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error allocating device memory for A");
  }

  double* dev_B;
  cuda_error = cudaMalloc((void**)&dev_B, size_B * sizeof(double));
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error allocating device memory for B");
  }

  double* dev_C;
  cuda_error = cudaMalloc((void**)&dev_C, size_C * sizeof(double));
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error allocating device memory for C");
  }

  auto* host_A = env->GetDoubleArrayElements(A, nullptr);
  cuda_error = cudaMemcpyAsync(dev_A, host_A, size_A * sizeof(double), cudaMemcpyDefault);
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error copying A to device");
  }

  auto* host_B = env->GetDoubleArrayElements(B, nullptr);
  cuda_error = cudaMemcpyAsync(dev_B, host_B, size_B * sizeof(double), cudaMemcpyDefault);
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error copying B to device");
  }

  cublasHandle_t handle;
  auto status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    env->ThrowNew(jlexception, "Error creating cuBLAS handle");
  }

  status = raft::linalg::cublasgemm(handle, convertToCublasOpEnum(transa), convertToCublasOpEnum(transb), m, n, k, &alpha, dev_A, lda, dev_B, ldb, &beta,
                       dev_C, ldc, stream);

  if (status != CUBLAS_STATUS_SUCCESS) {
    env->ThrowNew(jlexception, "Error calling cublasDgemm");
  }

  auto* host_C = env->GetDoubleArrayElements(C, nullptr);
  cuda_error = cudaMemcpyAsync(host_C, dev_C, size_C * sizeof(double), cudaMemcpyDefault);
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error copying C from device");
  }

  cuda_error = cudaFree(dev_A);
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error freeing A from device");
  }

    cuda_error = cudaFree(dev_B);
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error freeing B from device");
  }

  cuda_error = cudaFree(dev_C);
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error freeing C from device");
  }

  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    env->ThrowNew(jlexception, "Error destroying cuBLAS handle");
  }

  env->ReleaseDoubleArrayElements(A, host_A, JNI_ABORT);
  env->ReleaseDoubleArrayElements(B, host_B, JNI_ABORT);
  env->ReleaseDoubleArrayElements(C, host_C, 0);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_ml_linalg_JniRAPIDSML_dgemm_1b(JNIEnv* env, jclass, jint rows_a, jint cols_b, jint cols_a,
                                                                       jdoubleArray A, jdoubleArray B, jdoubleArray C, jint deviceID) {

  cudaSetDevice(deviceID);
  jclass jlexception = env->FindClass("java/lang/Exception");
  auto size_A = env->GetArrayLength(A);
  auto size_B = env->GetArrayLength(B);
  auto size_C = env->GetArrayLength(C);

  double* dev_A;
  auto cuda_error = cudaMalloc((void**)&dev_A, size_A * sizeof(double));
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error allocating device memory for A");
  }

   double* dev_B;
   cuda_error = cudaMalloc((void**)&dev_B, size_B * sizeof(double));
   if (cuda_error != cudaSuccess) {
     env->ThrowNew(jlexception, "Error allocating device memory for A");
   }

  double* dev_C;
  cuda_error = cudaMalloc((void**)&dev_C, size_C * sizeof(double));
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error allocating device memory for C");
  }

  auto* host_A = env->GetDoubleArrayElements(A, nullptr);
  cuda_error = cudaMemcpyAsync(dev_A, host_A, size_A * sizeof(double), cudaMemcpyDefault);
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error copying A to device");
  }

  auto* host_B = env->GetDoubleArrayElements(B, nullptr);
  cuda_error = cudaMemcpyAsync(dev_B, host_B, size_B * sizeof(double), cudaMemcpyDefault);
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error copying A to device");
  }

  cublasHandle_t handle;
  auto status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    env->ThrowNew(jlexception, "Error creating cuBLAS handle");
  }

  double alpha = 1.0;
  double beta = 0.0;
  status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows_a, cols_b, cols_a, &alpha, dev_A, cols_a, dev_B, cols_a, &beta,
                       dev_C, rows_a);
  if (status != CUBLAS_STATUS_SUCCESS) {
    env->ThrowNew(jlexception, "Error calling cublasDgemm");
  }

  auto* host_C = env->GetDoubleArrayElements(C, nullptr);
  cuda_error = cudaMemcpyAsync(host_C, dev_C, size_C * sizeof(double), cudaMemcpyDefault);
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error copying C from device");
  }

  cuda_error = cudaFree(dev_A);
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error freeing A from device");
  }

  cuda_error = cudaFree(dev_C);
  if (cuda_error != cudaSuccess) {
    env->ThrowNew(jlexception, "Error freeing C from device");
  }

  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    env->ThrowNew(jlexception, "Error destroying cuBLAS handle");
  }

  env->ReleaseDoubleArrayElements(A, host_A, JNI_ABORT);
  env->ReleaseDoubleArrayElements(C, host_C, 0);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_ml_linalg_JniRAPIDSML_calSVD
  (JNIEnv * env, jclass, jint m, jdoubleArray A, jdoubleArray U, jdoubleArray S, jint deviceID) {
    cudaSetDevice(deviceID);
    raft::handle_t handle;
    cudaStream_t stream = handle.get_stream();

    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;

    double *d_A = NULL;
    double *d_S = NULL;
    double *d_U = NULL;

    cudaStat1 = cudaMalloc ((void**)&d_A  , sizeof(double)*m*m);
    cudaStat2 = cudaMalloc ((void**)&d_S  , sizeof(double)*m);
    cudaStat3 = cudaMalloc ((void**)&d_U  , sizeof(double)*m*m);

    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    auto size_A = env->GetArrayLength(A);
    jdouble* host_A = env->GetDoubleArrayElements(A, JNI_FALSE);

    cudaStat1 = cudaMemcpy(d_A, host_A, sizeof(double)*m*m, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);

    auto* host_U = env->GetDoubleArrayElements(U, nullptr);
    auto cuda_error = cudaMemcpyAsync(host_U, d_U, m * m * sizeof(double), cudaMemcpyDefault);
    assert(cudaSuccess == cuda_error);

    auto* host_S = env->GetDoubleArrayElements(S, nullptr);
    cuda_error = cudaMemcpyAsync(host_S, d_S, m * sizeof(double), cudaMemcpyDefault);
    assert(cudaSuccess == cuda_error);

    raft::linalg::eigDC(handle, d_A, m, m, d_U, d_S, stream);
    raft::matrix::colReverse(d_U, m, m, stream);
    raft::matrix::rowReverse(d_S, m, 1, stream);
    raft::matrix::seqRoot(d_S, d_S, 1.0, m, stream, true);

    signFlip(d_U, m, m, d_U, m, stream);

    cudaStat1 = cudaMemcpy(host_U , d_U , sizeof(double)*m*m, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(host_S , d_S , sizeof(double)*m, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    if (d_A    ) cudaFree(d_A);
    if (d_S    ) cudaFree(d_S);
    if (d_U    ) cudaFree(d_U);
    env->ReleaseDoubleArrayElements(A, host_A, JNI_ABORT);
    env->ReleaseDoubleArrayElements(U, host_U, 0);
    env->ReleaseDoubleArrayElements(S, host_S, 0);
  }

}  // extern "C"



