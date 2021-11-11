#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <raft/linalg/cublas_wrappers.h>
#include <raft/linalg/eig.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/matrix/math.cuh>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <rmm/exec_policy.hpp>
#include <jni.h>
#include "ml_utils.cu"

extern "C" {

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

  auto status = raft::linalg::cublasgemm(raft_handle.get_cublas_handle(), convertToCublasOpEnum(transa), convertToCublasOpEnum(transb), m, n, k, &alpha, dev_A, lda, dev_B, ldb, &beta,
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

  env->ReleaseDoubleArrayElements(A, host_A, JNI_ABORT);
  env->ReleaseDoubleArrayElements(B, host_B, JNI_ABORT);
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

}// extern "C"