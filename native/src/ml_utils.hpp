#pragma once

#include <raft/linalg/cublas_wrappers.h>

namespace {

constexpr char const* RUNTIME_ERROR_CLASS = "java/lang/RuntimeException";
constexpr char const* ILLEGAL_ARG_CLASS   = "java/lang/IllegalArgumentException";

} // anonymous namespace

cublasOperation_t convertToCublasOpEnum(int int_type);

void signFlip(double* input, int n_rows, int n_cols, double* components,
              int n_cols_comp, cudaStream_t stream);

long dgemm(int transa, int transb, int m, int n,
           int k, double alpha, double* A, int size_A, int lda, long B,
           int ldb, double beta, int ldc, int deviceID);