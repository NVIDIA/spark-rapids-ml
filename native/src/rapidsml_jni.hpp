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

#pragma once

#include <raft/linalg/cublas_wrappers.h>

cublasOperation_t convertToCublasOpEnum(int int_type);

void signFlip(double* input, int n_rows, int n_cols, double* components,
              int n_cols_comp, cudaStream_t stream);

long dgemm(int transa, int transb, int m, int n,
           int k, double alpha, double* A, int size_A, int lda, long B,
           int ldb, double beta, int ldc, int deviceID);

void dgemmCov(int transa, int transb, int m, int n,int k, double alpha, long A, int lda,long B,
              int ldb, double beta, double* C, int ldc, int deviceID);
