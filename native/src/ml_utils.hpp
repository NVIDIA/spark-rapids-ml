#pragma once

namespace {

constexpr char const* RUNTIME_ERROR_CLASS = "java/lang/RuntimeException";
constexpr char const* ILLEGAL_ARG_CLASS   = "java/lang/IllegalArgumentException";

} // anonymous namespace

long dgemm(int transa, int transb, int m, int n,
           int k, double alpha, double* A, int size_A, int lda, long B,
           int ldb, double beta, int ldc, int deviceID);