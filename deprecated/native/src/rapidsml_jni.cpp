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

#include <jni.h>
#include <assert.h>
#include <iostream>
#include <stdlib.h>

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/sequence.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include "rapidsml_jni.hpp"
#include "jni_utils.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_ml_linalg_JniRAPIDSML_dgemmWithColumnViewPtr(
  JNIEnv *env, jclass, jint transa, jint transb, jint m, jint n, jint k, jdouble alpha,
  jdoubleArray A, jint lda, jlong B,jint ldb, jdouble beta, jint ldc, jint deviceID) {
  try {
    cudf::jni::native_jdoubleArray native_A(env, A);
    auto ret_column = dgemm(transa, transb, m, n, k, alpha, native_A.data(), native_A.size(), lda,
                            B, ldb, beta, ldc, deviceID);
    return ret_column;
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_ml_linalg_JniRAPIDSML_dgemmCov(JNIEnv *env, jclass,
  jint transa, jint transb, jint m, jint n, jint k, jdouble alpha, jlong A, jint lda, jlong B,
  jint ldb, jdouble beta, jdoubleArray C, jint ldc, jint deviceID) {
  try {
    cudf::jni::native_jdoubleArray native_C(env, C);
    dgemmCov(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, native_C.data(), ldc, deviceID);
  }
  CATCH_STD(env, 0);
}

}  // extern "C"
