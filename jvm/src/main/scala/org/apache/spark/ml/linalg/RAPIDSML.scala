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

package org.apache.spark.ml.linalg

import ai.rapids.cudf.ColumnView
import com.nvidia.spark.ml.linalg.JniRAPIDSML
import org.apache.spark.ml.linalg.RAPIDSML.CublasOperationT.{CUBLAS_OP_N, CUBLAS_OP_T, CublasOperationT}

/**
 * BLAS routines for MLlib's vectors and matrices.
 */

private[spark] object RAPIDSML extends Serializable {

  @transient private var _jniRAPIDSML: JniRAPIDSML = _

  private[spark] def jniRAPIDSML: JniRAPIDSML = {
    if (_jniRAPIDSML == null) {
      _jniRAPIDSML = JniRAPIDSML.getInstance
    }
    _jniRAPIDSML
  }

  object CublasOperationT extends Enumeration {
    type CublasOperationT = Value
    val CUBLAS_OP_N = Value(0)
    val CUBLAS_OP_T = Value(1)
    val CUBLAS_OP_C = Value(2)
    val CUBLAS_OP_CONJG = Value(3)
  }

  /**
   * Wrapper of cuBLAS GEMM routine to do matrix computation on GPU but with fewer parameters. This gemm is used to
   * calculate covariance of a matrix.
   * @param A Input ColumnView of LIST type, holding the input data on device.
   * @param cols_A number of columns of A.
   * @param deviceID the device that will run the computation.
   * @return value of Long type that represents the handle of a numeric type `cudf::column_view *` which is holding the
   *         computation output. Note it's a numeric column, not the LIST type because its device data will be used to
   *         do SVD, so no need to construct LIST type column.
   */
  def cov(A: ColumnView, cols_A: Int, C: DenseMatrix, deviceID: Int): Unit = {
    cov(CUBLAS_OP_N, CUBLAS_OP_T, cols_A, cols_A, A.getRowCount.toInt, 1.0, A, cols_A,
      A, cols_A, 0.0, C, cols_A, deviceID)
  }

  /**
   * Wrapper of cuBLAS GEMM routine to do matrix computation on GPU. This gemm is used to calculate covariance of a
   * matrix.
   * @param transa CublasOperationT enum value representing the operation op(A) that is non- or (conj.) transpose.
   * @param transb CublasOperationT enum value representing the operation op(B) that is non- or (conj.) transpose.
   * @param m number of rows of matrix op(A) and C.
   * @param n number of columns of matrix op(B) and C.
   * @param k number of columns of op(A) and rows of op(B).
   * @param alpha scalar used for multiplication.
   * @param A ColumnView that holds the device matrix data.
   *          (array of dimensions lda x k with lda>=max(1,m) if transa == CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.)
   * @param lda leading dimension of two-dimensional array used to store the matrix A.
   * @param B ColumnView that holds the device matrix data
   *          (Array of dimension ldb x n with ldb>=max(1,k) if transb == CUBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise.)
   * @param ldb leading dimension of two-dimensional array used to store matrix B.
   * @param beta scalar used for multiplication. If beta==0, C does not have to be a valid input.
   * @param ldc leading dimension of a two-dimensional array used to store the matrix C.
   * @param deviceID the device that will run the computation
   * @return value of Long type that represents the handle of a numeric type `cudf::column_view *` which is holding the
   *         computation output. Note it's a numeric column, not the LIST type because its device data will be used to
   *         do SVD, so no need to construct LIST type column.
   */
  def cov(transa: CublasOperationT, transb: CublasOperationT, m: Int, n: Int, k: Int, alpha: Double, A: ColumnView,
          lda: Int, B: ColumnView, ldb: Int,beta: Double, C: DenseMatrix, ldc: Int, deviceID: Int): Unit = {
    jniRAPIDSML.dgemmCov(transa.id, transb.id, m, n, k, alpha, A.getNativeView, lda, B.getNativeView, ldb, beta,
      C.values, ldc, deviceID)
  }

  /**
   *
   * @param lhs a `cudf::column_view *` holding covariance data for one partition
   * @param rhs a `cudf::column_view *` holding covariance data for another partition
   * @return a `cudf::column_view *` hold covariance data by applying `ADD` to 2 column data
   */
  def accumulateCov(lhs: Long, rhs: Long): Long = {
    jniRAPIDSML.accumulateCov(lhs, rhs)
  }

  /**
   * Wrapper of cuBLAS GEMM routine to do matrix computation on GPU.
   * Most parameters are the same as: https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
   * but modified B and removed C for integration compatibility.
   * B: ColumnView that is holding the matrix data to be transformed on device.
   * C: as the computation output is removed from parameter list and becomes a return value of Long type that represents
   * the handle of a LIST type `cudf::column_view *` which is holding the computation output.
   *
   * @param transa CublasOperationT enum value representing the operation op(A) that is non- or (conj.) transpose.
   * @param transb CublasOperationT enum value representing the operation op(B) that is non- or (conj.) transpose.
   * @param m number of rows of matrix op(A) and C.
   * @param n number of columns of matrix op(B) and C.
   * @param k number of columns of op(A) and rows of op(B).
   * @param alpha scalar used for multiplication.
   * @param A array of dimensions lda x k with lda>=max(1,m) if transa == CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.
   * @param lda leading dimension of two-dimensional array used to store the matrix A.
   * @param B ColumnView that holds the device matrix data
   *          (Array of dimension ldb x n with ldb>=max(1,k) if transb == CUBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise.)
   * @param ldb leading dimension of two-dimensional array used to store matrix B.
   * @param beta scalar used for multiplication. If beta==0, C does not have to be a valid input.
   * @param ldc leading dimension of a two-dimensional array used to store the matrix C.
   * @param deviceID the device that will run the computation
   * @return value of Long type that represents the handle of a LIST type `cudf::column_view *` which is holding the computation output.
   *         It can be used to construct ColumnVector.
   */
  def gemm(transa: CublasOperationT, transb: CublasOperationT, m: Int, n: Int, k: Int, alpha: Double, A: Array[Double],
           lda: Int, B: ColumnView, ldb: Int,beta: Double, ldc: Int, deviceID: Int): Long = {
    jniRAPIDSML.dgemmWithColumnViewPtr(transa.id, transb.id, m, n, k, alpha, A, lda, B.getNativeView, ldb, beta, ldc,
      deviceID)
  }

  /**
   * Wrapper for cuBLAS GEMM routine used for PCA transform but with fewer input parameters.
   * Detailed parameters can be inferred from input DenseMatrix.
   *
   * @param A input DenseMatrix, Here it's the principal components model from PCA training.
   * @param B ColumnView that holds B matrix data on device. Here it's raw data to be transformed by PCA.
   * @param deviceID the device that will run the computation
   * @return value of Long type that represents the handle of a LIST type `cudf::column_view *` which is holding the computation output.
   *         It can be used to construct ColumnVector.
   */
  def gemm(A: DenseMatrix, B: ColumnView, deviceID: Int): Long = {
    gemm(RAPIDSML.CublasOperationT.CUBLAS_OP_T, RAPIDSML.CublasOperationT.CUBLAS_OP_N, A.numCols, B.getRowCount.toInt,
      A.numRows, 1.0, A.values, A.numRows, B, A.numRows, 0.0, A.numCols, deviceID)
  }

  /**
   *
   * @param m size of sqiare matrix A
   * @param A raw matrix to be decomposed, size of m * m
   * @param U left matrix after decomposition
   * @param S middle vector after decomposition
   * @param deviceID the device that will run the computation
   */
  def calSVD(m: Int, A: DenseMatrix, U: DenseMatrix, S: DenseMatrix, deviceID: Int): Unit = {
    jniRAPIDSML.calSVD(m, A.values, U.values, S.values, deviceID);
  }
  
}
