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

import com.nvidia.spark.ml.linalg.JniRAPIDSML

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

  /**
   * Adds alpha * v * v.t to a matrix in-place. This is the same as BLAS's ?SPR.
   *
   * @param U the upper triangular part of the matrix packed in an array (column major)
   */
  def spr(v: DenseVector, U: Array[Double]): Unit = {
    jniRAPIDSML.dspr(v.size, v.values, U)
  }

  /**
   * C := B.transpose * B
   *
   * @param B the matrix B that will be left multiplied by its transpose. Size of m x n.
   * @param C the resulting matrix C. Size of n x n.
   */
  def gemm(B: DenseMatrix, C: DenseMatrix, deviceID: Int): Unit = {
    val rows = B.numRows
    val cols = B.numCols
    require(B.isTransposed, "B is not transposed")
    require(C.numRows == cols, s"The rows of C don't match the columns of B. C: ${C.numRows}, B: $cols")
    require(C.numCols == cols, s"The columns of C don't match the columns of B. C: ${C.numCols}, B: $cols")
    // Since B is transposed, we treat it as A in JNI.
    jniRAPIDSML.dgemm(rows, cols, B.values, C.values, deviceID)
  }

  def gemm_b(A: DenseMatrix, B: DenseMatrix, C: DenseMatrix, deviceID: Int): Unit = {

    require(C.numRows == A.numRows, s"The rows of C don't match the rows of A. C: ${C.numRows}, A: ${A.numRows}")
    require(C.numCols == B.numCols, s"The columns of C don't match the columns of B. C: ${C.numCols}, B: ${B.numCols}")
    jniRAPIDSML.dgemm_b(A.numRows, B.numCols, A.numCols, A.values, B.values, C.values, deviceID)
  }

  def calSVD(m: Int, A: DenseMatrix, U: DenseMatrix, S: DenseMatrix, deviceID: Int): Unit = {
    require(m == A.numRows, s"The rows of A don't match required dimension")
    require(m == A.numCols, s"The cols of A don't match required dimension")
    jniRAPIDSML.calSVD(m, A.values, U.values, S.values, deviceID);
  }
  
}
