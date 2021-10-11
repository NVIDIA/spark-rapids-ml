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

package org.apache.spark.ml.linalg.distributed

import java.util.{Arrays => JavaArrays}
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, svd => brzSvd}
import breeze.linalg.Matrix._
import com.nvidia.spark.ml.linalg.{NvtxColor, NvtxRange}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg._
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import org.apache.spark.TaskContext

class RapidsRowMatrix(
    val rows: RDD[Vector],
    val meanCentering: Boolean,
    val useGemm: Boolean,
    val useCuSolverSVD: Boolean,
    val gpuId: Int,
    private var nRows: Long,
    private var nCols: Int) extends Logging {

  /** Alternative constructor leaving matrix dimensions to be determined automatically. */
  def this(rows: RDD[Vector],
           meanCentering: Boolean = true,
           useGemm: Boolean = false,
           useCuSolverSVD: Boolean = false,
           gpuId: Int = -1) =
    this(rows, meanCentering, useGemm, useCuSolverSVD,gpuId, 0L, 0)

  /** Gets or computes the number of rows. */
  def numRows(): Long = {
    if (nRows <= 0L) {
      nRows = rows.count()
      if (nRows == 0L) {
        sys.error("Cannot determine the number of rows because it is not specified in the " +
            "constructor and the rows RDD is empty.")
      }
    }
    nRows
  }

  /**
   * Computes the top k principal components and a vector of proportions of
   * variance explained by each principal component.
   * Rows correspond to observations and columns correspond to variables.
   * The principal components are stored a local matrix of size n-by-k.
   * Each column corresponds for one principal component,
   * and the columns are in descending order of component variance.
   * The row data do not need to be "centered" first; it is not necessary for
   * the mean of each column to be 0. But, if the number of columns are more than
   * 65535, then the data need to be "centered".
   *
   * @param k number of top principal components.
   * @return a matrix of size n-by-k, whose columns are principal components, and
   *         a vector of values which indicate how much variance each principal component
   *         explains
   */
  def computePrincipalComponentsAndExplainedVariance(k: Int): (DenseMatrix, DenseVector) = {
    val n = numCols().toInt
    require(k > 0 && k <= n, s"k = $k out of range (0, n = $n]")
    val nvtxRangeCov = new NvtxRange("compute cov", NvtxColor.RED)

    val Cov = try {
      computeCovariance()
    } finally {
      nvtxRangeCov.close()
    }


    if (useCuSolverSVD) {
      val nvtxRangeSVD = new NvtxRange("cuSolver SVD", NvtxColor.BLUE)

      val dense_U = DenseMatrix.zeros(n, n)

      val dense_S = DenseMatrix.zeros(1, n)
      try {
        // this is done on driver, so no task resources here, assign 0 manually.
        RAPIDSML.calSVD(n, Cov, dense_U, dense_S, 0)
      } finally {
        nvtxRangeSVD.close()
      }
      val u = dense_U.asBreeze.asInstanceOf[BDM[Double]]
      val s = dense_S.asBreeze.asInstanceOf[BDM[Double]]
      val eigenSum = s.data.sum
      val explainedVariance = s.data.map(_ / eigenSum)

      if (k == n) {
        (new DenseMatrix(n, k, u.data), new DenseVector(explainedVariance))
      } else {
        (new DenseMatrix(n, k, JavaArrays.copyOfRange(u.data, 0, n * k)),
          new DenseVector(JavaArrays.copyOfRange(explainedVariance, 0, k)))
      }
    } else {
      val nvtxRangeSVD = new NvtxRange("cpu SVD", NvtxColor.BLUE)
      val Cov_brz = Cov.asBreeze.asInstanceOf[BDM[Double]]
      val brzSvd.SVD(u: BDM[Double], s: BDV[Double], _) = brzSvd(Cov_brz)
      nvtxRangeSVD.close()
      val eigenSum = s.data.sum
      val explainedVariance = s.data.map(_ / eigenSum)

      if (k == n) {
        (new DenseMatrix(n, k, u.data), new DenseVector(explainedVariance))
      } else {
        (new DenseMatrix(n, k, JavaArrays.copyOfRange(u.data, 0, n * k)),
          new DenseVector(JavaArrays.copyOfRange(explainedVariance, 0, k)))
      }
    }
  }

  /** Gets or computes the number of columns. */
  def numCols(): Long = {
    if (nCols <= 0) {
      try {
        // Calling `first` will throw an exception if `rows` is empty.
        nCols = rows.first().size
      } catch {
        case err: UnsupportedOperationException =>
          sys.error("Cannot determine the number of cols because it is not specified in the " +
              "constructor and the rows RDD is empty.")
      }
    }
    nCols
  }

  /**
   * Computes the covariance matrix, treating each row as an observation.
   *
   * @return a local dense matrix of size n x n
   *
   * @note This cannot be computed on matrices with more than 65535 columns.
   */
  private def computeCovariance(): DenseMatrix = {
    val n = numCols().toInt

    val meanBC = if (meanCentering) {
      val nvtxRangeMean = new NvtxRange("mean center", NvtxColor.ORANGE)
      val summary = try {
        Statistics.colStats(rows.map(v => OldVectors.fromML(v)))
      }  finally {
           nvtxRangeMean.close()
         }
      val m = summary.count
      require(m > 1, s"RapidsRowMatrix.computeCovariance called on matrix with only $m rows." +
          "  Cannot compute the covariance of a RowMatrix with <= 1 row.")
      rows.context.broadcast(summary.mean)
    } else {
      rows.context.broadcast(OldVectors.zeros(0))
    }
    val gpuIdBC = rows.context.broadcast(gpuId)

    val M = if (useGemm) {
      val sqrtn = scala.math.sqrt(n - 1.0)
      val cov = rows.mapPartitions(iterator => {
        val gpu = if (gpuIdBC.value == -1) {
          TaskContext.get().resources()("gpu").addresses(0).toInt
        } else {
          gpuIdBC.value
        }
        val means = meanBC.value.asBreeze
        val partition = iterator.toList
        val bas = if (means.size == 0) {
          partition.map(v => (v.asBreeze /:/ sqrtn).toArray)
        } else {
          partition.map(v => ((v.asBreeze - means) /:/ sqrtn).toArray)
        }
        val nvtxRangeConcat = new NvtxRange("concat before cov", NvtxColor.PURPLE)

        val B = try {
          new DenseMatrix(bas.length, n, Array.concat(bas: _*), isTransposed = true)
        } finally {
          nvtxRangeConcat.close()
        }

        val C = DenseMatrix.zeros(n, n)

        val nvtxRangeGemm = new NvtxRange("cublas gemm", NvtxColor.GREEN)
        try {
          RAPIDSML.gemm(RAPIDSML.CublasOperationT.CUBLAS_OP_N.id, RAPIDSML.CublasOperationT.CUBLAS_OP_T.id, B.numCols, B.numCols,
            B.numRows, 1.0, B, B.numCols, B, B.numCols, 0.0, C, B.numCols, gpu)
        } finally  {
          nvtxRangeGemm.close()
        }
        Iterator.single(C.asBreeze)
      })
      cov.reduce((a, b) => a + b)
    } else {
      // Computes n*(n+1)/2, avoiding overflow in the multiplication.
      // This succeeds when n <= 65535, which is checked above
      val nt = if (n % 2 == 0) (n / 2) * (n + 1) else n * ((n + 1) / 2)

      val MU = rows.treeAggregate(null.asInstanceOf[BDV[Double]])(
        seqOp = (maybeU, v) => {
          val U =
            if (maybeU == null) {
              new BDV[Double](nt)
            } else {
              maybeU
            }

          val n = v.size
          val na = Array.ofDim[Double](n)
          val means = meanBC.value

          val ta = v.toArray
          for (index <- 0 until n) {
            na(index) = ta(index) - means(index)
          }
          BLAS.spr(1.0, new DenseVector(na), U.data)
          U
        }, combOp = (U1, U2) =>
          if (U1 == null) {
            U2
          } else if (U2 == null) {
            U1
          } else {
            U1 += U2
          }
      )

      val M = RapidsRowMatrix.triuToFull(n, MU.data).asBreeze

      var i = 0
      var j = 0
      val m1 = numRows() - 1.0
      while (i < n) {
        j = i
        while (j < n) {
          val Mij = M(i, j) / m1
          M(i, j) = Mij
          M(j, i) = Mij
          j += 1
        }
        i += 1
      }
      M
    }

    meanBC.destroy()
    gpuIdBC.destroy()
    Matrices.fromBreeze(M).toDense
  }
}

object RapidsRowMatrix {

  /**
   * Fills a full square matrix from its upper triangular part.
   */
  private def triuToFull(n: Int, U: Array[Double]): Matrix = {
    val G = new BDM[Double](n, n)

    var row = 0
    var col = 0
    var idx = 0
    var value = 0.0
    while (col < n) {
      row = 0
      while (row < col) {
        value = U(idx)
        G(row, col) = value
        G(col, row) = value
        idx += 1
        row += 1
      }
      G(col, col) = U(idx)
      idx += 1
      col += 1
    }

    Matrices.dense(n, n, G.data)
  }
}
