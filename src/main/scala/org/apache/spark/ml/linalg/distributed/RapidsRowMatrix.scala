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

import ai.rapids.cudf.{ColumnVector, ColumnView, NvtxColor, NvtxRange}

import java.util.{Arrays => JavaArrays}
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, svd => brzSvd}
import breeze.linalg.Matrix._
import com.nvidia.spark.RapidsUDF
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg._
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import org.apache.spark.TaskContext
import org.apache.spark.sql.DataFrame

import scala.collection.mutable

class RapidsRowMatrix(
    val listColumn: DataFrame,
    val meanCentering: Boolean,
    val gpuId: Int,
    private var nRows: Long,
    private var nCols: Int) extends Logging {

  /** Alternative constructor leaving matrix dimensions to be determined automatically. */
  def this(listColumn: DataFrame,
           meanCentering: Boolean = true,
           gpuId: Int = -1) =
    this(listColumn, meanCentering,gpuId, 0L, 0)

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

  def num

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

  }

  /** Gets or computes the number of columns. */
  def numCols(): Long = {
    listColumn.first().size
  }

  /**
   * Computes the covariance matrix, treating each row as an observation.
   *
   * @return a ColumnView of LIST type, size n x n
   *
   */
  private def computeCovariance(): ColumnView = {
    val meanBC = if (meanCentering) {
      val nvtxRangeMean = new NvtxRange("mean center", NvtxColor.ORANGE)
      // TODO: add proper solution for this
    } else {
      listColumn.rdd.context.broadcast(OldVectors.zeros(0))
    }
    val gpuIdBC = listColumn.rdd.context.broadcast(gpuId)

    class gpuTrain extends Function[mutable.WrappedArray[Double], Array[Double]] with RapidsUDF with Serializable {
      override def evaluateColumnar(args: ColumnVector*): ColumnVector = {
        logDebug("==========using GPU train==========")
        val gpu = if (gpuIdBC.value == -1) {
          TaskContext.get().resources()("gpu").addresses(0).toInt
        } else {
          gpuIdBC.value
        }

        require(args.length == 1, s"Unexpected argument count: ${args.length}")
        val input = args.head
        RAPIDSML.cov(input, numCols().toInt, gpu)



      }

      override def apply(v1: mutable.WrappedArray[Double]): Array[Double] = ???
    }

    val M = {

    }

    gpuIdBC.destroy()
    M
  }
}
