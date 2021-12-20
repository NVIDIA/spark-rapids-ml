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

import ai.rapids.cudf.{NvtxColor, NvtxRange, Table}

import java.util.{Arrays => JavaArrays}
import breeze.linalg.{DenseMatrix => BDM}
import com.nvidia.spark.rapids.ColumnarRdd
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg._
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}
import org.apache.spark.TaskContext
import org.apache.spark.sql.DataFrame

class RapidsRowMatrix (
    val listColumn: DataFrame,
    val meanCentering: Boolean,
    val gpuId: Int,
    private var nCols: Int) extends Logging with Serializable {

  /** Alternative constructor leaving matrix dimensions to be determined automatically. */
  def this(listColumn: DataFrame,
           numCols: Int,
           meanCentering: Boolean = true,
           gpuId: Int = -1) =
    this(listColumn, meanCentering,gpuId, numCols)

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
    val n = nCols
    require(k > 0 && k <= n, s"k = $k out of range (0, n = $n]")
    val nvtxRangeCov = new NvtxRange("compute cov", NvtxColor.RED)

    val Cov = try {
      computeCovariance()
    } finally {
      nvtxRangeCov.close()
    }

    val nvtxRangeSVD = new NvtxRange("cuSolver SVD", NvtxColor.BLUE)

    var svdOutput: Array[(DenseMatrix, DenseMatrix)] = null
    val gpuIdBC = listColumn.sparkSession.sparkContext.broadcast(gpuId)
    try {
      val rddTmp = listColumn.sparkSession.sparkContext.parallelize(Seq(0), 1)
      val svdRdd = rddTmp.mapPartitions( _ => {
        val gpu = if (gpuIdBC.value == -1) {
          TaskContext.get().resources()("gpu").addresses(0).toInt
        } else {
          gpuIdBC.value
        }
        val dense_U = DenseMatrix.zeros(n, n)
        val dense_S = DenseMatrix.zeros(1, n)
        RAPIDSML.calSVD(n, Cov, dense_U, dense_S, gpu)
        Iterator.single(dense_U, dense_S)
      })
      svdOutput = svdRdd.collect()
    } finally {
      nvtxRangeSVD.close()
    }
    val u = svdOutput.head._1.asBreeze.asInstanceOf[BDM[Double]]
    val s = svdOutput.head._2.asBreeze.asInstanceOf[BDM[Double]]
    val eigenSum = s.data.sum
    val explainedVariance = s.data.map(_ / eigenSum)

    if (k == n) {
      (new DenseMatrix(n, k, u.data), new DenseVector(explainedVariance))
    } else {
      (new DenseMatrix(n, k, JavaArrays.copyOfRange(u.data, 0, n * k)),
        new DenseVector(JavaArrays.copyOfRange(explainedVariance, 0, k)))
    }

  }

  /**
   * Computes the covariance matrix, treating each row as an observation.
   *
   * @return a ColumnView of LIST type, size n x n
   *
   */
  private def computeCovariance(): DenseMatrix = {
    val meanBC = if (meanCentering) {
      // val nvtxRangeMean = new NvtxRange("mean center", NvtxColor.ORANGE)
      // TODO: add proper solution for this.
      // Now the mean centering is done as a ETL preprocess in PCA application
    } else {
      listColumn.sparkSession.sparkContext.broadcast(OldVectors.zeros(0))
    }
    val gpuIdBC = listColumn.sparkSession.sparkContext.broadcast(gpuId)
    val columnarRdd = ColumnarRdd(listColumn)
    val cov = {
      columnarRdd.mapPartitions( iterator => {
        val gpu = if (gpuIdBC.value == -1) {
          TaskContext.get().resources()("gpu").addresses(0).toInt
        } else {
          gpuIdBC.value
        }
        // only input column in this table
        val partition = iterator.toList

        val bigTable = if (partition.length > 1) {
          Table.concatenate(partition: _*)
        } else {
          partition.head
        }
        try {
          assert(bigTable.getNumberOfColumns == 1)
          val C = DenseMatrix.zeros(nCols, nCols)
          val inputCol = bigTable.getColumn(0)
          RAPIDSML.cov(inputCol, nCols, C, gpu)
          Iterator.single(C.asBreeze)
        } finally {
          bigTable.close()
        }
      })
    }
    val M = cov.reduce((a, b) => a + b)
    Matrices.fromBreeze(M).toDense
  }
}
