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

package org.apache.spark.ml.feature

import ai.rapids.cudf.CudfUtil.buildDeviceMemoryBuffer
import com.nvidia.spark.RapidsUDF
import org.apache.hadoop.fs.Path
import org.apache.spark.ml._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.linalg.distributed.RapidsRowMatrix
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.StructType
import ai.rapids.cudf.{ColumnVector, DType, DeviceMemoryBuffer, Scalar}

import java.util.Optional

trait RapidsPCAParams extends PCAParams {
  /**
   * Whether to apply mean centering to the input data.
   *
   * @group param
   */
  final val meanCentering: BooleanParam = new BooleanParam(this, "meanCentering", "whether to apply mean centering")
  setDefault(meanCentering, true)

  /** @group getParam */
  def getMeanCentering: Boolean = $(meanCentering)

  /**
   * Whether to use GEMM to compute the covariance matrix.
   *
   * @group param
   */
  final val useGemm: BooleanParam =
    new BooleanParam(this, "useGemm", "whether to use GEMM to compute the covariance matrix")
  setDefault(useGemm, true)

  /** @group getParam */
  def getUseGemm: Boolean = $(useGemm)

  /**
   * Whether to use cuSolver for SVD computation.
   *
   * @group param
   */
  final val useCuSolverSVD: BooleanParam = new BooleanParam(this, "useCuSolverSVD", "whether to use cuSolver for svd")
  setDefault(useCuSolverSVD, true)

  /** @group getParam */
  def getUseCuSolverSVD: Boolean = $(useCuSolverSVD)


  /**
   * The GPU ID to use.
   *
   * @group param
   */
  private[ml] final val gpuId: IntParam = new IntParam(this, "gpuId", "the GPU ID to use")
  setDefault(gpuId, -1)

  /** @group getParam */
  private[ml] def getGpuId: Int = $(gpuId)
}

/**
 * PCA trains a model to project vectors to a lower dimensional space of the top `PCA!.k`
 * principal components.
 */
class RapidsPCA(override val uid: String)
  extends Estimator[RapidsPCAModel] with RapidsPCAParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("pca"))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setMeanCentering(value: Boolean): this.type = set(meanCentering, value)

  /** @group setParam */
  def setUseGemm(value: Boolean): this.type = set(useGemm, value)

  /** @group setParam */
  def setUseCuSolverSVD(value: Boolean): this.type = set(useCuSolverSVD, value)


  /** @group setParam */
  def setGpuId(value: Int): this.type = set(gpuId, value)

  /**
   * Computes a [[RapidsPCAModel]] that contains the principal components of the input vectors.
   */
  override def fit(dataset: Dataset[_]): RapidsPCAModel = {
    transformSchema(dataset.schema, logging = true)

    val input = dataset.select($(inputCol)).rdd.map {
      case Row(v: Vector) => v
    }
    val numFeatures = input.first().size
    require(getK <= numFeatures,
      s"source vector size $numFeatures must be no less than k=$k")

    val mat = new RapidsRowMatrix(input, $(meanCentering), getUseGemm, getUseCuSolverSVD, $(gpuId))
    val (pc, explainedVariance) = mat.computePrincipalComponentsAndExplainedVariance(getK)
    val model = new RapidsPCAModel(uid, pc, explainedVariance)
    copyValues(model.setParent(this))
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): RapidsPCA = defaultCopy(extra)
}

object RapidsPCA extends DefaultParamsReadable[RapidsPCA] {

  override def load(path: String): RapidsPCA = super.load(path)
}

/**
 * Model fitted by [[RapidsPCA]]. Transforms vectors to a lower dimensional space.
 *
 * @param pc                A principal components Matrix. Each column is one principal component.
 * @param explainedVariance A vector of proportions of variance explained by each principal
 *                          component.
 */
class RapidsPCAModel(
                      override val uid: String,
                      val pc: DenseMatrix,
                      val explainedVariance: DenseVector)
  extends Model[RapidsPCAModel] with RapidsPCAParams with MLWritable {

  import RapidsPCAModel._

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  def setUseGemm(value: Boolean): this.type = set(useGemm, value)

  /**
   * Transform a vector by computed Principal Components.
   *
   * @note Vectors to be transformed must be the same length as the source vectors given to
   *       `PCA.fit()`.
   */
  override def transform(dataset: Dataset[_]): DataFrame = {
    val outputSchema = transformSchema(dataset.schema, logging = true)
    val cols_A = dataset.select($(inputCol)).first.size

    /**
     * UDF class to speedup transform process of PCA
     */
    class gpuTransform extends Function[Array[Double], Array[Double]] with RapidsUDF with Serializable {
      override def evaluateColumnar(args: ColumnVector*): ColumnVector = {
        require(args.length == 1, s"Unexpected argument count: ${args.length}")
        val input = args.head
        val rows_A = input.getRowCount.toInt
        val AdevAddr = input.getData.getAddress
        var C: Long = 0
        // A: raw data, B: pc, C: output, deviceID= 0 for test
        C = RAPIDSML.gemm_test(RAPIDSML.CublasOperationT.CUBLAS_OP_N.id,
          RAPIDSML.CublasOperationT.CUBLAS_OP_N.id,
          rows_A, pc.numCols, cols_A, 1.0, AdevAddr, cols_A, pc, cols_A, 0.0, C, rows_A, 0)
        val dmb = buildDeviceMemoryBuffer(C, (rows_A * pc.numCols * DType.FLOAT64.getSizeInBytes).toLong)
        val childColumn = new ColumnVector(DType.FLOAT64, rows_A.toLong, Optional.of(0), dmb,
          null, null)
        val offsetCV = ColumnVector.sequence(Scalar.fromInt(0), Scalar.fromInt(pc.numCols), rows_A)
        val toClose = new java.util.ArrayList[DeviceMemoryBuffer]()
        toClose.add(dmb)
        val childHandles = Array(childColumn.getNativeView)
        val offsetDMB = offsetCV.getData.sliceWithCopy(0, offsetCV.getRowCount * 4)
        new ColumnVector(DType.LIST, rows_A, Optional.of(0), null, null, offsetDMB, toClose, childHandles)
      }

      override def apply(v1: Array[Double]): Array[Double] = ???
    }

    if (getUseGemm) {
      val transform_udf = dataset.sparkSession.udf.register("transform", new gpuTransform())
      dataset.select(transform_udf(col($(inputCol))))
      // TODO(rongou): make this faster and re-enable.
      //    if (getUseGemm) {
      //      val transformed = dataset.toDF().rdd.mapPartitions(iterator => {
      //        val gpuID = TaskContext.get().resources()("gpu").addresses(0)
      //        val partition = iterator.toList
      //            val A = Matrices.fromVectors(partition.map(_.getAs[Vector]($(inputCol)))).toDense
      //        val C = Matrices.zeros(partition.length, getK).toDense
      //        CUBLAS.gemm_b(A, pc, C, gpuID)
      //        C.rowIter.zip(partition.iterator).map { case (v, r) =>
      //          Row.fromSeq(r.toSeq ++ Seq(v))
      //        }
      //      })
      //      dataset.sparkSession.createDataFrame(transformed, outputSchema)
    }
    else {
      val transposed = pc.transpose
      val transformer = udf { vector: Vector => transposed.multiply(vector) }
      dataset.withColumn($(outputCol), transformer(col($(inputCol))), outputSchema($(outputCol)).metadata)
    }
  }

  override def transformSchema(schema: StructType): StructType = {
    var outputSchema = validateAndTransformSchema(schema)
    if ($(outputCol).nonEmpty) {
      outputSchema = SchemaUtils.updateAttributeGroupSize(outputSchema,
        $(outputCol), $(k))
    }
    outputSchema
  }

  override def copy(extra: ParamMap): RapidsPCAModel = {
    val copied = new RapidsPCAModel(uid, pc, explainedVariance)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new PCAModelWriter(this)

  override def toString: String = s"PCAModel: uid=$uid, k=${$(k)}"
}

object RapidsPCAModel extends MLReadable[RapidsPCAModel] {

  override def read: MLReader[RapidsPCAModel] = new RapidsPCAModelReader

  override def load(path: String): RapidsPCAModel = super.load(path)

  private[RapidsPCAModel] class PCAModelWriter(instance: RapidsPCAModel) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.pc, instance.explainedVariance)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }

    private case class Data(pc: DenseMatrix, explainedVariance: DenseVector)
  }

  private class RapidsPCAModelReader extends MLReader[RapidsPCAModel] {

    private val className = classOf[RapidsPCAModel].getName

    /**
     * Loads a [[RapidsPCAModel]] from data located at the input path. Note that the model includes an
     * `explainedVariance` member that is not recorded by Spark 1.6 and earlier. A model can be
     * loaded from such older data but will have an empty vector for `explainedVariance`.
     *
     * @param path path to serialized model data
     * @return a [[RapidsPCAModel]]
     */
    override def load(path: String): RapidsPCAModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val Row(pc: DenseMatrix, explainedVariance: DenseVector) =
        sparkSession.read.parquet(dataPath)
          .select("pc", "explainedVariance")
          .head()
      val model = new RapidsPCAModel(metadata.uid, pc, explainedVariance)
      metadata.getAndSetParams(model)
      model
    }
  }
}
