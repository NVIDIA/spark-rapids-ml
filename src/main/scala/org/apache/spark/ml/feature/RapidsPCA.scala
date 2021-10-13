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

import com.nvidia.spark.ml.linalg.{NvtxColor, NvtxRange}
import org.apache.hadoop.fs.Path
import org.apache.spark.sql.types.{ArrayType, DoubleType}
import org.apache.spark.TaskContext
import org.apache.spark.ml._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.linalg.distributed.RapidsRowMatrix
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.StructType

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

    // TODO(rongou): make this faster and re-enable.
    if (getUseGemm) {
      val input = dataset.select($(inputCol)).rdd.map {
        case Row(v: Vector) => v
      }
      val n = input.first().size

      val transformed = input.mapPartitions(iterator => {
        val gpuID = TaskContext.get().resources()("gpu").addresses(0).toInt
        val partition = iterator.toList
        val bas = partition.map(v => v.asBreeze.toArray)
        val nvtxRangeConcat = new NvtxRange("concat before transform", NvtxColor.PURPLE)

        val A = try {
          new DenseMatrix(bas.length, n, Array.concat(bas: _*), isTransposed = true)
        } finally {
          nvtxRangeConcat.close()
        }

        val C = DenseMatrix.zeros(partition.length, getK)
        val nvtxRangeGemm = new NvtxRange("cublas gemm transform", NvtxColor.GREEN)
        try {
          RAPIDSML.gemm_b(A, pc, C, 0)
        } finally {
          nvtxRangeGemm.close()
        }
        Iterator.single(C)
      }).cache()

      def toSeqOfArray(m: Matrix): Seq[Array[Double]] = {
        val columns = m.toArray.grouped(m.numRows)
        val rows = columns.toSeq.transpose
        rows.map(row => row.toArray)
      }

      val seqOfArray = transformed.flatMap(toSeqOfArray)
      // Return df that only contains transform result column.
      // This is fast, 16 seconds.
      //      val rrdd = seqOfArray.map( v => {
      //        Row.fromSeq(Seq(v))
      //      })
      //      val schema = StructType(Array(
      //        StructField($(outputCol), ArrayType(DoubleType), true)
      //      ))
      //      val dfSchema = dataset.sparkSession.createDataFrame(rrdd, schema)
      //      dfSchema

      val hack_schema = dataset.schema.add($(outputCol), ArrayType(DoubleType), true)
      val result = dataset.toDF().rdd.zip(seqOfArray).map {
        case (left, right) => Row.fromSeq(left.toSeq ++ Seq(right))
      }
      val resultDf = dataset.sparkSession.createDataFrame(result, hack_schema)
      resultDf
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
