/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

import com.nvidia.spark.RapidsUDF
import org.apache.hadoop.fs.Path
import org.apache.spark.ml._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.linalg.distributed.RapidsRowMatrix
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{col}
import org.apache.spark.sql.types.StructType
import org.apache.spark.TaskContext
import ai.rapids.cudf.ColumnVector

import scala.collection.mutable

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

  /**
   * Computes a [[RapidsPCAModel]] that contains the principal components of the input vectors.
   */
  override def fit(dataset: Dataset[_]): RapidsPCAModel = {
    val input = dataset.select($(inputCol))
    val numCols = input.first().get(0).asInstanceOf[mutable.WrappedArray[Any]].length

    val mat = new RapidsRowMatrix(input, $(meanCentering), numCols)
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

  /**
   * Transform a vector by computed Principal Components.
   *
   * @note Vectors to be transformed must be the same length as the source vectors given to
   *       `PCA.fit()`.
   */
  override def transform(dataset: Dataset[_]): DataFrame = {

    val isLocal = dataset.sparkSession.sparkContext.isLocal
    /**
     * UDF class to speedup transform process of PCA
     */
    class gpuTransform extends Function[mutable.WrappedArray[Double], Array[Double]]
      with RapidsUDF with Serializable {
      override def evaluateColumnar(numRows: Int, args: ColumnVector*): ColumnVector = {
        logDebug("==========using GPU transform==========")
        val gpu = if (isLocal) {
          0
        } else {
          TaskContext.get().resources()("gpu").addresses(0).toInt
        }
        require(args.length == 1, s"Unexpected argument count: ${args.length}")
        val input = args.head
        // Due to the layout of LIST type ColumnVector, cublas gemm function should return the transposed result matrix
        // for compatibility. e.g. an expected output matrix(actually a columnar vector of LIST type)
        // [1,2]
        // [3,4]
        // [5,6]
        // its memory data layout from Cublas GEMM is [1,3,5,2,4,6]. However, it will be displayed in LIST ColumnVector as :
        // [1,3]
        // [5,2]
        // [4,6]
        // To fill the gap between native memory and CV(ColumnVector) data storage, we consider the following nature:
        // if A * B = C, then BT * AT = CT. (T means transpose). In this case the output matrix becomes:
        // [1,3,5]
        // [2,4,6]
        // whose memory data layout is [1,2,3,4,5,6]. Then it can be consumed by CV directly.
        val C = RAPIDSML.gemm(pc, input ,gpu)
        new ColumnVector(C)
      }

      override def apply(v1: mutable.WrappedArray[Double]): Array[Double] = {
        logDebug("==========using CPU transform==========")
        pc.transpose.multiply(Vectors.dense(v1.toArray)).toArray
      }
    }


    val transform_udf = dataset.sparkSession.udf.register("pca_transform", new gpuTransform())
    dataset.withColumn($(outputCol), transform_udf(col($(inputCol))))
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
