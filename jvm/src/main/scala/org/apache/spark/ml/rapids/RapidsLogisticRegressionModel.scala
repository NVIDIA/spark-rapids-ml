/**
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.rapids

import com.nvidia.rapids.ml.RapidsModel

import org.apache.hadoop.fs.Path
import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{MLReadable, MLReader, MLWritable, MLWriter}
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * Model produced by RapidsLogisticRegression.
 *
 * RapidsLogisticRegressionModel extends from the Spark LogisticRegressionModel and stores
 * the model attributes training by spark-rapids-ml python in string format.
 */
class RapidsLogisticRegressionModel(override val uid: String,
                                    protected override val cpuModel: LogisticRegressionModel,
                                    override val modelAttributes: String,
                                    private val isMultinomial: Boolean)
  extends LogisticRegressionModel(uid, cpuModel.coefficientMatrix, cpuModel.interceptVector,
    cpuModel.numClasses, isMultinomial) with MLWritable with RapidsModel {

  private[ml] def this() = this("", null, "", false)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformOnPython(dataset)
  }

  /**
   * The model name
   */
  override def name: String = "LogisticRegressionModel"

  override def copy(extra: ParamMap): RapidsLogisticRegressionModel = {
    val newModel = copyValues(
      new RapidsLogisticRegressionModel(uid, cpuModel, modelAttributes, isMultinomial), extra)
    newModel.setSummary(trainingSummary).setParent(parent)
    newModel
  }

  override def write: MLWriter =
    new RapidsLogisticRegressionModel.RapidsLogisticRegressionModelWriter(this)
}

object RapidsLogisticRegressionModel extends MLReadable[RapidsLogisticRegressionModel] {

  override def read: MLReader[RapidsLogisticRegressionModel] = new RapidsLogisticRegressionModelReader

  override def load(path: String): RapidsLogisticRegressionModel = super.load(path)

  private class RapidsLogisticRegressionModelWriter(instance: RapidsLogisticRegressionModel)
    extends MLWriter with Logging {

    override protected def saveImpl(path: String): Unit = {
      val writer = instance.cpuModel.write
      if (shouldOverwrite) {
        writer.overwrite()
      }
      optionMap.foreach { case (k, v) => writer.option(k, v) }
      writer.save(path)

      val attributesPath = new Path(path, "attributes").toString
      sparkSession.createDataFrame(
          Seq(Tuple3(instance.uid, instance.isMultinomial, instance.modelAttributes))
        ).write.parquet(attributesPath)
    }
  }

  private class RapidsLogisticRegressionModelReader extends MLReader[RapidsLogisticRegressionModel] {

    override def load(path: String): RapidsLogisticRegressionModel = {
      val cpuModel = LogisticRegressionModel.load(path)
      val attributesPath = new Path(path, "attributes").toString
      val row = sparkSession.read.parquet(attributesPath).first()
      val model = new RapidsLogisticRegressionModel(row.getString(0),
        cpuModel, row.getString(2), row.getBoolean(1))
      cpuModel.paramMap.toSeq.foreach(p => model.set(p.param.name, p.value))
      model
    }
  }
}
