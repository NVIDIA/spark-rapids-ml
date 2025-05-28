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

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.linalg.{Matrix, Vector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{MLReadable, MLReader, MLWritable}
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * Model produced by RapidsLogisticRegression.
 *
 * RapidsLogisticRegressionModel extends from the Spark LogisticRegressionModel and stores
 * the model attributes trained by spark-rapids-ml python in string format.
 */
class RapidsLogisticRegressionModel(override val uid: String,
                                    override val coefficientMatrix: Matrix,
                                    override val interceptVector: Vector,
                                    override val numClasses: Int,
                                    override val modelAttributes: String)
  extends LogisticRegressionModel(uid, coefficientMatrix, interceptVector,
    numClasses, numClasses != 2) with MLWritable with RapidsModel {

  private[ml] def this() = this("", null, null, 2, null)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformOnPython(dataset, super.transform)
  }

  /**
   * The model name
   */
  override def name: String = "LogisticRegressionModel"

  override def copy(extra: ParamMap): RapidsLogisticRegressionModel = {
    val newModel = copyValues(
      new RapidsLogisticRegressionModel(uid, coefficientMatrix, interceptVector,
        numClasses, modelAttributes), extra)
    newModel.setSummary(trainingSummary).setParent(parent)
    newModel
  }

  override def cpu: LogisticRegressionModel = {
    copyValues(
      new LogisticRegressionModel(uid, coefficientMatrix, interceptVector, numClasses, numClasses != 2))
  }
}

object RapidsLogisticRegressionModel extends MLReadable[RapidsLogisticRegressionModel] {

  override def read: MLReader[RapidsLogisticRegressionModel] = new RapidsLogisticRegressionModelReader

  override def load(path: String): RapidsLogisticRegressionModel = super.load(path)

  private class RapidsLogisticRegressionModelReader extends MLReader[RapidsLogisticRegressionModel] {

    override def load(path: String): RapidsLogisticRegressionModel = {
      val cpuModel = LogisticRegressionModel.load(path)
      val attributesPath = new Path(path, "attributes").toString
      val row = sparkSession.read.parquet(attributesPath).first()
      val model = new RapidsLogisticRegressionModel(row.getString(0),
        cpuModel.coefficientMatrix, cpuModel.interceptVector, cpuModel.numClasses, row.getString(1))
      cpuModel.paramMap.toSeq.foreach(p => model.set(p.param.name, p.value))
      model
    }
  }
}
