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
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.util.{GeneralMLWriter, MLReadable, MLReader}
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * Model produced by RapidsLinearRegression.
 *
 * RapidsLinearRegressionModel extends from the Spark LinearRegressionModel and stores
 * the model attributes trained by spark-rapids-ml python in string format.
 */
class RapidsLinearRegressionModel(override val uid: String,
                                  protected[ml] override val cpuModel: LinearRegressionModel,
                                  override val modelAttributes: String)
  extends LinearRegressionModel(uid, cpuModel.coefficients, cpuModel.intercept, cpuModel.scale)
    with RapidsModel {

  private[ml] def this() = this("", null, "")

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformOnPython(dataset)
  }

  /**
   * The model name
   */
  override def name: String = "LinearRegressionModel"

  override def copy(extra: ParamMap): RapidsLinearRegressionModel = {
    copyValues(
      new RapidsLinearRegressionModel(uid, cpuModel, modelAttributes), extra)
  }

  override def write: GeneralMLWriter = new RapidsModelWriter(this)
}

object RapidsLinearRegressionModel extends MLReadable[RapidsLinearRegressionModel] {

  override def read: MLReader[RapidsLinearRegressionModel] = new RapidsLinearRegressionModelReader

  override def load(path: String): RapidsLinearRegressionModel = super.load(path)

  private class RapidsLinearRegressionModelReader extends MLReader[RapidsLinearRegressionModel] {

    override def load(path: String): RapidsLinearRegressionModel = {
      val cpuModel = LinearRegressionModel.load(path)
      val attributesPath = new Path(path, "attributes").toString
      val row = sparkSession.read.parquet(attributesPath).first()
      val model = new RapidsLinearRegressionModel(row.getString(0),
        cpuModel, row.getString(1))
      cpuModel.paramMap.toSeq.foreach(p => model.set(p.param.name, p.value))
      model
    }
  }
}
