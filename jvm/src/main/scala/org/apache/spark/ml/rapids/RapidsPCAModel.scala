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
import org.apache.spark.ml.feature.PCAModel
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{MLReadable, MLReader}
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * Model produced by RapidsPCA.
 *
 * RapidsPCAModel extends from the Spark PCAModel and stores
 * the model attributes trained by spark-rapids-ml python in string format.
 */
class RapidsPCAModel(override val uid: String,
                     protected[ml] override val cpuModel: PCAModel,
                     override val modelAttributes: String)
  extends PCAModel(uid, cpuModel.pc, cpuModel.explainedVariance) with RapidsModel {

  private[ml] def this() = this("", null, null)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformOnPython(dataset)
  }

  /**
   * The model name
   */
  override def name: String = "PCAModel"

  override def copy(extra: ParamMap): RapidsPCAModel = {
    val newModel = copyValues(
      new RapidsPCAModel(uid, cpuModel, modelAttributes), extra)
    newModel
  }

  override def featureName: String = getInputCol

}


object RapidsPCAModel extends MLReadable[RapidsPCAModel] {

  override def read: MLReader[RapidsPCAModel] = new RapidsPCAModelReader

  override def load(path: String): RapidsPCAModel = super.load(path)

  private class RapidsPCAModelReader extends MLReader[RapidsPCAModel] {

    override def load(path: String): RapidsPCAModel = {
      val cpuModel = PCAModel.load(path)
      val attributesPath = new Path(path, "attributes").toString
      val row = sparkSession.read.parquet(attributesPath).first()
      val model = new RapidsPCAModel(row.getString(0),
        cpuModel, row.getString(1))
      cpuModel.paramMap.toSeq.foreach(p => model.set(p.param.name, p.value))
      model
    }
  }
}
