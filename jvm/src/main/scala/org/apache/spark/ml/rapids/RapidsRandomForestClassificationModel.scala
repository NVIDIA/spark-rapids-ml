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
import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{MLReadable, MLReader, MLWritable, MLWriter}
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * Model produced by RapidsRandomForestClassifier.
 *
 * RapidsRandomForestClassificationModel extends from the Spark RandomForestClassificationModel and stores
 * the model attributes trained by spark-rapids-ml python in string format.
 */
class RapidsRandomForestClassificationModel(override val uid: String,
                                            protected[ml] override val cpuModel: RandomForestClassificationModel,
                                            override val modelAttributes: String)
  extends RandomForestClassificationModel(uid, cpuModel.trees, cpuModel.numFeatures,
    cpuModel.numClasses) with MLWritable with RapidsModel {

  private[ml] def this() = this("", null, "")

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformOnPython(dataset)
  }

  /**
   * The model name
   */
  override def name: String = "RandomForestClassificationModel"

  override def copy(extra: ParamMap): RapidsRandomForestClassificationModel = {
    copyValues(
      new RapidsRandomForestClassificationModel(uid, cpuModel, modelAttributes), extra)
  }

}

object RapidsRandomForestClassificationModel extends MLReadable[RapidsRandomForestClassificationModel] {

  override def read: MLReader[RapidsRandomForestClassificationModel] = new RapidsRandomForestClassificationModelReader

  override def load(path: String): RapidsRandomForestClassificationModel = super.load(path)

  private class RapidsRandomForestClassificationModelReader extends MLReader[RapidsRandomForestClassificationModel] {

    override def load(path: String): RapidsRandomForestClassificationModel = {
      val cpuModel = RandomForestClassificationModel.load(path)
      val attributesPath = new Path(path, "attributes").toString
      val row = sparkSession.read.parquet(attributesPath).first()
      val model = new RapidsRandomForestClassificationModel(row.getString(0),
        cpuModel, row.getString(1))
      cpuModel.paramMap.toSeq.foreach(p => model.set(p.param.name, p.value))
      model
    }
  }
}
