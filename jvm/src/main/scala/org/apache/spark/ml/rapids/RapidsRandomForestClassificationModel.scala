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
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, RandomForestClassificationModel}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{MLReadable, MLReader, MLWritable}
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * Model produced by RapidsRandomForestClassifier.
 *
 * RapidsRandomForestClassificationModel extends from the Spark RandomForestClassificationModel and stores
 * the model attributes trained by spark-rapids-ml python in string format.
 */
class RapidsRandomForestClassificationModel(override val uid: String,
                                            private val _trees: Array[DecisionTreeClassificationModel],
                                            override val numFeatures: Int,
                                            override val numClasses: Int,
                                            override val modelAttributes: String)
  extends RandomForestClassificationModel(uid, _trees, numFeatures, numClasses)
    with MLWritable with RapidsModel {

  private[ml] def this() = this("", null, 1, 1, "")

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformOnPython(dataset, super.transform)
  }

  /**
   * The model name
   */
  override def name: String = "RandomForestClassificationModel"

  override def copy(extra: ParamMap): RapidsRandomForestClassificationModel = {
    copyValues(new RapidsRandomForestClassificationModel(uid, _trees, numFeatures,
      numClasses, modelAttributes), extra)
  }

  override def cpu: RandomForestClassificationModel = {
    copyValues(new RandomForestClassificationModel(uid, _trees, numFeatures, numClasses))
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
        cpuModel.trees, cpuModel.numFeatures, cpuModel.numClasses, row.getString(1))
      cpuModel.paramMap.toSeq.foreach(p => model.set(p.param.name, p.value))
      model
    }
  }
}
