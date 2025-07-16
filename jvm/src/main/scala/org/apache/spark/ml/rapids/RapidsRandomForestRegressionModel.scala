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
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, RandomForestRegressionModel}
import org.apache.spark.ml.util.{MLReadable, MLReader, MLWritable}
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * Model produced by RapidsRandomForestClassifier.
 *
 * RapidsRandomForestRegressionModel extends from the Spark RandomForestRegressionModel and stores
 * the model attributes trained by spark-rapids-ml python in string format.
 */
class RapidsRandomForestRegressionModel(override val uid: String,
                                        private val _trees: Array[DecisionTreeRegressionModel],
                                        override val numFeatures: Int,
                                        override val modelAttributes: String)
  extends RandomForestRegressionModel(uid, _trees, numFeatures)
    with MLWritable with RapidsModel {

  private[ml] def this() = this("", null, 1, "")

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformOnPython(dataset, super.transform)
  }

  /**
   * The model name
   */
  override def name: String = "RandomForestRegressionModel"

  override def copy(extra: ParamMap): RapidsRandomForestRegressionModel = {
    copyValues(
      new RapidsRandomForestRegressionModel(uid, _trees, numFeatures, modelAttributes), extra)
  }

  override def cpu: RandomForestRegressionModel = {
    copyValues(new RandomForestRegressionModel(uid, _trees, numFeatures))
  }
}

object RapidsRandomForestRegressionModel extends MLReadable[RapidsRandomForestRegressionModel] {

  override def read: MLReader[RapidsRandomForestRegressionModel] = new RapidsRandomForestRegressionModelReader

  override def load(path: String): RapidsRandomForestRegressionModel = super.load(path)

  private class RapidsRandomForestRegressionModelReader extends MLReader[RapidsRandomForestRegressionModel] {

    override def load(path: String): RapidsRandomForestRegressionModel = {
      val cpuModel = RandomForestRegressionModel.load(path)
      val attributesPath = new Path(path, "attributes").toString
      val row = sparkSession.read.parquet(attributesPath).first()
      val model = new RapidsRandomForestRegressionModel(row.getString(0), cpuModel.trees,
        cpuModel.numFeatures, row.getString(1))
      cpuModel.paramMap.toSeq.foreach(p => model.set(p.param.name, p.value))
      model
    }
  }
}
