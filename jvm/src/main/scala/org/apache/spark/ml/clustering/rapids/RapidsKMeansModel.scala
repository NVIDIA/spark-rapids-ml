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

package org.apache.spark.ml.clustering.rapids

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.rapids.{RapidsModel, RapidsModelWriter}
import org.apache.spark.ml.util.{GeneralMLWriter, MLReadable, MLReader}
import org.apache.spark.mllib.clustering.{KMeansModel => MLlibKMeansModel}
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * Model fitted by RapidsKMeans.
 *
 * RapidsKMeansModel extends from the Spark KMeansModel and stores
 * the model attributes trained by spark-rapids-ml python in string format.
 */
class RapidsKMeansModel(override val uid: String,
                        override private[clustering] val parentModel: MLlibKMeansModel,
                        override val modelAttributes: String)
  extends KMeansModel(uid, parentModel) with RapidsModel {

  private[ml] def this() = this("", null, null)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformOnPython(dataset, super.transform)
  }

  /**
   * The model name
   */
  override def name: String = "KMeansModel"

  override def copy(extra: ParamMap): RapidsKMeansModel = {
    val newModel = copyValues(
      new RapidsKMeansModel(uid, parentModel, modelAttributes), extra)
    newModel
  }

  override def write: GeneralMLWriter = new RapidsModelWriter(this)

  override def cpu: KMeansModel = {
    copyValues(new KMeansModel(uid, parentModel))
  }
}

object RapidsKMeansModel extends MLReadable[RapidsKMeansModel] {

  override def read: MLReader[RapidsKMeansModel] = new RapidsKMeansModelReader

  override def load(path: String): RapidsKMeansModel = super.load(path)

  private class RapidsKMeansModelReader extends MLReader[RapidsKMeansModel] {

    override def load(path: String): RapidsKMeansModel = {
      val cpuModel = KMeansModel.load(path)
      val attributesPath = new Path(path, "attributes").toString
      val row = sparkSession.read.parquet(attributesPath).first()
      val model = new RapidsKMeansModel(row.getString(0),
        cpuModel.parentModel, row.getString(1))
      cpuModel.paramMap.toSeq.foreach(p => model.set(p.param.name, p.value))
      model
    }
  }
}
