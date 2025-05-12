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

package com.nvidia.rapids.ml

import org.apache.spark.ml.feature.{PCA, PCAModel}
import org.apache.spark.ml.rapids.RapidsPCAModel
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.Dataset

/**
 * RapidsPCA is a JVM wrapper of PCA in spark-rapids-ml python package.
 *
 * The training process is going to launch a Python Process where to run spark-rapids-ml
 * PCA and return the corresponding model
 *
 * @param uid unique ID of the estimator
 */
class RapidsPCA(override val uid: String) extends PCA with DefaultParamsWritable
  with RapidsEstimator {

  def this() = this(Identifiable.randomUID("rfc"))

  override def fit(dataset: Dataset[_]): RapidsPCAModel = {
    val trainedModel = trainOnPython(dataset)
    val cpuModel = copyValues(trainedModel.model.asInstanceOf[PCAModel])
    copyValues(new RapidsPCAModel(uid, cpuModel, trainedModel.modelAttributes))
  }

  /**
   * The estimator name
   */
  override def name: String = "PCA"
}

object RapidsPCA extends DefaultParamsReadable[RapidsPCA] {

  override def load(path: String): RapidsPCA = super.load(path)

}
