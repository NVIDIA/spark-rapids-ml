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

import org.apache.spark.ml.clustering.rapids.RapidsKMeansModel
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType

/**
 * RapidsKMeans is a JVM wrapper of KMeans in spark-rapids-ml python package.
 *
 * The training process is going to launch a Python Process where to run spark-rapids-ml
 * KMeans and return the corresponding model
 *
 * @param uid unique ID of the estimator
 */
class RapidsKMeans(override val uid: String) extends KMeans with DefaultParamsWritable
  with RapidsEstimator {

  def this() = this(Identifiable.randomUID("kmeans"))

  override def fit(dataset: Dataset[_]): RapidsKMeansModel = {
    val trainedModel = trainOnPython(dataset)
    val cpuModel = copyValues(trainedModel.model.asInstanceOf[KMeansModel])
    copyValues(new RapidsKMeansModel(uid, cpuModel, trainedModel.modelAttributes))
  }

  // Override this function to allow feature to be array
  override def transformSchema(schema: StructType): StructType = schema

  /**
   * The estimator name
   */
  override def name: String = "KMeans"
}

object RapidsKMeans extends DefaultParamsReadable[RapidsKMeans] {

  override def load(path: String): RapidsKMeans = super.load(path)

}
