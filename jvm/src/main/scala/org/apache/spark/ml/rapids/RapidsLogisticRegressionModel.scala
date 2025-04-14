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
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * Model produced by RapidsLogisticRegression.
 *
 * RapidsLogisticRegressionModel extends from the Spark LogisticRegressionModel and stores
 * the model attributes training by spark-rapids-ml python in string format.
 */
class RapidsLogisticRegressionModel(override val uid: String,
                                    protected override val cpuModel: LogisticRegressionModel,
                                    protected override val modelAttributes: String,
                                    private val isMultinomial: Boolean)
  extends LogisticRegressionModel(uid, cpuModel.coefficientMatrix, cpuModel.interceptVector,
    cpuModel.numClasses, isMultinomial) with RapidsModel {

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

}
