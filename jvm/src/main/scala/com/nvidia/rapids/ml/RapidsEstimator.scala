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

import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.rapids.{Fit, PythonRunnerUtils, RapidsUtils}
import org.apache.spark.sql.Dataset

/** Implementation of the automatic-resource-management pattern */
object Arm {
  /** Executes the provided code block and then closes the resource */
  def withResource[T <: AutoCloseable, V](r: T)(block: T => V): V = {
    try {
      block(r)
    } finally {
      r.close()
    }
  }
}

trait RapidsEstimator extends Params {

  /**
   * The estimator name
   * @return
   */
  def estimatorName: String

  def trainOnPython(dataset: Dataset[_]): Object = {
    // Get the user-defined parameters and pass them to python process as a dictionary
    val params = RapidsUtils.getUserDefinedParams(this)

    val runner = new PythonRunnerUtils(
      Fit(estimatorName, params),
      dataset.toDF)

    Arm.withResource(runner) { _ =>
      runner.runInPython(useDaemon = false)
    }
  }

}
