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

import org.apache.spark.sql.connect.plugin.MLBackendPlugin

import java.util.Optional

/**
 * Spark connect ml plugin is used to replace the spark built-in algorithms with
 * spark-rapids-ml python implementations.
 */
class Plugin extends MLBackendPlugin {

  override def transform(mlName: String): Optional[String] = {
    mlName match {
      case "org.apache.spark.ml.classification.LogisticRegression" =>
        Optional.of("com.nvidia.rapids.ml.RapidsLogisticRegression")
      case _ => Optional.empty()
    }
  }
}
