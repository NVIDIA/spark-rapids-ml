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

import com.nvidia.rapids.ml.Arm
import org.apache.commons.logging.LogFactory
import org.apache.hadoop.fs.Path
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Model
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.util.{GeneralMLWriter, MLWritable, MLWriter}
import org.apache.spark.sql.{DataFrame, Dataset}

trait RapidsModel extends MLWritable with Params {

  /**
   * The attributes of the corresponding spark-rapids-ml model, it has been
   * encoded to json format. We don't need to access it
   */
  protected[ml] val modelAttributes: String

  /**
   * The correspond CPU model which can be used to transform directly.
   */
  protected[ml] val cpuModel: Model[_]

  /**
   * The model name
   */
  def name: String

  protected val logger = LogFactory.getLog("Spark-Rapids-ML Plugin")

  def transformOnPython(dataset: Dataset[_]): DataFrame = {
    val usePython = dataset.sparkSession.conf.get("spark.rapids.ml.python.transform.enabled", "true").toBoolean
    if (usePython) {
      logger.info("Transform in python")
      // Get the user-defined parameters and pass them to python process as a dictionary
      val params = RapidsUtils.getUserDefinedParams(this)

      val runner = new PythonModelRunner(
        Transform(name, params, modelAttributes),
        dataset.toDF)

      Arm.withResource(runner) { _ =>
        runner.runInPython(useDaemon = false)
      }
    } else {
      logger.info(s"Transform using CPU $name")
      cpuModel.transform(dataset)
    }
  }

  override def write: MLWriter = new RapidsModelWriter(this)
}

class RapidsModelWriter(instance: RapidsModel) extends
  GeneralMLWriter(instance.asInstanceOf[Model[_]]) with Logging {

  override protected def saveImpl(path: String): Unit = {
    val writer = instance.cpuModel.asInstanceOf[MLWritable].write
    if (shouldOverwrite) {
      writer.overwrite()
    }
    optionMap.foreach { case (k, v) => writer.option(k, v) }
    writer.save(path)

    val attributesPath = new Path(path, "attributes").toString
    sparkSession.createDataFrame(
      Seq(Tuple2(instance.uid, instance.modelAttributes))
    ).write.parquet(attributesPath)
  }
}
