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
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.param.shared.HasFeaturesCol
import org.apache.spark.ml.util.{GeneralMLWriter, MLWritable, MLWriter}
import org.apache.spark.sql.{DataFrame, Dataset}

trait RapidsModel extends MLWritable with Params with HasFeaturesCol {

  /**
   * The attributes of the corresponding spark-rapids-ml model, it has been
   * encoded to json format. We don't need to access it
   */
  protected[ml] val modelAttributes: String

  /**
   * The model name
   */
  def name: String

  def featureName: String = getFeaturesCol

  protected val logger = LogFactory.getLog("Spark-Rapids-ML Plugin")

  def transformOnPython(dataset: Dataset[_],
                        cpuTransformFunc: Dataset[_] => DataFrame): DataFrame = {
    val usePython = dataset.sparkSession.conf.get("spark.rapids.ml.python.transform.enabled", "true").toBoolean
    val isVector = dataset.schema(featureName).dataType.isInstanceOf[VectorUDT]
    if (!isVector && !usePython) {
      throw new IllegalArgumentException("Please enable spark.rapids.ml.python.transform.enabled to " +
        "transform dataset in python for non-vector input.")
    }

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
      cpuTransformFunc(dataset)
    }
  }

  override def write: MLWriter = new RapidsModelWriter(this)

  def cpu: Model[_]
}

class RapidsModelWriter(instance: RapidsModel) extends
  GeneralMLWriter(instance.asInstanceOf[Model[_]]) with Logging {

  override protected def saveImpl(path: String): Unit = {
    val writer = instance.cpu.asInstanceOf[MLWritable].write
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
