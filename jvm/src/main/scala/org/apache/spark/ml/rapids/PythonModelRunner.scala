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

import java.io.{DataInputStream, DataOutputStream}

import net.razorvine.pickle.Pickler

import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.api.python.{PythonFunction, PythonRDD, PythonWorkerUtils}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.execution.python.PythonPlannerRunner


case class Transform(name: String, params: String, modelAttributes: String)

/**
 * PythonModelRunner is a bridge to launch and manage Python process. it sends the
 * model-related messages to the python process and runs it.
 *
 * @param transform the transform information
 * @param dataset   input dataset
 */
class PythonModelRunner(transform: Transform,
                        dataset: DataFrame,
                        func: PythonFunction = PythonRunnerUtils.RAPIDS_PYTHON_FUNC)
  extends PythonPlannerRunner[DataFrame](func) with AutoCloseable {

  private val datasetKey = PythonRunnerUtils.putNewObjectToPy4j(dataset)
  private val jscKey = PythonRunnerUtils.putNewObjectToPy4j(new JavaSparkContext(dataset.sparkSession.sparkContext))

  override protected val workerModule: String = "spark_rapids_ml.connect_plugin"

  override protected def writeToPython(dataOut: DataOutputStream, pickler: Pickler): Unit = {
    PythonRDD.writeUTF(PythonRunnerUtils.AUTH_TOKEN, dataOut)
    PythonRDD.writeUTF(transform.name, dataOut)
    PythonRDD.writeUTF(transform.params, dataOut)
    PythonRDD.writeUTF(jscKey, dataOut)
    PythonRDD.writeUTF(datasetKey, dataOut)
    PythonRDD.writeUTF(transform.modelAttributes, dataOut)
  }

  override protected def receiveFromPython(dataIn: DataInputStream): DataFrame = {
    // Read the dataset target id in py4j server
    val dfId = PythonWorkerUtils.readUTF(dataIn)
    PythonRunnerUtils.getObjectAndDeref(dfId).asInstanceOf[DataFrame]
  }

  override def close(): Unit = {
    PythonRunnerUtils.deleteObject(jscKey)
    PythonRunnerUtils.deleteObject(datasetKey)
  }
}
