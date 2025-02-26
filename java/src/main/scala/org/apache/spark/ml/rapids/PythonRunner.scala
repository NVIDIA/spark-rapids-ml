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

import net.razorvine.pickle.Pickler
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.api.python.{PythonFunction, PythonRDD, SimplePythonFunction}
import PythonRunner.AUTH_TOKEN
import org.apache.spark.ml.Model
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.execution.python.PythonPlannerRunner

import java.util.Base64
import py4j.GatewayServer.GatewayServerBuilder

import java.io.{DataInputStream, DataOutputStream}
import java.security.SecureRandom
import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters._
import scala.sys.process.Process


private[this] object PythonRunner {
  private def generateSecrets = {
    val rnd = new SecureRandom()
    val token = new Array[Byte](32)
    rnd.nextBytes(token)
    Base64.getEncoder.encodeToString(token)
  }

  lazy val AUTH_TOKEN: String = generateSecrets

  private lazy val RAPIDS_PYTHON_FUNC = {
    val defaultPythonExec: String = sys.env.getOrElse(
      "PYSPARK_DRIVER_PYTHON", sys.env.getOrElse("PYSPARK_PYTHON", "python3"))
    val pythonVer: String =
      Process(
        Seq(defaultPythonExec, "-c", "import sys; print('%d.%d' % sys.version_info[:2])")).!!.trim()

    new SimplePythonFunction(
      command = Array[Byte](),
      envVars = Map(
        "PYSPARK_PYTHON" -> defaultPythonExec,
        "PYSPARK_DRIVER_PYTHON" -> defaultPythonExec,
      ).asJava,
      pythonIncludes = ArrayBuffer("").asJava,
      pythonExec = defaultPythonExec,
      pythonVer = pythonVer,
      broadcastVars = List.empty.asJava,
      accumulator = null
    )
  }

  private val gwLock = new Object() // Lock object

  private lazy val gw: py4j.Gateway = gwLock.synchronized {
    val server = new GatewayServerBuilder().authToken(AUTH_TOKEN).build()
    server.start()
    server.getGateway
  }

  def putNewObjectToPy4j(o: Object): String = gwLock.synchronized {
    gw.putNewObject(o)
  }

  def deleteObject(key: String): Unit = gwLock.synchronized {
    gw.deleteObject(key)
  }
}

case class Fit(name: String, uid: String, params: String)

/**
 * PythonRunner is a bridge to launch/manage Python process. And it sends the
 * estimator related message to python process and run.
 *
 * @param fit     the estimator information
 * @param dataset input dataset
 */
class PythonRunner(fit: Fit,
                   dataset: DataFrame,
                   callBack: (String, DataInputStream) => Model[_],
                   func: PythonFunction = PythonRunner.RAPIDS_PYTHON_FUNC)
  extends PythonPlannerRunner[Model[_]](func) with AutoCloseable {

  private val datasetKey = PythonRunner.putNewObjectToPy4j(dataset)
  private val jscKey = PythonRunner.putNewObjectToPy4j(new JavaSparkContext(dataset.sparkSession.sparkContext))

  override protected val workerModule: String = "spark_rapids_ml.connect_plugin"

  override protected def writeToPython(dataOut: DataOutputStream, pickler: Pickler): Unit = {
    println(s"in writeToPython ${fit.name}")
    PythonRDD.writeUTF(AUTH_TOKEN, dataOut)
    PythonRDD.writeUTF(fit.name, dataOut)
    PythonRDD.writeUTF(fit.params, dataOut)
    PythonRDD.writeUTF(jscKey, dataOut)
    PythonRDD.writeUTF(datasetKey, dataOut)
  }

  override protected def receiveFromPython(dataIn: DataInputStream): Model[_] = {
    callBack(fit.uid, dataIn)
  }

  override def close(): Unit = {
    PythonRunner.deleteObject(jscKey)
    PythonRunner.deleteObject(datasetKey)
  }
}
