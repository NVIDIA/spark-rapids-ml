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

import java.security.SecureRandom
import java.util.Base64
import java.io.File

import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters._
import scala.sys.process.Process

import py4j.GatewayServer.GatewayServerBuilder
import org.apache.spark.api.python.SimplePythonFunction
import org.apache.spark.ml.param.{ParamPair, Params}
import org.apache.spark.util.Utils
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{compact, parse, render}

object RapidsUtils {

  def getUserDefinedParams(instance: Params): String = {
    compact(render(instance.paramMap.toSeq.map { case ParamPair(p, v) =>
      p.name -> parse(p.jsonEncode(v))
    }.toList))
  }

  def createTempDir(namePrefix: String = "spark"): File = {
    Utils.createTempDir(namePrefix)
  }

  def deleteRecursively(file: File): Unit = {
    Utils.deleteRecursively(file)
  }

}

object PythonRunnerUtils {
  private def generateSecrets = {
    val rnd = new SecureRandom()
    val token = new Array[Byte](32)
    rnd.nextBytes(token)
    Base64.getEncoder.encodeToString(token)
  }

  private[rapids] lazy val AUTH_TOKEN: String = generateSecrets

  private[rapids] lazy val RAPIDS_PYTHON_FUNC = {
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

  /**
   * Get the model from py4j server and remove its reference in py4j server
   */
  def getObjectAndDeref(id: String): Object = gwLock.synchronized {
    val o = gw.getObject(id)
    gw.deleteObject(id)
    o
  }
}
