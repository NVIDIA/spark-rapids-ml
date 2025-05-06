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
import org.apache.spark.ml.Model
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.param.{ParamMap, ParamPair, Params}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel}
import org.apache.spark.ml.util.MetaAlgorithmReadWrite
import org.apache.spark.util.ArrayImplicits.SparkArrayOps
import org.apache.spark.util.Utils
import org.json4s.{DefaultFormats, JObject, JString}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{compact, parse, render}

object RapidsUtils {

  def transform(name: String): Option[String] = {
    name match {
      case "org.apache.spark.ml.classification.LogisticRegression" =>
        Some("com.nvidia.rapids.ml.RapidsLogisticRegression")
      case "org.apache.spark.ml.classification.LogisticRegressionModel" =>
        Some("org.apache.spark.ml.rapids.RapidsLogisticRegressionModel")
      case _ => None
    }
  }

  // Just copy the user defined parameters
  def copyParams[T <: Params, S <: Params](src: S, to: T): T = {
    src.extractParamMap().toSeq.foreach { p =>
      val name = p.param.name
      if (to.hasParam(name) && src.isSet(p.param)) {
        to.set(to.getParam(name), p.value)
      }
    }
    to
  }

  def createModel(name: String, uid: String, src: Params, trainedModel: TrainedModel): Model[_] = {
    if (name.contains("LogisticRegression")) {
      val cpuModel = copyParams(src, trainedModel.model.asInstanceOf[LogisticRegressionModel])
      val isMultinomial = cpuModel.numClasses != 2
      copyParams(src, new RapidsLogisticRegressionModel(uid, cpuModel, trainedModel.modelAttributes, isMultinomial))
    } else {
      throw new RuntimeException(s"$name Not supported")
    }
  }

  def extractParamMap(cv: CrossValidator, parameters: String): Array[ParamMap] = {
    val evaluator = cv.getEvaluator
    val estimator = cv.getEstimator
    val uidToParams = Map(evaluator.uid -> evaluator) ++ MetaAlgorithmReadWrite.getUidMap(estimator)
    val paraMap = parse(parameters)

    implicit val format = DefaultFormats
    paraMap.extract[Seq[Seq[Map[String, String]]]].map {
        pMap =>
          val paramPairs = pMap.map { pInfo: Map[String, String] =>
            val est = uidToParams(pInfo("parent"))
            val param = est.getParam(pInfo("name"))
              val value = param.jsonDecode(pInfo("value"))
              param -> value
            }
          ParamMap(paramPairs: _*)
      }.toArray
  }

  def setParams(
                 instance: Params,
                 parameters: String): Unit = {
    implicit val format = DefaultFormats
    val paramsToSet = parse(parameters)
    paramsToSet match {
      case JObject(pairs) =>
        pairs.foreach { case (paramName, jsonValue) =>
          val param = instance.getParam(paramName)
          val value = param.jsonDecode(compact(render(jsonValue)))
          instance.set(param, value)
        }
      case _ =>
        throw new IllegalArgumentException(
          s"Cannot recognize JSON metadata: ${parameters}.")
    }
  }

  def createCrossValidatorModel(uid: String, model: Model[_]): CrossValidatorModel = {
    new CrossValidatorModel(uid, model, Array.empty[Double])
  }

  def getUserDefinedParams(instance: Params,
                           skipParams: List[String] = List.empty,
                           extra: Map[String, String] = Map.empty): String = {
    compact(render(
      instance.paramMap.toSeq
        .filter { case ParamPair(p, _) => !skipParams.contains(p.name) }
        .map { case ParamPair(p, v) =>
          p.name -> parse(p.jsonEncode(v))
        }.toList ++ extra.map { case (k, v) => k -> JString(v) }.toList
    ))
  }

  def getEstimatorParamMapsJson(estimatorParamMaps: Array[ParamMap]): String = {
    compact(render(
      estimatorParamMaps.map { paramMap =>
        paramMap.toSeq.map { case ParamPair(p, v) =>
          Map("parent" -> JString(p.parent),
            "name" -> JString(p.name),
            "value" -> parse(p.jsonEncode(v)))
        }
      }.toImmutableArraySeq
    ))
  }

  def getJson(params: Map[String, String] = Map.empty): String = {
    compact(render(
      params.map { case (k, v) => k -> parse(v) }.toList
    ))
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
