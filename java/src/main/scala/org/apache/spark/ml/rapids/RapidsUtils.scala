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

import org.apache.spark.api.python.PythonWorkerUtils
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.param.{ParamPair, Params}
import org.apache.spark.ml.python.MLSerDe
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{compact, parse, render}
import org.apache.spark.ml.linalg.{DenseMatrix, Vector}

import java.io.DataInputStream

object RapidsUtils {

  def buildLogisticRegressionModel(uid: String, dataIn: DataInputStream): LogisticRegressionModel = {
    val numClasses = dataIn.readInt()
    val pickledCoefficients: Array[Byte] = PythonWorkerUtils.readBytes(dataIn)
    val coefficients = MLSerDe.loads(pickledCoefficients).asInstanceOf[DenseMatrix]
    val pickledIntercept: Array[Byte] = PythonWorkerUtils.readBytes(dataIn)
    val intercepts = MLSerDe.loads(pickledIntercept).asInstanceOf[Vector]
    val multinomial = dataIn.readInt()
    val isMultinomial: Boolean = multinomial == 1
    new LogisticRegressionModel(uid, coefficients, intercepts, numClasses, isMultinomial)
  }

  def getUserDefinedParams(instance: Params): String = {
    compact(render(instance.paramMap.toSeq.map { case ParamPair(p, v) =>
      p.name -> parse(p.jsonEncode(v))
    }.toList))
  }

}
