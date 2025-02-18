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
    val coefficients = MLSerDe.loads(pickledCoefficients)
    val pickledIntercept: Array[Byte] = PythonWorkerUtils.readBytes(dataIn)
    val intercepts = MLSerDe.loads(pickledIntercept)
    val multinomial = dataIn.readInt()
    val isMultinomial: Boolean = multinomial == 1
//    new LogisticRegressionModel(uid, coefficients, intercepts, numClasses, isMultinomial)
    new LogisticRegressionModel()
  }

  def getUserDefinedParams(instance: Params): String = {
    compact(render(instance.paramMap.toSeq.map { case ParamPair(p, v) =>
      p.name -> parse(p.jsonEncode(v))
    }.toList))
  }

}
