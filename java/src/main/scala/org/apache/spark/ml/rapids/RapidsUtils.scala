package org.apache.spark.ml.rapids

import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.param.{ParamPair, Params}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{parse, render, compact}

object RapidsUtils {

  def dummyLogisticRegressionModel: LogisticRegressionModel = new LogisticRegressionModel()

  def getUserDefinedParams(instance: Params): String = {
    compact(render(instance.paramMap.toSeq.map { case ParamPair(p, v) =>
      p.name -> parse(p.jsonEncode(v))
    }.toList))
  }

}
