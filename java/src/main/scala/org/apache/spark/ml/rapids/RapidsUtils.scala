package org.apache.spark.ml.rapids

import org.apache.spark.ml.classification.LogisticRegressionModel

object RapidsUtils {

  def dummyLogisticRegressionModel: LogisticRegressionModel = new LogisticRegressionModel()

}
