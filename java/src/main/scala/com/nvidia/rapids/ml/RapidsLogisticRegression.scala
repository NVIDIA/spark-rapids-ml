package com.nvidia.rapids.ml

import org.apache.commons.logging.LogFactory
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.rapids.{PythonRunner, RapidsMLFunction, RapidsUtils}


class RapidsLogisticRegression(override val uid: String) extends LogisticRegression with Rapids {

  private val logger = LogFactory.getLog("com.nvidia.rapids.ml.RapidsLogisticRegression")

  def this() = this(Identifiable.randomUID("logreg"))

  override def train(dataset: Dataset[_]): LogisticRegressionModel = {
    logger.info("Bobby train in SparkRapidsML library.")

    withResource(
      new PythonRunner("LogisticRegression",
        Map.empty, dataset.toDF,
        new RapidsMLFunction())) { runner =>
      runner.runInPython(useDaemon = true)
    }
    RapidsUtils.dummyLogisticRegressionModel
  }

}
