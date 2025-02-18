package com.nvidia.rapids.ml

import org.apache.commons.logging.LogFactory
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.rapids.{Fit, PythonRunner, RapidsMLFunction, RapidsUtils}


class RapidsLogisticRegression(override val uid: String) extends LogisticRegression with RapidsEstimator {

  private val logger = LogFactory.getLog("com.nvidia.rapids.ml.RapidsLogisticRegression")

  def this() = this(Identifiable.randomUID("logreg"))

  override def train(dataset: Dataset[_]): LogisticRegressionModel = {
    logger.info("Bobby train in SparkRapidsML library.")

    // Get the user-defined parameters and pass them to python process as a dictionary
    //    val params = this.paramMap
    val params = RapidsUtils.getUserDefinedParams(this)
    println(s"--------------------parameters of lr -------- ${params}")

    // TODO get the parameters (coefficients and intercepts) and construct the LogisticRegressionModel
    withResource(
      new PythonRunner(
        Fit(estimatorName, uid, params),
        dataset.toDF,
        RapidsUtils.buildLogisticRegressionModel,
        new RapidsMLFunction())) { runner =>
      runner.runInPython(useDaemon = false)
    }.asInstanceOf[LogisticRegressionModel]
  }


  /**
   * The estimator name
   */
  override def estimatorName: String = "LogisticRegression"
}
