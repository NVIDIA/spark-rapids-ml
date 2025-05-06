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

package com.nvidia.rapids.ml

import org.apache.spark.ml.Estimator
import org.apache.spark.ml.evaluation.{Evaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.rapids.{Fit, PythonEstimatorRunner, RapidsUtils, TrainedModel}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.connect.ml.rapids.RapidsConnectUtils

class RapidsCrossValidator(override val uid: String) extends CrossValidator with RapidsEstimator {

  def this() = this(Identifiable.randomUID("cv"))

  override def fit(dataset: Dataset[_]): CrossValidatorModel = {
    val trainedModel = trainOnPython(dataset)

    val bestModel = RapidsUtils.createModel(getName(getEstimator.getClass.getName),
      getEstimator.uid, getEstimator, trainedModel)
    copyValues(RapidsUtils.createCrossValidatorModel(this.uid, bestModel))
  }

  private def getName(name: String): String = {
    RapidsUtils.transform(name).getOrElse(name)
  }

  /**
   * The estimator name
   *
   * @return
   */
  override def name: String = "CrossValidator"

  override def trainOnPython(dataset: Dataset[_]): TrainedModel = {
    logger.info(s"Training $name ...")

    val estimatorName = getName(getEstimator.getClass.getName)
    // TODO estimator could be a PipeLine which contains multiple stages.
    val cvParams = RapidsUtils.getJson(Map(
      "estimator" -> RapidsUtils.getUserDefinedParams(getEstimator,
        extra = Map(
          "estimator_name" -> estimatorName,
          "uid" -> getEstimator.uid)),
      "evaluator" -> RapidsUtils.getUserDefinedParams(getEvaluator,
        extra = Map(
          "evaluator_name" -> getName(getEvaluator.getClass.getName),
          "uid" -> getEvaluator.uid)),
      "estimatorParaMaps" -> RapidsUtils.getEstimatorParamMapsJson(getEstimatorParamMaps),
      "cv" -> RapidsUtils.getUserDefinedParams(this,
        List("estimator", "evaluator", "estimatorParamMaps"))
    ))
    val runner = new PythonEstimatorRunner(
      Fit(name, cvParams),
      dataset.toDF)

    val trainedModel = Arm.withResource(runner) { _ =>
      runner.runInPython(useDaemon = false)
    }

    logger.info(s"Finished $name training.")
    trainedModel
  }
}

object RapidsCrossValidator {

  def fit(cvProto: proto.CrossValidatorRelation, dataset: Dataset[_]): CrossValidatorModel = {

    val estProto = cvProto.getEstimator
    var estimator: Option[Estimator[_]] = None
    if (estProto.getName == "LogisticRegression") {
      estimator = Some(new RapidsLogisticRegression(uid = estProto.getUid))
      val estParams = estProto.getParams
      RapidsUtils.setParams(estimator.get, estParams)

    }
    val evalProto = cvProto.getEvaluator
    var evaluator: Option[Evaluator] = None
    if (evalProto.getName == "MulticlassClassificationEvaluator") {
      evaluator = Some(new MulticlassClassificationEvaluator(uid = evalProto.getUid))
      val evalParams = evalProto.getParams
      RapidsUtils.setParams(evaluator.get, evalParams)
    }

    val cv = new RapidsCrossValidator(uid = cvProto.getUid)
    RapidsUtils.setParams(cv, cvProto.getParams)

    cv.setEstimator(estimator.get).setEvaluator(evaluator.get)
    val paramGrid = RapidsUtils.extractParamMap(cv, cvProto.getEstimatorParamMaps)
    cv.setEstimatorParamMaps(paramGrid)
    cv.fit(dataset)
  }
}
