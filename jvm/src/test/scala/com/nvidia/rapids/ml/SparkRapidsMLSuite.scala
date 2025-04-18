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

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.LabeledPoint

import java.io.File
import org.scalatest.BeforeAndAfterEach
import org.scalatest.funsuite.AnyFunSuite
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.rapids.{RapidsLogisticRegressionModel, RapidsUtils}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, rand, when}
import org.json4s._
import org.json4s.jackson.JsonMethods._

import scala.util.Random

class SparkRapidsMLSuite extends AnyFunSuite with BeforeAndAfterEach {
  @transient var ss: SparkSession = _
  @transient var _tempDir: File = _

  protected def tempDir: File = _tempDir

  override def beforeEach(): Unit = {
    try {
      ss = SparkSession.builder()
        .master("local[1]")
        // TODO add spark-rapids after spark-rapids has supported 4.0
//        .config("spark.sql.adaptive.enabled", "false")
//        .config("spark.stage.maxConsecutiveAttempts", "1")
//        .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
//        .config("spark.rapids.memory.gpu.pool", "none") // Disable RMM for unit tests.
        .appName("SparkRapidsML-connect-plugin")
        .getOrCreate()

      _tempDir = RapidsUtils.createTempDir(namePrefix = this.getClass.getName)

    } finally {
      super.beforeEach()
    }
  }

  override def afterEach(): Unit = {
    try {
      RapidsUtils.deleteRecursively(_tempDir)
      if (ss != null) {
        ss.stop()
        ss = null
      }
    } finally {
      super.afterEach()
    }
  }

  private def generateLogisticInput(offset: Double,
                                    scale: Double,
                                    nPoints: Int,
                                    seed: Int): Seq[LabeledPoint] = {
    val rnd = new Random(seed)
    val x1 = Array.fill[Double](nPoints)(rnd.nextGaussian())

    val y = (0 until nPoints).map { i =>
      val p = 1.0 / (1.0 + math.exp(-(offset + scale * x1(i))))
      if (rnd.nextDouble() < p) 1.0 else 0.0
    }

    val testData = (0 until nPoints).map(i => LabeledPoint(y(i), Vectors.dense(Array(x1(i)))))
    testData
  }

  test("CrossValidator") {
    val spark = ss
    import spark.implicits._
    val dataset = ss.sparkContext.parallelize(generateLogisticInput(1.0, 1.0, 100, 42), 2)
      .toDF("test_label", "test_feature")
    val dfWithRandom = dataset.repartition(1).withColumn("random", rand(100L))
    val foldCol = when(col("random") < 0.33, 0).when(col("random") < 0.66, 1).otherwise(2)
//    val datasetWithFold = dfWithRandom.withColumn("fold", foldCol).drop("random").repartition(2)

    val lr = new LogisticRegression()
      .setFeaturesCol("test_feature")
      .setLabelCol("test_label")

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.maxIter, Array(3, 11))
      .addGrid(lr.tol, Array(0.03, 0.11))
      .build()

    val rcv = new RapidsCrossValidator()
      .setEstimator(lr)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("test_label"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)
      .setParallelism(2)

    val model = rcv.fit(dfWithRandom)
    assert(model.bestModel.isInstanceOf[RapidsLogisticRegressionModel])
    val rlrm = model.bestModel.asInstanceOf[RapidsLogisticRegressionModel]
    assert(rlrm.getFeaturesCol == "test_feature")
    assert(rlrm.getLabelCol == "test_label")
    assert(model.getNumFolds == 2)
    model.transform(dfWithRandom).show()

  }

  test("RapidsLogisticRegression") {
    val df = ss.createDataFrame(
      Seq(
        (Vectors.dense(1.0, 2.0), 1.0f),
        (Vectors.dense(1.0, 3.0), 1.0f),
        (Vectors.dense(2.0, 1.0), 0.0f),
        (Vectors.dense(3.0, 1.0), 0.0f))
    ).toDF("test_feature", "class")

    val lr = new RapidsLogisticRegression()
      .setFeaturesCol("test_feature")
      .setLabelCol("class")
      .setMaxIter(23)
      .setTol(0.03)
    //    .setThreshold(0.51) this is going to fail due to spark-rapids-ml has mapped threshold to None

    val path = new File(tempDir, "LogisticRegression").getPath
    lr.write.overwrite().save(path)

    val loadedLr = RapidsLogisticRegression.load(path)
    assert(loadedLr.getFeaturesCol == "test_feature")
    assert(loadedLr.getTol == 0.03)
    assert(loadedLr.getLabelCol == "class")
    assert(loadedLr.getMaxIter == 23)

    def check(model: RapidsLogisticRegressionModel): Unit = {
      assert(model.getFeaturesCol == "test_feature")
      assert(model.getTol == 0.03)
      assert(model.getLabelCol == "class")
      assert(model.getMaxIter == 23)
    }

    val model = loadedLr.fit(df).asInstanceOf[RapidsLogisticRegressionModel]
    check(model)
    model.write.overwrite().save(path)
    val loadedModel = RapidsLogisticRegressionModel.load(path)
    check(loadedModel)

    assert(model.uid == loadedModel.uid)
    assert(model.modelAttributes == loadedModel.modelAttributes)

    // Transform using Spark-Rapids-ML model by default
    val dfGpu = model.transform(df)

    // Transform using CPU model by disabling "spark.rapids.ml.python.transform.enabled"
    df.sparkSession.conf.set("spark.rapids.ml.python.transform.enabled", "false")
    val dfCpu = model.transform(df)

    // The order of the column is different
    assert(!(dfGpu.schema.names sameElements dfCpu.schema.names))
    assert(dfGpu.schema.names.sorted sameElements dfCpu.schema.names.sorted)

    // No exception while collecting data for both CPU and GPU
    dfGpu.collect()
    dfCpu.collect()
  }
}
