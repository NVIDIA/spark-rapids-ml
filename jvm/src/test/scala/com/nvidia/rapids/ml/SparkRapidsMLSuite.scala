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


import java.io.File
import org.scalatest.BeforeAndAfterEach
import org.scalatest.funsuite.AnyFunSuite
import org.apache.spark.ml.clustering.rapids.RapidsKMeansModel
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.rapids.{RapidsLinearRegressionModel, RapidsLogisticRegressionModel, RapidsPCAModel, RapidsRandomForestClassificationModel, RapidsRandomForestRegressionModel, RapidsUtils}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.ArrayType

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

  test("RapidsRandomForestClassifier") {
    val df = ss.createDataFrame(
      Seq(
        (Vectors.dense(1.0, 2.0), 1.0f),
        (Vectors.dense(1.0, 3.0), 1.0f),
        (Vectors.dense(2.0, 1.0), 0.0f),
        (Vectors.dense(3.0, 1.0), 0.0f))
    ).toDF("test_feature", "class")

    val rfc = new RapidsRandomForestClassifier()
      .setFeaturesCol("test_feature")
      .setLabelCol("class")
      .setMaxDepth(4)
      .setMaxBins(7)

    val path = new File(tempDir, "RapidsRandomForestClassifier").getPath
    rfc.write.overwrite().save(path)

    val loadedRfc = RapidsRandomForestClassifier.load(path)
    assert(loadedRfc.getFeaturesCol == "test_feature")
    assert(loadedRfc.getMaxDepth == 4)
    assert(loadedRfc.getLabelCol == "class")
    assert(loadedRfc.getMaxBins == 7)

    def check(model: RapidsRandomForestClassificationModel): Unit = {
      assert(model.getFeaturesCol == "test_feature")
      assert(model.getMaxDepth == 4)
      assert(model.getLabelCol == "class")
      assert(model.getMaxBins == 7)
    }

    val model = loadedRfc.fit(df).asInstanceOf[RapidsRandomForestClassificationModel]
    check(model)
    model.write.overwrite().save(path)
    val loadedModel = RapidsRandomForestClassificationModel.load(path)
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

  test("RapidsPCA") {
    val df = ss.createDataFrame(
      Seq(
        Tuple1(Vectors.dense(0.0, 1.0, 0.0, 7.0, 0.0)),
         Tuple1(Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0)),
        Tuple1(Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)),
      )).toDF("test_feature")

    val pca = new RapidsPCA()
      .setInputCol("test_feature")
      .setOutputCol("pca_feature")
      .setK(1)

    val path = new File(tempDir, "RapidsPCA").getPath
    pca.write.overwrite().save(path)

    val loadedPca = RapidsPCA.load(path)
    assert(loadedPca.getInputCol == "test_feature")
    assert(loadedPca.getOutputCol == "pca_feature")
    assert(loadedPca.getK == 1)

    def check(model: RapidsPCAModel): Unit = {
      assert(model.getInputCol == "test_feature")
      assert(model.getOutputCol == "pca_feature")
      assert(model.getK == 1)
    }

    val model = loadedPca.fit(df)
    check(model)
    model.write.overwrite().save(path)
    val loadedModel = RapidsPCAModel.load(path)
    check(loadedModel)

    assert(model.uid == loadedModel.uid)
    assert(model.modelAttributes == loadedModel.modelAttributes)

    // Transform using Spark-Rapids-ML model by default
    val dfGpu = model.transform(df)

    // Transform using CPU model by disabling "spark.rapids.ml.python.transform.enabled"
    df.sparkSession.conf.set("spark.rapids.ml.python.transform.enabled", "false")
    val dfCpu = model.transform(df)

    assert(dfGpu.schema.names.sorted sameElements dfCpu.schema.names.sorted)

    // No exception while collecting data for both CPU and GPU
    dfGpu.collect()
    dfCpu.collect()
  }

  test("RapidsRandomForestRegressor") {
    val df = ss.createDataFrame(
      Seq(
        (Vectors.dense(1.0, 2.0), 1.0f),
        (Vectors.dense(1.0, 3.0), 1.0f),
        (Vectors.dense(2.0, 1.0), 0.0f),
        (Vectors.dense(3.0, 1.0), 0.0f))
    ).toDF("test_feature", "value")

    val rfc = new RapidsRandomForestRegressor()
      .setFeaturesCol("test_feature")
      .setLabelCol("value")
      .setMaxDepth(4)
      .setMaxBins(7)

    val path = new File(tempDir, "RapidsRandomForestRegressor").getPath
    rfc.write.overwrite().save(path)

    val loadedRfc = RapidsRandomForestRegressor.load(path)
    assert(loadedRfc.getFeaturesCol == "test_feature")
    assert(loadedRfc.getMaxDepth == 4)
    assert(loadedRfc.getLabelCol == "value")
    assert(loadedRfc.getMaxBins == 7)

    def check(model: RapidsRandomForestRegressionModel): Unit = {
      assert(model.getFeaturesCol == "test_feature")
      assert(model.getMaxDepth == 4)
      assert(model.getLabelCol == "value")
      assert(model.getMaxBins == 7)
    }

    val model = loadedRfc.fit(df).asInstanceOf[RapidsRandomForestRegressionModel]
    check(model)
    model.write.overwrite().save(path)
    val loadedModel = RapidsRandomForestRegressionModel.load(path)
    check(loadedModel)

    assert(model.uid == loadedModel.uid)
    assert(model.modelAttributes == loadedModel.modelAttributes)

    // Transform using Spark-Rapids-ML model by default
    val dfGpu = model.transform(df)

    // Transform using CPU model by disabling "spark.rapids.ml.python.transform.enabled"
    df.sparkSession.conf.set("spark.rapids.ml.python.transform.enabled", "false")
    val dfCpu = model.transform(df)

    assert(dfGpu.schema.names.sorted sameElements dfCpu.schema.names.sorted)

    // No exception while collecting data for both CPU and GPU
    dfGpu.collect()
    dfCpu.collect()
  }

  test("RapidsLinearRegression") {
    val df = ss.createDataFrame(
      Seq(
        (Vectors.dense(1.0, 2.0), 1.0f),
        (Vectors.dense(1.0, 3.0), 1.0f),
        (Vectors.dense(2.0, 1.0), 0.0f),
        (Vectors.dense(3.0, 1.0), 0.0f))
    ).toDF("test_feature", "value")

    val rlr = new RapidsLinearRegression()
      .setFeaturesCol("test_feature")
      .setLabelCol("value")
      .setMaxIter(7)
      .setTol(0.00003)


    val path = new File(tempDir, "RapidsLinearRegression").getPath
    rlr.write.overwrite().save(path)

    val loadedRlr = RapidsLinearRegression.load(path)
    assert(loadedRlr.getFeaturesCol == "test_feature")
    assert(loadedRlr.getMaxIter == 7)
    assert(loadedRlr.getLabelCol == "value")
    assert(loadedRlr.getTol == 0.00003)

    def check(model: RapidsLinearRegressionModel): Unit = {
      assert(model.getFeaturesCol == "test_feature")
      assert(model.getMaxIter == 7)
      assert(model.getLabelCol == "value")
      assert(model.getTol == 0.00003)
    }

    val model = loadedRlr.fit(df).asInstanceOf[RapidsLinearRegressionModel]
    check(model)
    model.write.overwrite().save(path)
    val loadedModel = RapidsLinearRegressionModel.load(path)
    check(loadedModel)

    assert(model.uid == loadedModel.uid)
    assert(model.modelAttributes == loadedModel.modelAttributes)

    // Transform using Spark-Rapids-ML model by default
    val dfGpu = model.transform(df)

    // Transform using CPU model by disabling "spark.rapids.ml.python.transform.enabled"
    df.sparkSession.conf.set("spark.rapids.ml.python.transform.enabled", "false")
    val dfCpu = model.transform(df)

    assert(dfGpu.schema.names.sorted sameElements dfCpu.schema.names.sorted)

    // No exception while collecting data for both CPU and GPU
    dfGpu.collect()
    dfCpu.collect()
  }

  test("RapidsKMeans") {
    val df = ss.createDataFrame(
      Seq(
        Tuple1(Vectors.dense(0.0, 0.0)),
        Tuple1(Vectors.dense(1.0, 1.0)),
        Tuple1(Vectors.dense(9.0, 8.0)),
        Tuple1(Vectors.dense(8.0, 9.0)))
    ).toDF("test_feature")

    val kmeans = new RapidsKMeans()
      .setFeaturesCol("test_feature")
      .setK(2)
      .setMaxIter(7)
      .setTol(0.00003)


    val path = new File(tempDir, "RapidsKmeans").getPath
    kmeans.write.overwrite().save(path)

    val loadedRKmeans = RapidsKMeans.load(path)
    assert(loadedRKmeans.getFeaturesCol == "test_feature")
    assert(loadedRKmeans.getK == 2)
    assert(loadedRKmeans.getMaxIter == 7)
    assert(loadedRKmeans.getTol == 0.00003)

    def check(model: RapidsKMeansModel): Unit = {
      assert(model.getFeaturesCol == "test_feature")
      assert(model.getMaxIter == 7)
      assert(model.getK == 2)
      assert(model.getTol == 0.00003)
      assert(Seq(Vectors.dense(8.5, 8.5), Vectors.dense(0.5, 0.5)).contains(model.clusterCenters(0)))
      assert(Seq(Vectors.dense(8.5, 8.5), Vectors.dense(0.5, 0.5)).contains(model.clusterCenters(1)))
    }

    val model = loadedRKmeans.fit(df)
    check(model)
    model.write.overwrite().save(path)
    val loadedModel = RapidsKMeansModel.load(path)
    check(loadedModel)

    assert(model.uid == loadedModel.uid)
    assert(model.modelAttributes == loadedModel.modelAttributes)

    // Transform using Spark-Rapids-ML model by default
    val dfGpu = model.transform(df)

    // Transform using CPU model by disabling "spark.rapids.ml.python.transform.enabled"
    df.sparkSession.conf.set("spark.rapids.ml.python.transform.enabled", "false")
    val dfCpu = model.transform(df)

    assert(dfGpu.schema.names.sorted sameElements dfCpu.schema.names.sorted)

    // No exception while collecting data for both CPU and GPU
    dfGpu.collect()
    dfCpu.collect()
  }

  test("array input") {
    val df = ss.createDataFrame(Seq(
      (1.0f, Seq(1.0, 3.0)),
      (0.0f, Seq(2.0, 1.0)),
      (0.0f, Seq(3.0, 1.0)),
    )).toDF("value", "test_feature")

    assert(df.schema("test_feature").dataType.isInstanceOf[ArrayType])

    val rfc = new RapidsRandomForestRegressor()
      .setFeaturesCol("test_feature")
      .setLabelCol("value")
      .setMaxDepth(4)
      .setMaxBins(7)

    val model = rfc.fit(df)
    val dfGpu = model.transform(df)
    // transform on Array input without any issue.
    dfGpu.collect()

    df.sparkSession.conf.set("spark.rapids.ml.python.transform.enabled", "false")

    val thrown = intercept[IllegalArgumentException] {
      model.transform(df).collect()
    }

    assert(thrown.getMessage.contains("Please enable spark.rapids.ml.python.transform.enabled " +
      "to transform dataset in python for non-vector input"))

  }
}
