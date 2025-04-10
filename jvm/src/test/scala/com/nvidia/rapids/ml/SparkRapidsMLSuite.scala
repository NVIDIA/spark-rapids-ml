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

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.rapids.{RapidsLogisticRegressionModel, RapidsUtils}
import org.apache.spark.sql.SparkSession

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
}
