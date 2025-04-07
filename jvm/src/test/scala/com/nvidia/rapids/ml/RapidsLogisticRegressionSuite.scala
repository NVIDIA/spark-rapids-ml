package com.nvidia.rapids.ml

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite


class RapidsLogisticRegressionSuite extends AnyFunSuite {

  test("RapidsLogisticRegression parameters") {
    val spark = SparkSession.builder().master("local[1]").getOrCreate()

    val df = spark.createDataFrame(
      Seq(
        (Vectors.dense(1.0, 2.0), 1.0f),
        (Vectors.dense(1.0, 3.0), 1.0f),
        (Vectors.dense(2.0, 1.0), 0.0f),
        (Vectors.dense(3.0, 1.0), 0.0f))
    ).toDF("test_feature", "class")

    val lr = new RapidsLogisticRegression().setFeaturesCol("test_feature").setLabelCol("class")

    val model = lr.fit(df)


  }

}
