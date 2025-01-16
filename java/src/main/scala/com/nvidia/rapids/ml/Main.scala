package com.nvidia.rapids.ml

import org.apache.spark.ml.Estimator
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.rapids.SparkInternal
import org.apache.spark.sql.SparkSession
import org.apache.spark.util.Utils

import java.util.ServiceLoader
import scala.collection.mutable

object Main {

  def main(args: Array[String]): Unit = {
    println("----")


    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("DeveloperApiExample")
      .getOrCreate()

    val x = SparkInternal.loadOperators(classOf[Estimator[_]])
    println("sdfadsa")

//    println("++++++++++++++++++++++++++++++++++++++++++++++++")
//    SparkInternal.estimators.foreach { case (key, value) =>
//      println("---------------------- " + key)
//    }

//    val estName = "com.nvidia.rapids.ml.LogisticRegression"
//
//    val loadedMethod = SparkInternal.estimators.get(estName).get.getMethod("load", classOf[String])
//    loadedMethod.invoke(null, "/tmp/abcd")

//    // Prepare training data.
//    val training = spark.createDataFrame(spark.sparkContext.parallelize(Seq(
//      LabeledPoint(1.0, Vectors.dense(0.0, 1.1, 0.1)),
//      LabeledPoint(0.0, Vectors.dense(2.0, 1.0, -1.0)),
//      LabeledPoint(0.0, Vectors.dense(2.0, 1.3, 1.0)),
//      LabeledPoint(1.0, Vectors.dense(0.0, 1.2, -0.5))))).toDF("label", "features")
//
//    training.show()
//
//    val lr = new LogisticRegression()
//    val model =lr.train(training)
//    println("xxxxxxxxxxxxxxxxxxx => " + model.intercept)
//    println("xxxxxxxxxxxxxxxxxxx => " + model.coefficientMatrix)
  }

}
