package com.nvidia.rapids.ml

import org.apache.spark.sql.connect.plugin.MLBackendPlugin

import java.util.Optional

class Plugin extends MLBackendPlugin {

  override def transform(mlName: String): Optional[String] = {
    mlName match {
      case "org.apache.spark.ml.classification.LogisticRegression" =>
        Optional.of("com.nvidia.rapids.ml.RapidsLogisticRegression")
      case _ => Optional.empty()
    }
  }
}
