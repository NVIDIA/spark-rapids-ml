package org.apache.spark.sql.connect.ml.rapids

import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.classic.{Dataset, SparkSession}
import org.apache.spark.sql.connect.service.SessionHolder

object RapidsConnectUtils {

  def ofRows(session: SparkSession, logicalPlan: LogicalPlan): DataFrame = {
    Dataset.ofRows(session, logicalPlan)
  }

  def getLogicalPlan(df: Dataset[_]): LogicalPlan = df.logicalPlan

  def cache(sessionHolder: SessionHolder, model: Object): String = {
    sessionHolder.mlCache.register(model)
  }
}
