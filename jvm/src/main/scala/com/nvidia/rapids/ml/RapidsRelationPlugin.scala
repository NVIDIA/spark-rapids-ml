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

import org.apache.commons.logging.LogFactory
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.Row
import org.apache.spark.sql.connect.planner.SparkConnectPlanner
import org.apache.spark.sql.connect.plugin.RelationPlugin
import org.apache.spark.connect.{proto => sparkProto}
import org.apache.spark.sql.connect.ml.rapids.RapidsConnectUtils
import org.apache.spark.sql.types.{StringType, StructField, StructType}

import java.util.Optional
import scala.jdk.CollectionConverters.SeqHasAsJava

class RapidsRelationPlugin extends RelationPlugin {
  protected val logger = LogFactory.getLog("Spark-Rapids-ML RapidsRelationPlugin")

  override def transform(bytes: Array[Byte], sparkConnectPlanner: SparkConnectPlanner): Optional[LogicalPlan] = {
    logger.info("In RapidsRelationPlugin")

    val rel = com.google.protobuf.Any.parseFrom(bytes)
    val sparkSession = sparkConnectPlanner.session

    // CrossValidation
    if (rel.is(classOf[proto.CrossValidatorRelation])) {
      val cvProto = rel.unpack(classOf[proto.CrossValidatorRelation])
      val dataLogicalPlan = sparkProto.Plan.parseFrom(cvProto.getDataset.toByteArray)
      val dataset = RapidsConnectUtils.ofRows(sparkSession,
        sparkConnectPlanner.transformRelation(dataLogicalPlan.getRoot))
      val cvModel = RapidsCrossValidator.fit(cvProto, dataset)
      val modelId = RapidsConnectUtils.cache(sparkConnectPlanner.sessionHolder, cvModel.bestModel)
      val resultDf = sparkSession.createDataFrame(
        List(Row(s"$modelId")).asJava,
        StructType(Seq(StructField("best_model_id", StringType))))
      Optional.of(RapidsConnectUtils.getLogicalPlan(resultDf))
    } else {
      Optional.empty()
    }
  }
}
