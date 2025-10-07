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

package org.apache.spark.ml.rapids

import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.mllib.clustering.{KMeansModel => MLlibKMeansModel}
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Matrices, Matrix, Vector, Vectors}
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.tree.{ContinuousSplit, InternalNode, LeafNode, Node}
import org.apache.spark.mllib.tree.impurity.{EntropyCalculator, GiniCalculator, ImpurityCalculator, VarianceCalculator}
import org.json4s.{DefaultFormats, Formats}
import org.json4s.JsonAST.{JArray, _}
import org.json4s.jackson.JsonMethods.parse

object ModelHelper {

  implicit val formats: Formats = DefaultFormats

  private def createCalcAndPred(impurity: String,
                                instanceCount: Int,
                                stats: Array[Double] = Array.empty): (ImpurityCalculator, Double) = {
    impurity match {
      case "gini" => (new GiniCalculator(stats, instanceCount), Vectors.dense(stats).argmax)
      case "entropy" => (new EntropyCalculator(stats, instanceCount), Vectors.dense(stats).argmax)
      case "variance" => (new VarianceCalculator(Array.fill(3)(0), instanceCount), stats(0))
      case _ => throw new RuntimeException(s"Unsupported impurity: $impurity")
    }
  }

  private def getNodes(json: JValue, impurity: String): List[Node] = {
    json match {
      case JArray(arr) => arr.map(e => parseTreeNode(e, impurity))
      case _ => throw new IllegalArgumentException(s"Expected JArray of TreeLite trees, found $json")
    }
  }

  private def parseTreeNode(treeliteJson: JValue, impurity: String): Node = {
    // Extract the nodes array from treelite JSON
    val nodes = treeliteJson \ "nodes" match {
      case JArray(arr) => arr
      case _ => throw new RuntimeException("Expected 'nodes' array in treelite JSON")
    }
    
    // Create a map from node_id to node data for efficient lookup
    val nodeMap = nodes.map { node =>
      val nodeId = (node \ "node_id").extract[Int]
      nodeId -> node
    }.toMap
    
    // Parse starting from root node (node_id = 0)
    parseTreeNodeRecursive(nodeMap, 0, impurity)
  }
  
  private def parseTreeNodeRecursive(nodeMap: Map[Int, JValue], nodeId: Int, impurity: String): Node = {
    val node = nodeMap(nodeId)
    
    node match {
      case JObject(fields) =>
        val fieldMap = fields.toMap
        
        // Check if this is an internal node (has split_feature_id)
        if (fieldMap.contains("split_feature_id")) {
          val leftChildId = fieldMap("left_child").extract[Int]
          val rightChildId = fieldMap("right_child").extract[Int]
          val gain = fieldMap.get("gain").map(_.extract[Double]).getOrElse(0.0)
          
          val split = new ContinuousSplit(
            featureIndex = fieldMap("split_feature_id").extract[Int],
            threshold = fieldMap("threshold").extract[Double])
          
          val instanceCount = fieldMap.get("instance_count").map(_.extract[Int]).getOrElse(1)
          val leafValue = Array.fill(instanceCount)(0.0)
          val (calc, _) = createCalcAndPred(impurity, instanceCount, leafValue)
          
          new InternalNode(
            0.0, // prediction value is no-use for internal node, just fake it
            0.0, // impurity value is no-use for internal node. just fake it
            gain,
            parseTreeNodeRecursive(nodeMap, leftChildId, impurity),
            parseTreeNodeRecursive(nodeMap, rightChildId, impurity),
            split,
            calc)
        } 
        // Check if this is a leaf node (has leaf_value)
        else if (fieldMap.contains("leaf_value")) {
          val leafValue = fieldMap("leaf_value") match {
            case JDouble(value) => Array(value)
            case JArray(arr) => arr.map(_.extract[Double]).toArray
            case _ => throw new RuntimeException(s"Unexpected leaf_value format: ${fieldMap("leaf_value")}")
          }
          
          val instanceCount = fieldMap.get("instance_count").map(_.extract[Int]).getOrElse(1)
          val (calc, pred) = createCalcAndPred(impurity, instanceCount, leafValue)
          
          new LeafNode(pred, 0.0, calc)
        } else {
          throw new RuntimeException(s"Unknown node type in treelite JSON: $node")
        }
      case _ => throw new RuntimeException(s"Failed to parse treelite node: $node")
    }
  }

  private def parseTreeLiteTrees(treeLiteModelJson: String): JValue = {
    parse(treeLiteModelJson) match {
        case JObject(fields) => {
            val fieldMap = fields.toMap
            fieldMap("trees")
        }
        case _ => throw new RuntimeException(s"Failed to find trees in treelite model json")
    }
  }

  def createRandomForestRegressionModel(modelAttributes: String,
                                        impurity: String,
                                        uid: String): (Array[DecisionTreeRegressionModel], Int) = {
    val parsedJson = parse(modelAttributes)
    implicit val format = DefaultFormats
    val (treeModelJson, numFeatures) = parsedJson match {
      case JArray(arr) => (arr.last.extract[String], arr.head.extract[Int])
      case _ => throw new IllegalArgumentException(s"Failed to parse $parsedJson")
    }
    val treeLiteTrees = parseTreeLiteTrees(treeModelJson)
    val nodes = getNodes(treeLiteTrees, impurity)
    val trees = nodes.map(n => new DecisionTreeRegressionModel(uid, n, numFeatures)).toArray
    (trees, numFeatures)
  }

  def createRandomForestClassificationModel(modelAttributes: String,
                                            impurity: String,
                                            uid: String): (Array[DecisionTreeClassificationModel], Int, Int) = {
    val parsedJson = parse(modelAttributes)
    implicit val format = DefaultFormats
    val (treeModelJson, numFeatures, numClasses) = parsedJson match {
      case JArray(arr) =>
        (arr.dropRight(1).last.extract[String], arr.head.extract[Int], arr.last.extract[Int])
      case _ => throw new IllegalArgumentException(s"Failed to parse $parsedJson")
    }
    val treeLiteTrees = parseTreeLiteTrees(treeModelJson)
    val nodes = getNodes(treeLiteTrees, impurity)
    val trees = nodes.map(n => new DecisionTreeClassificationModel(uid, n, numFeatures, numClasses)).toArray
    (trees, numFeatures, numClasses)
  }

  def createLinearRegressionModel(modelAttributes: String): (Vector, Double, Double) = {
    val parsedJson = parse(modelAttributes)
    implicit val format = DefaultFormats
    parsedJson match {
      case JArray(arr) =>
        val coef = arr.head.extract[List[Double]].toArray
        val intercept = arr(1).extract[Double]
        (Vectors.dense(coef), intercept, 1.0)
      case _ => throw new IllegalArgumentException(s"Failed to parse $parsedJson")
    }
  }

  def createLogisticRegressionModel(modelAttributes: String): (Matrix, Vector, Int) = {
    val parsedJson = parse(modelAttributes)
    implicit val format = DefaultFormats
    parsedJson match {
      case JArray(arr) =>
        val coef = arr.head.extract[List[List[Double]]]
        val rows = coef.length
        val cols = coef.head.length
        val coefMatrix = new DenseMatrix(rows, cols, coef.flatten.toArray, true)
        val intercept = arr(1).extract[List[Double]].toArray
        val numClasses = arr(2).extract[List[Double]].length
        (coefMatrix, Vectors.dense(intercept), numClasses)
      case _ => throw new IllegalArgumentException(s"Failed to parse $parsedJson")
    }
  }

  def createPCAModel(modelAttributes: String): (DenseMatrix, DenseVector) = {
    val parsedJson = parse(modelAttributes)
    implicit val format = DefaultFormats
    parsedJson match {
      case JArray(arr) =>
        val pc = arr(1).extract[List[List[Double]]]
        val rows = pc.length
        val cols = arr(4).extract[Int]
        // DenseMatrix is column major, so flip rows/cols
        val pcMatrix = Matrices.dense(cols, rows, pc.flatten.toArray).asInstanceOf[DenseMatrix]
        val explainedVariance = arr(2).extract[List[Double]].toArray
        (pcMatrix, Vectors.dense(explainedVariance).asInstanceOf[DenseVector])
      case _ => throw new IllegalArgumentException(s"Failed to parse $parsedJson")
    }
  }

  def createKMeansModel(modelAttributes: String): MLlibKMeansModel = {
    val parsedJson = parse(modelAttributes)
    implicit val format = DefaultFormats
    parsedJson match {
      case JArray(arr) =>
        val clusterCenters = arr.head.extract[List[List[Double]]]
        val clusterCentersVectors = clusterCenters.map(v =>
          org.apache.spark.mllib.linalg.Vectors.dense(v.toArray)).toArray
        new MLlibKMeansModel(clusterCentersVectors)
      case _ => throw new IllegalArgumentException(s"Failed to parse $parsedJson")
    }
  }
}
