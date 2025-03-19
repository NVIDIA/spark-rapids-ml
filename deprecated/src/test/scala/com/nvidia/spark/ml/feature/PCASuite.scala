/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.ml.feature

import org.apache.spark.ml.feature.RapidsPCAModel
import org.apache.spark.ml.linalg.{Vectors, _}
import org.apache.spark.ml.param.ParamsSuite
import org.apache.spark.ml.util.{DefaultReadWriteTest, MLTestingUtils, RapidsMLTest}
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.sql.{functions, Row}
import org.apache.spark.sql.functions.col

class PCASuite extends RapidsMLTest with DefaultReadWriteTest {

  import testImplicits._

  test("params") {
    ParamsSuite.checkParams(new PCA)
    val mat = Matrices.dense(2, 2, Array(0.0, 1.0, 2.0, 3.0)).asInstanceOf[DenseMatrix]
    val explainedVariance = Vectors.dense(0.5, 0.5).asInstanceOf[DenseVector]
    val model = new RapidsPCAModel("pca", mat, explainedVariance)
    ParamsSuite.checkParams(model)
  }

  // cannot run with transform as the output column is not vector now.
  test("pca using gpu") {
    val data = Seq(
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
      Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    )

    val data_gpu = Seq(
      Array(2.0, 0.0, 3.0, 4.0, 5.0),
      Array(0.0, 1.0, 0.0, 7.0, 0.0),
      Array(4.0, 0.0, 0.0, 6.0, 7.0)
    )

    val dataRDD = sc.parallelize(data, 2)
    val dataRDD_gpu = sc.parallelize(data_gpu,2)

    val mat = new RowMatrix(dataRDD.map(OldVectors.fromML))
    val pc = mat.computePrincipalComponents(3)
    val expected = mat.multiply(pc).rows.map(_.asML)

    val df = dataRDD_gpu.toDF("array_features")

    val pca = new PCA()
        .setInputCol("array_features")
        .setOutputCol("pca_features")
        .setK(3)

    val pcaModel = pca.fit(df)
    val transformed = pcaModel.transform(df)
    MLTestingUtils.checkCopyAndUids(pca, pcaModel)
    val convertToVector = functions.udf((array: Seq[Float]) => {
      Vectors.dense(array.map(_.toDouble).toArray)
    })
    val vectorDf = transformed.withColumn("pca_features_vec",
      convertToVector(col("pca_features")))
    val vec_col = vectorDf.select("pca_features_vec").rdd.map {
      case Row(v: Vector) => v
    }
    vec_col.zip(expected).map(tuple =>{
      assert(
          Vectors.dense(tuple._1.toArray.map(x=> x.abs))
          ~==
              Vectors.dense(tuple._2.toArray.map(x=> x.abs))
                  absTol 1e-5,
        "Transformed vector is different with expected vector.")
    })
  }


  test("PCA read/write") {
    val t = new PCA()
        .setInputCol("myInputCol")
        .setOutputCol("myOutputCol")
        .setK(3)
    testDefaultReadWrite(t)
  }

  test("PCAModel read/write") {
    val instance = new RapidsPCAModel("myPCAModel",
      Matrices.dense(2, 2, Array(0.0, 1.0, 2.0, 3.0)).asInstanceOf[DenseMatrix],
      Vectors.dense(0.5, 0.5).asInstanceOf[DenseVector])
    val newInstance = testDefaultReadWrite(instance)
    assert(newInstance.pc === instance.pc)
  }
}
