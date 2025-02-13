package org.apache.spark.ml.rapids

import net.razorvine.pickle.Pickler
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.api.python.{PythonFunction, PythonRDD, SimplePythonFunction}
import PythonRunner.AUTH_TOKEN
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.execution.python.PythonPlannerRunner
import py4j.GatewayServer.GatewayServerBuilder

import java.io.{DataInputStream, DataOutputStream}
import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters._

private object PythonRunner {
  val AUTH_TOKEN = "SPARK-RAPIDS-ML"

  val PYTHON_EXEC = "/home/bobwang/anaconda3/envs/rapids-24.10/bin/python"

  private lazy val gw: py4j.Gateway = {
    val server = new GatewayServerBuilder().authToken(AUTH_TOKEN).build()
    server.start()
    server.getGateway
  }

  def putNewObjectToPy4j(o: Object): String = {
    gw.putNewObject(o)
  }

  def deleteObject(key: String): Unit = {
    gw.deleteObject(key)
  }
}

class RapidsMLFunction extends SimplePythonFunction(
  command = Array[Byte](),
  envVars = Map(
    "PYSPARK_PYTHON" -> PythonRunner.PYTHON_EXEC, // TODO, how to get the PYSPARK_PYTHON?
    "PYSPARK_DRIVER_PYTHON" -> PythonRunner.PYTHON_EXEC,
  ).asJava,
  pythonIncludes = ArrayBuffer("").asJava,
  pythonExec = PythonRunner.PYTHON_EXEC,
  pythonVer = "3.10", // TODO, run the python process to get the python version.
  broadcastVars = List.empty.asJava,
  accumulator = null)

/**
 * PythonRunner is a bridge to launch/manage Python process. And it sends the
 * estimator related message to python process and run.
 *
 * @param name    estimator name, not the java qualification name. Eg, LogisticRegression
 * @param params  estimator parameters which has been encoded to json string
 * @param dataset input dataset
 */
class PythonRunner(name: String,
                   params: String,
                   dataset: DataFrame,
                   func: PythonFunction) extends PythonPlannerRunner[Int](func) with AutoCloseable {

  private val datasetKey = PythonRunner.putNewObjectToPy4j(dataset)
  private val jscKey = PythonRunner.putNewObjectToPy4j(new JavaSparkContext(dataset.sparkSession.sparkContext))

    override protected val workerModule: String = "spark_rapids_ml.connect_plugin"

  override protected def writeToPython(dataOut: DataOutputStream, pickler: Pickler): Unit = {
    println("in writeToPython")
    PythonRDD.writeUTF(AUTH_TOKEN, dataOut)
    PythonRDD.writeUTF(name, dataOut)
    PythonRDD.writeUTF(params, dataOut)
    PythonRDD.writeUTF(jscKey, dataOut)
    PythonRDD.writeUTF(datasetKey, dataOut)
  }

  override protected def receiveFromPython(dataIn: DataInputStream): Int = {
    println("in receiveFromPython ")
    1234
  }

  override def close(): Unit = {
    PythonRunner.deleteObject(jscKey)
    PythonRunner.deleteObject(datasetKey)
  }
}
