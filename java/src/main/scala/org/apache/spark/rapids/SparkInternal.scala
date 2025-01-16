package org.apache.spark.rapids

import org.apache.spark.ml.Estimator
import org.apache.spark.util.Utils

import java.util.ServiceLoader
import java.util.stream.Collectors
import scala.collection.mutable
import scala.jdk.CollectionConverters.{IterableHasAsScala, MapHasAsScala}

object SparkInternal {

  def loadOperators(mlCls: Class[_]): mutable.HashMap[String, Class[_]] = {
    val loader = Utils.getContextOrSparkClassLoader
    val serviceLoader = ServiceLoader.load(mlCls, loader)
    val providers = serviceLoader.asScala.toList
    mutable.HashMap.from(providers.map(est => est.getClass.getName -> est.getClass).toMap)
  }

//  val estimators = loadOperators(classOf[Estimator[_]])

  def loadOperatorsWithoutInstantiating(mlCls: Class[_]): Map[String, Class[_]] = {
    val loader = Utils.getContextOrSparkClassLoader
    val serviceLoader = ServiceLoader.load(mlCls, loader)
//    val providers = serviceLoader.asScala.toList
//    providers.map(est => est.getClass.getName -> est.getClass).toMap
    // Instead of using the iterator, we use the "stream()" method that allows
    // to iterate over a collection of providers that do not instantiate the class
    // directly. Since there is no good way to convert a Java stream to a Scala stream,
    // we collect the Java stream to a Java map and then convert it to a Scala map.
    serviceLoader
      .stream()
      .collect(
        Collectors.toMap(
          (est: ServiceLoader.Provider[_]) => est.`type`().getName,
          (est: ServiceLoader.Provider[_]) => est.`type`()))
      .asScala
      .toMap
  }
}
