package com.nvidia.rapids.ml

trait Foo

object Bar extends Foo {
}

object ScalaMain {

  def main(args: Array[String]): Unit = {
    val barClass = Class.forName("com.nvidia.rapids.ml.Bar$")
    val fooClass = classOf[Foo]

    if (barClass.isAssignableFrom(fooClass)) {
      println("------------- true")
    } else {
      println("------------- false")
    }

  }

}
