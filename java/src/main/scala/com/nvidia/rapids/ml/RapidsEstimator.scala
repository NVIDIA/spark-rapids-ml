package com.nvidia.rapids.ml

trait RapidsEstimator {

  /**
   * The estimator name
   * @return
   */
  def estimatorName: String

  /** Executes the provided code block and then closes the resource */
  def withResource[T <: AutoCloseable, V](r: T)(block: T => V): V = {
    try {
      block(r)
    } finally {
      r.close()
    }
  }

}
