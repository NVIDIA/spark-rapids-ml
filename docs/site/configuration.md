---
title: Configuration
nav_order: 6
---
# Configuration

The following configurations can be supplied as Spark properties.

| Property name                     | Default | Meaning                                                                                                                                                                                                                                                                                                                                                                                        |
|:----------------------------------|:--------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| spark.rapids.ml.uvm.enabled       | false   | if set to true, enables [CUDA unified virtual memory](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/) (aka managed memory) during estimator.fit() operations to allow processing of larger datasets than would fit in GPU memory                                                                                                                                             |
| spark.rapids.ml.sam.enabled       | false   | if set to true, enables System Allocated Memory (SAM) on [HMM](https://developer.nvidia.com/blog/simplifying-gpu-application-development-with-heterogeneous-memory-management/) or [ATS](https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/) systems during estimator.fit() operations to allow processing of larger datasets than would fit in GPU memory |
| spark.rapids.ml.sam.headroom      | None    | when using System Allocated Memory (SAM) and GPU memory is oversubscribed, we may need to reserve some GPU memory as headroom to allow other CUDA calls to function without running out memory. Set a size appropriate for your application                                                                                                                                                    |
| spark.executorEnv.CUPY_ENABLE_SAM | 0       | if set to 1, enables System Allocated Memory (SAM) for CuPy operations. This enabled CuPy to work with SAM, and also avoid unnecessary memory coping                                                                                                                                                                                                                                           |

