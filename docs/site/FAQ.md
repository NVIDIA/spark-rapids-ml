---
title: Frequently Asked Questions
nav_order: 4
---
# Frequently Asked Questions

* TOC
{:toc}

### What versions of Apache Spark are supported?

Apache Spark version 3.3.1 or higher.

### What versions of Python are supported

Python 3.10 or higher.

### How do I fix the "java.lang.IllegalArgumentException: valueCount must be >= 0" error?

This error occurs when the product of Arrow batch size and row dimension exceeds 2,147,483,647 (INT32_MAX), typically with very wide datasets (many features per row), causing Arrow serialization to fail. For example, if you set `max_records_per_batch = 10000` and your data has `row_dimension = 300000` (i.e., 300,000 features per row), then `10000 × 300000 = 3,000,000,000`, which exceeds the Arrow limit of 2,147,483,647 (INT32_MAX) and will cause this error.

Be aware that some Spark Rapids ML algorithms (such as NearestNeighbors) may convert sparse vectors to dense format internally if the underlying cuML algorithm does not support sparse input. This conversion can significantly increase memory usage, especially with wide datasets, and may make the Arrow size limit error more likely. To mitigate this, lower the value of `spark.sql.execution.arrow.maxRecordsPerBatch` (for example, to 5,000 or less) so that the product of the batch size and the number of elements per row stays within Arrow's maximum allowed size.
