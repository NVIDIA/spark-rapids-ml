#
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#
# to run on cpu -> python test_no_import_change.py 0.2
# to run on gpu -> python -m spark_rapids_ml test_no_import_change.py 0.2
#
# spark-submit based
#
# to run on cpu -> spark-submit --master local[1] test_no_import_change.py 0.2
# to run on gpu -> spark-rapids-submit --master local[1] test_no_import_change.py 0.2
#
# notice no imports from spark_rapids_ml (except for verifying model types)
#

import sys
import tempfile

from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.functions import array_to_vector
from pyspark.ml.tuning import CrossValidator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

reg_param = float(sys.argv[1])

data = [
    ([1.0, 2.0], 1.0),
    ([1.0, 3.0], 1.0),
    ([2.0, 1.0], 0.0),
    ([3.0, 1.0], 0.0),
]
schema = "features array<float>, label float"

spark = SparkSession.builder.getOrCreate()
df = spark.createDataFrame(data, schema=schema)
df = df.select(array_to_vector(col("features")).alias("features"), "label")
df.show()

lr_estimator = LogisticRegression()
print(lr_estimator.setFeaturesCol("features"))

print(lr_estimator.setLabelCol("label"))

print(lr_estimator.setRegParam(reg_param))

print(lr_estimator.setStandardization(False))

lr_model = lr_estimator.fit(df)

from spark_rapids_ml.classification import (
    LogisticRegressionModel as RapidsLogisticRegressionModel,
)

assert isinstance(lr_model, LogisticRegressionModel)

if "spark_rapids_ml.install" in sys.modules.keys():
    assert isinstance(lr_model, RapidsLogisticRegressionModel)

print(lr_model.coefficients)

print(lr_model.intercept)

path = tempfile.mkdtemp()
lr_model_path = path + "/lr_model"
lr_model.write().save(lr_model_path)
loaded_lr_model = LogisticRegressionModel.load(lr_model_path)
assert isinstance(loaded_lr_model, LogisticRegressionModel)

if "spark_rapids_ml.install" in sys.modules.keys():
    assert isinstance(loaded_lr_model, RapidsLogisticRegressionModel)


cv = CrossValidator()
cv.set

from pyspark.ml.classification import (
    RandomForestClassificationModel,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder

dataset = spark.createDataFrame(
    [
        (Vectors.dense([0.0]), 0.0),
        (Vectors.dense([0.4]), 1.0),
        (Vectors.dense([0.5]), 0.0),
        (Vectors.dense([0.6]), 2.0),
        (Vectors.dense([1.0]), 1.0),
    ]
    * 10,
    ["features", "label"],
)
rfc = RandomForestClassifier()
grid = ParamGridBuilder().addGrid(rfc.maxBins, [8, 16]).build()
evaluator = MulticlassClassificationEvaluator()
cv = CrossValidator(
    estimator=rfc, estimatorParamMaps=grid, evaluator=evaluator, parallelism=2
)
cvModel = cv.fit(dataset)

assert isinstance(cvModel.bestModel, RandomForestClassificationModel)

from spark_rapids_ml.classification import (
    RandomForestClassificationModel as RapidsRandomForestClassificationModel,
)

if "spark_rapids_ml.install" in sys.modules.keys():
    assert isinstance(cvModel.bestModel, RapidsRandomForestClassificationModel)

print(f"cvModel.getNumFolds(): {cvModel.getNumFolds()}")
print(f"cvModel.avgMetrics[0]: {cvModel.avgMetrics[0]}")
print(
    f"evaluator.evaluate(cvModel.transform(dataset)): {evaluator.evaluate(cvModel.transform(dataset))}"
)

cv_model_path = path + "/cv_model"
cvModel.write().save(cv_model_path)
cvModelRead = CrossValidatorModel.read().load(cv_model_path)

assert isinstance(cvModelRead.bestModel, RandomForestClassificationModel)

if "spark_rapids_ml.install" in sys.modules.keys():
    assert isinstance(cvModelRead.bestModel, RapidsRandomForestClassificationModel)

print(f"cvModelRead.avgMetrics: {cvModelRead.avgMetrics}")
print(
    f"evaluator.evaluate(cvModel.transform(dataset)): {evaluator.evaluate(cvModel.transform(dataset))}"
)
print(
    f"evaluator.evaluate(cvModelRead.transform(dataset)): {evaluator.evaluate(cvModelRead.transform(dataset))}"
)
