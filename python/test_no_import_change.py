
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
# to run on gpu -> pythom -m spark_rapids_ml test_no_import_change.py 0.2
#
# notice no imports from spark_rapids_ml
#

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator
from pyspark.sql import SparkSession
from pyspark.ml.functions import array_to_vector
import sys

reg_param=float(sys.argv[1])

data = [
    ([1.0, 2.0], 1.0),
    ([1.0, 3.0], 1.0),
    ([2.0, 1.0], 0.0),
    ([3.0, 1.0], 0.0),
]
schema = "features array<float>, label float"

spark = SparkSession.builder.getOrCreate()
df = spark.createDataFrame(data, schema=schema)
df = df.select(array_to_vector("features").alias("features"), "label")
df.show()

lr_estimator = LogisticRegression()
print(lr_estimator.setFeaturesCol("features"))

print(lr_estimator.setLabelCol("label"))

print(lr_estimator.setRegParam(reg_param))
    
lr_model = lr_estimator.fit(df)
print(lr_model.coefficients)
    
print(lr_model.intercept)

cv = CrossValidator()
cv.set

from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import ParamGridBuilder, CrossValidatorModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.classification import RandomForestClassifier
import tempfile

dataset = spark.createDataFrame(
    [(Vectors.dense([0.0]), 0.0),
     (Vectors.dense([0.4]), 1.0),
     (Vectors.dense([0.5]), 0.0),
     (Vectors.dense([0.6]), 2.0),
     (Vectors.dense([1.0]), 1.0)] * 10,
    ["features", "label"])
rfc = RandomForestClassifier()
grid = ParamGridBuilder().addGrid(rfc.maxBins, [8, 16]).build()
evaluator = MulticlassClassificationEvaluator()
cv = CrossValidator(estimator=rfc, estimatorParamMaps=grid, evaluator=evaluator,
                    parallelism=2)
cvModel = cv.fit(dataset)

print(f"cvModel.getNumFolds(): {cvModel.getNumFolds()}")
print(f"cvModel.avgMetrics[0]: {cvModel.avgMetrics[0]}")
print(f"evaluator.evaluate(cvModel.transform(dataset)): {evaluator.evaluate(cvModel.transform(dataset))}")
path = tempfile.mkdtemp()
model_path = path + "/model"
cvModel.write().save(model_path)
cvModelRead = CrossValidatorModel.read().load(model_path)
print(f"cvModelRead.avgMetrics: {cvModelRead.avgMetrics}")
print(f"evaluator.evaluate(cvModel.transform(dataset)): {evaluator.evaluate(cvModel.transform(dataset))}")
print(f"evaluator.evaluate(cvModelRead.transform(dataset)): {evaluator.evaluate(cvModelRead.transform(dataset))}")

