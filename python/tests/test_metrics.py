#
# Copyright (c) 2023, NVIDIA CORPORATION.
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

import numpy as np
import pandas as pd
import pytest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from python.tests.sparksession import CleanSparkSession
from spark_rapids_ml.metrics.MulticlassMetrics import MulticlassMetrics


def get_multi_class_metrics(pdf: pd.DataFrame, num_classes: int) -> MulticlassMetrics:
    confusion = pdf.groupby(["label", "prediction"]).size().reset_index(name="total")

    tp_by_class = {}
    fp_by_class = {}
    label_count_by_class = {}
    label_count = 0

    for i in range(num_classes):
        tp_by_class[float(i)] = 0.0
        label_count_by_class[float(i)] = 0.0
        fp_by_class[float(i)] = 0.0

    for index, row in confusion.iterrows():
        label_count += row.total
        label_count_by_class[row.label] += row.total

        if row.label == row.prediction:
            tp_by_class[row.label] += row.total
        else:
            fp_by_class[row.prediction] += row.total

    return MulticlassMetrics(
        tp=tp_by_class,
        fp=fp_by_class,
        label=label_count_by_class,
        label_count=label_count,
    )


@pytest.mark.parametrize("num_classes", [4])
@pytest.mark.parametrize(
    "metric_name",
    [
        "f1",
        "accuracy",
        "weightedPrecision",
        "weightedRecall",
        "weightedTruePositiveRate",
        "weightedFalsePositiveRate",
        "weightedFMeasure",
        "truePositiveRateByLabel",
        "falsePositiveRateByLabel",
        "precisionByLabel",
        "recallByLabel",
        "fMeasureByLabel",
        "hammingLoss",
    ],
)
def test_multi_class_metrics(
    num_classes: int,
    metric_name: str,
) -> None:
    columns = ["prediction", "label"]
    pdf = pd.DataFrame(
        np.random.randint(0, num_classes, size=(1000, 2)), columns=columns
    ).astype(np.float64)

    metrics = get_multi_class_metrics(pdf, num_classes)

    with CleanSparkSession() as spark:
        sdf = spark.createDataFrame(
            pdf.to_numpy().tolist(), ", ".join([f"{n} double" for n in columns])
        )
        evaluator = MulticlassClassificationEvaluator(
            predictionCol="prediction",
            labelCol="label",
        )

        evaluator.setMetricName(metric_name)  # type: ignore
        print(evaluator.evaluate(sdf))
        print(metrics.evaluate(evaluator))
