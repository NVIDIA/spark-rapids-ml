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
import math

import numpy as np
import pandas as pd
import pytest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator

from spark_rapids_ml.metrics.MulticlassMetrics import MulticlassMetrics
from spark_rapids_ml.metrics.RegressionMetrics import RegressionMetrics

from .sparksession import CleanSparkSession


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
    MulticlassMetrics.SUPPORTED_MULTI_CLASS_METRIC_NAMES,
)
def test_multi_class_metrics(
    num_classes: int,
    metric_name: str,
) -> None:
    columns = ["prediction", "label"]
    np.random.seed(10)
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
        assert math.fabs(evaluator.evaluate(sdf) - metrics.evaluate((evaluator))) < 1e-6


def get_regression_metrics(
    pdf: pd.DataFrame, label_name: str, prediction_name: str
) -> RegressionMetrics:
    pdf = pdf.copy(True)
    pdf.insert(1, "gap", pdf[label_name] - pdf[prediction_name])
    mean = pdf.mean()
    m2 = pdf.pow(2).sum()
    l1 = pdf.abs().sum()
    sum = pdf.sum()
    total_cnt = pdf.shape[0]
    m2n = m2 - sum * sum / pdf.shape[0]

    return RegressionMetrics.create(mean, m2n, m2, l1, total_cnt)


@pytest.mark.parametrize("metric_name", ["rmse", "mse", "r2", "mae", "var"])
def test_regression_metrics(metric_name: str) -> None:
    columns = ["label", "prediction"]
    np.random.seed(10)
    pdf1 = pd.DataFrame(
        np.random.uniform(low=-20, high=20, size=(1000, 2)), columns=columns
    ).astype(np.float64)
    np.random.seed(100)
    pdf2 = pd.DataFrame(
        np.random.uniform(low=-20, high=20, size=(1000, 2)), columns=columns
    ).astype(np.float64)

    metrics1 = get_regression_metrics(pdf1, columns[0], columns[1])
    metrics2 = get_regression_metrics(pdf2, columns[0], columns[1])

    metrics = metrics1.merge(metrics2)
    pdf = pd.concat([pdf1, pdf2])

    with CleanSparkSession() as spark:
        sdf = spark.createDataFrame(
            pdf.to_numpy().tolist(), ", ".join([f"{n} double" for n in columns])
        )
        evaluator = RegressionEvaluator(
            predictionCol="prediction",
            labelCol="label",
        )
        evaluator.setMetricName(metric_name)  # type: ignore
        assert math.fabs(evaluator.evaluate(sdf) - metrics.evaluate((evaluator))) < 1e-6
