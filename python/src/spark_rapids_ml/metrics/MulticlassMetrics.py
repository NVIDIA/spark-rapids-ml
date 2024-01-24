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

from typing import Dict

import numpy as np
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# sklearn's version will not support fixed eps starting v1.5
def log_loss(labels: np.ndarray, probs: np.ndarray, eps: float) -> float:
    if np.any(labels < 0) or np.any(labels > probs.shape[1] - 1):
        raise ValueError(f"labels must be in the range [0,{probs.shape[1]-1}]")
    if np.any(probs < 0) or np.any(probs > 1.0):
        raise ValueError("probs must be in the range [0.0, 1.0]")
    probs_for_labels = probs[np.arange(probs.shape[0]), labels.astype(np.int32)]
    probs_for_labels = np.maximum(probs_for_labels, eps)
    return sum(-np.log(probs_for_labels))


class MulticlassMetrics:
    """Metrics for multiclass classification."""

    SUPPORTED_MULTI_CLASS_METRIC_NAMES = [
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
        "logLoss",
    ]

    # This class is aligning with MulticlassMetrics scala version.

    def __init__(
        self,
        tp: Dict[float, float] = {},
        fp: Dict[float, float] = {},
        label: Dict[float, float] = {},
        label_count: int = 0,
        log_loss: float = -1,
    ) -> None:
        self._tp_by_class = tp
        self._fp_by_class = fp
        self._label_count_by_class = label
        self._label_count = label_count
        self._log_loss = log_loss

    def _precision(self, label: float) -> float:
        """Returns precision for a given label (category)"""
        tp = self._tp_by_class[label]
        fp = self._fp_by_class[label]
        return 0.0 if (tp + fp == 0) else tp / (tp + fp)

    def _recall(self, label: float) -> float:
        """Returns recall for a given label (category)"""
        return self._tp_by_class[label] / self._label_count_by_class[label]

    def _f_measure(self, label: float, beta: float = 1.0) -> float:
        """Returns f-measure for a given label (category)"""
        p = self._precision(label)
        r = self._recall(label)
        beta_sqrd = beta * beta
        return 0.0 if (p + r == 0) else (1 + beta_sqrd) * p * r / (beta_sqrd * p + r)

    def false_positive_rate(self, label: float) -> float:
        """Returns false positive rate for a given label (category)"""
        fp = self._fp_by_class[label]
        return fp / (self._label_count - self._label_count_by_class[label])

    def weighted_fmeasure(self, beta: float = 1.0) -> float:
        """Returns weighted averaged f1-measure"""
        sum = 0.0
        for k, v in self._label_count_by_class.items():
            sum += self._f_measure(k, beta) * v / self._label_count
        return sum

    def accuracy(self) -> float:
        """Returns accuracy (equals to the total number of correctly classified instances
        out of the total number of instances.)"""
        return sum(self._tp_by_class.values()) / self._label_count

    def weighted_precision(self) -> float:
        """Returns weighted averaged precision"""
        return sum(
            [
                self._precision(category) * count / self._label_count
                for category, count in self._label_count_by_class.items()
            ]
        )

    def weighted_recall(self) -> float:
        """Returns weighted averaged recall (equals to precision, recall and f-measure)"""
        return sum(
            [
                self._recall(category) * count / self._label_count
                for category, count in self._label_count_by_class.items()
            ]
        )

    def weighted_true_positive_rate(self) -> float:
        """Returns weighted true positive rate. (equals to precision, recall and f-measure)"""
        return self.weighted_recall()

    def weighted_false_positive_rate(self) -> float:
        """Returns weighted false positive rate"""
        return sum(
            [
                self.false_positive_rate(category) * count / self._label_count
                for category, count in self._label_count_by_class.items()
            ]
        )

    def true_positive_rate_by_label(self, label: float) -> float:
        """Returns true positive rate for a given label (category)"""
        return self._recall(label)

    def hamming_loss(self) -> float:
        """Returns Hamming-loss"""
        numerator = sum(self._fp_by_class.values())
        denominator = self._label_count
        return numerator / denominator

    def log_loss(self) -> float:
        """Returns log loss"""
        return self._log_loss / self._label_count

    def evaluate(self, evaluator: MulticlassClassificationEvaluator) -> float:
        metric_name = evaluator.getMetricName()
        if metric_name == "f1":
            return self.weighted_fmeasure()
        elif metric_name == "accuracy":
            return self.accuracy()
        elif metric_name == "weightedPrecision":
            return self.weighted_precision()
        elif metric_name == "weightedRecall":
            return self.weighted_recall()
        elif metric_name == "weightedTruePositiveRate":
            return self.weighted_true_positive_rate()
        elif metric_name == "weightedFalsePositiveRate":
            return self.weighted_false_positive_rate()
        elif metric_name == "weightedFMeasure":
            return self.weighted_fmeasure(evaluator.getBeta())
        elif metric_name == "truePositiveRateByLabel":
            return self.true_positive_rate_by_label(evaluator.getMetricLabel())
        elif metric_name == "falsePositiveRateByLabel":
            return self.false_positive_rate(evaluator.getMetricLabel())
        elif metric_name == "precisionByLabel":
            return self._precision(evaluator.getMetricLabel())
        elif metric_name == "recallByLabel":
            return self._recall(evaluator.getMetricLabel())
        elif metric_name == "fMeasureByLabel":
            return self._f_measure(evaluator.getMetricLabel(), evaluator.getBeta())
        elif metric_name == "hammingLoss":
            return self.hamming_loss()
        elif metric_name == "logLoss":
            return self.log_loss()
        else:
            raise ValueError(f"Unsupported metric name, found {metric_name}")
