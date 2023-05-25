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

from typing import Dict


class MulticlassMetrics:
    """Metrics for multiclass classification."""

    # This class is aligning with MulticlassMetrics scala version.

    def __init__(
        self,
        num_class: int,
        tp: Dict[float, float],
        fp: Dict[float, float],
        label: Dict[float, float],
        label_count: int,
    ) -> None:
        if num_class <= 2:
            raise RuntimeError(
                f"MulticlassMetrics requires at least 3 classes. Found {num_class}"
            )

        self._num_classes = 3
        self._tp_by_class = tp
        self._fp_by_class = fp
        self._label_count_by_class = label
        self._label_count = label_count

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

    def weighted_fmeasure(self, beta: float = 1.0) -> float:
        """Returns weighted averaged f1-measure"""
        sum = 0.0
        for k, v in self._label_count_by_class.items():
            sum += self._f_measure(k, beta) * v / self._label_count
        return sum
