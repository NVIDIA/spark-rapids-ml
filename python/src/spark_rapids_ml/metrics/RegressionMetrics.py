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
from typing import List, cast

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import DenseVector, Vectors


# This class is aligning with Spark SummarizerBuffer scala version
class _SummarizerBuffer:
    def __init__(
        self,
        mean: List[float],
        m2n: List[float],
        m2: List[float],
        l1: List[float],
        total_cnt: int,
    ):
        """All of the mean/m2n/m2/l1 have the same length which must be equal to 3,
        and the order of their values is [label, label-prediction, prediction]
        """
        self._curr_mean = mean
        self._curr_m2n = m2n
        self._curr_m2 = m2
        self._curr_l1 = l1
        self._num_cols = len(mean)
        # spark-rapids-ml doesn't support weight col, so default to 1 for each sample.
        self._total_cnt = total_cnt
        self._total_weight_sum = total_cnt
        self._weight_square_sum = total_cnt
        self._curr_weight_sum = [total_cnt] * self._num_cols

    def merge(self, other: "_SummarizerBuffer") -> "_SummarizerBuffer":
        """Merge the other into self and return a new SummarizerBuffer"""
        self._total_cnt += other._total_cnt
        self._total_weight_sum += other._total_weight_sum
        self._weight_square_sum += other._weight_square_sum

        for i in range(self._num_cols):
            thisWeightSum = self._curr_weight_sum[i]
            otherWeightSum = other._curr_weight_sum[i]
            totalWeightSum = thisWeightSum + otherWeightSum

            if totalWeightSum != 0.0:
                deltaMean = other._curr_mean[i] - self._curr_mean[i]
                # merge mean together
                self._curr_mean[i] += deltaMean * otherWeightSum / totalWeightSum
                # merge m2n together
                self._curr_m2n[i] += (
                    other._curr_m2n[i]
                    + deltaMean
                    * deltaMean
                    * thisWeightSum
                    * otherWeightSum
                    / totalWeightSum
                )

            self._curr_weight_sum[i] = totalWeightSum
            self._curr_m2[i] += other._curr_m2[i]
            self._curr_l1[i] += other._curr_l1[i]

        return _SummarizerBuffer(
            self._curr_mean,
            self._curr_m2n,
            self._curr_m2,
            self._curr_l1,
            self._total_cnt,
        )

    @property
    def total_count(self) -> int:
        return self._total_cnt

    @property
    def norm_l2(self) -> DenseVector:
        """L2 (Euclidean) norm of each dimension."""
        real_magnitude = [math.sqrt(m2) for m2 in self._curr_m2]
        return Vectors.dense(real_magnitude)

    @property
    def norm_l1(self) -> DenseVector:
        """L1 norm of each dimension."""
        return Vectors.dense(self._curr_l1)

    @property
    def mean(self) -> DenseVector:
        """mean of each dimension."""
        real_mean = [
            self._curr_mean[i] * (self._curr_weight_sum[i] / self._total_weight_sum)
            for i in range(self._num_cols)
        ]
        return Vectors.dense(real_mean)

    def _compute_variance(self) -> List[float]:
        denominator = self._total_weight_sum - (
            self._weight_square_sum / self._total_weight_sum
        )
        if denominator > 0.0:
            delta_mean = self._curr_mean
            real_variance = [
                max(
                    (
                        self._curr_m2n[i]
                        + delta_mean[i]
                        * delta_mean[i]
                        * self._curr_weight_sum[i]
                        * (self._total_weight_sum - self._curr_weight_sum[i])
                        / self._total_weight_sum
                    )
                    / denominator,
                    0.0,
                )
                for i in range(self._num_cols)
            ]
        else:
            real_variance = [0] * self._num_cols
        return real_variance

    @property
    def weight_sum(self) -> int:
        """Sum of weights."""
        return self._total_weight_sum

    @property
    def variance(self) -> DenseVector:
        """Unbiased estimate of sample variance of each dimension."""
        return Vectors.dense(self._compute_variance())


# This class is aligning with Spark RegressionMetrics scala version.
class RegressionMetrics:
    """Metrics for regression case."""

    def __init__(self, summary: _SummarizerBuffer):
        self._summary = summary

    @staticmethod
    def create(
        mean: List[float],
        m2n: List[float],
        m2: List[float],
        l1: List[float],
        total_cnt: int,
    ) -> "RegressionMetrics":
        return RegressionMetrics(_SummarizerBuffer(mean, m2n, m2, l1, total_cnt))

    def merge(self, other: "RegressionMetrics") -> "RegressionMetrics":
        """Merge other to self and return a new RegressionMetrics"""
        summary = self._summary.merge(other._summary)
        return RegressionMetrics(summary)

    @property
    def _ss_y(self) -> float:
        """sum of squares for label"""
        return math.pow(self._summary.norm_l2[0], 2)

    @property
    def _ss_err(self) -> float:
        """sum of squares for 'label-prediction'"""
        return math.pow(self._summary.norm_l2[1], 2)

    @property
    def _ss_tot(self) -> float:
        """total sum of squares"""
        variance = cast(float, self._summary.variance[0])
        return variance * (self._summary.weight_sum - 1)

    @property
    def _ss_reg(self) -> float:
        return (
            math.pow(self._summary.norm_l2[2], 2)  # type: ignore
            + math.pow(self._summary.mean[0], 2) * self._summary.weight_sum
            - 2
            * self._summary.mean[0]
            * self._summary.mean[2]
            * self._summary.weight_sum
        )

    @property
    def mean_squared_error(self) -> float:
        """Returns the mean squared error, which is a risk function corresponding to the
        expected value of the squared error loss or quadratic loss."""
        return self._ss_err / self._summary.weight_sum

    @property
    def root_mean_squared_error(self) -> float:
        """Returns the root mean squared error, which is defined as the square root of
        the mean squared error."""
        return math.sqrt(self.mean_squared_error)

    def r2(self, through_origin: bool) -> float:
        """Returns R^2^, the unadjusted coefficient of determination."""
        return (
            (1 - self._ss_err / self._ss_y)
            if through_origin
            else (1 - self._ss_err / self._ss_tot)
        )

    @property
    def mean_absolute_error(self) -> float:
        """Returns the mean absolute error, which is a risk function corresponding to the
        expected value of the absolute error loss or l1-norm loss."""
        norm_l1 = cast(float, self._summary.norm_l1[1])
        return norm_l1 / self._summary.weight_sum

    @property
    def explained_variance(self) -> float:
        """Returns the variance explained by regression.
        explained_variance = $\sum_i (\hat{y_i} - \bar{y})^2^ / n$"""
        return self._ss_reg / self._summary.weight_sum

    def evaluate(self, evaluator: RegressionEvaluator) -> float:
        metric_name = evaluator.getMetricName()

        if metric_name == "rmse":
            return self.root_mean_squared_error
        elif metric_name == "mse":
            return self.mean_squared_error
        elif metric_name == "r2":
            return self.r2(evaluator.getThroughOrigin())
        elif metric_name == "mae":
            return self.mean_absolute_error
        elif metric_name == "var":
            return self.explained_variance
        else:
            raise ValueError(f"Unsupported metric name, found {metric_name}")
