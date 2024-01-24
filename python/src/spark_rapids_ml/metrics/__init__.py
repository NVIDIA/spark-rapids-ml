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

from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

# Global parameter used by core and subclasses.
TransformEvaluateMetric = namedtuple(
    "TransformEvaluateMetric", ("accuracy_like", "log_loss", "regression")
)
transform_evaluate_metric = TransformEvaluateMetric(
    "accuracy_like", "log_loss", "regression"
)


@dataclass
class EvalMetricInfo:
    """Class for holding info about
    Spark evaluators to be passed in to transform_evaluate local computations"""

    # MulticlassClassificationEvaluator
    eps: float = 1.0e-15  # logLoss
    # BinaryClassificationEvaluator - placeholder till we support
    numBins: int = 1000

    eval_metric: Optional[str] = None
