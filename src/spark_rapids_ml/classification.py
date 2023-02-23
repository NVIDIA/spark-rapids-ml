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
from typing import Any, Type, Union

from pyspark import Row
from pyspark.ml.classification import _RandomForestClassifierParams
from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType, FloatType, IntegerType, IntegralType

from spark_rapids_ml.core import alias
from spark_rapids_ml.tree import (
    _RandomForestClass,
    _RandomForestCumlParams,
    _RandomForestEstimator,
    _RandomForestModel,
)


class RandomForestClassifier(
    _RandomForestClass,
    _RandomForestEstimator,
    _RandomForestCumlParams,
    _RandomForestClassifierParams,
):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.set_params(**kwargs)

    def setNumTrees(self, value: int) -> "RandomForestClassifier":
        """
        Sets the value of :py:attr:`numTrees`.
        """
        return self._set(numTrees=value)

    def _pre_process_label(
        self, dataset: DataFrame, feature_type: Union[Type[FloatType], Type[DoubleType]]
    ) -> Column:
        """Cuml RandomForestClassifier requires the int32 type of label column"""
        label_name = self.getLabelCol()
        label_datatype = dataset.schema[label_name].dataType
        if isinstance(label_datatype, (IntegralType, FloatType, DoubleType)):
            label_col = col(label_name).cast(IntegerType()).alias(alias.label)
        else:
            raise ValueError(
                "Label column must be integral types or float/double types."
            )

        return label_col

    def _create_pyspark_model(self, result: Row) -> "RandomForestClassificationModel":
        return RandomForestClassificationModel.from_row(result)

    def _is_classification(self) -> bool:
        return True


class RandomForestClassificationModel(
    _RandomForestClass,
    _RandomForestModel,
    _RandomForestCumlParams,
    _RandomForestClassifierParams,
):
    def _is_classification(self) -> bool:
        return True

    @property
    def hasSummary(self) -> bool:
        """Indicates whether a training summary exists for this model instance."""
        return False

    @property
    def numClasses(self) -> int:
        """Number of classes (values which the label can take)."""
        raise NotImplementedError
