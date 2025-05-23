#
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
from typing import Optional, Tuple

from pyspark.sql import DataFrame

from spark_rapids_ml.core import (
    EvalMetricInfo,
    _ConstructFunc,
    _EvaluateFunc,
    _TransformFunc,
)
from spark_rapids_ml.knn import ApproximateNearestNeighborsModel


class CPUNearestNeighborsModel(ApproximateNearestNeighborsModel):
    def __init__(self, item_df: DataFrame):
        super().__init__(item_df)

    def kneighbors(
        self, query_df: DataFrame, sort_knn_df_by_query_id: bool = True
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        self._item_df_withid = self._ensureIdCol(self._item_df_withid)
        return super().kneighbors(
            query_df, sort_knn_df_by_query_id=sort_knn_df_by_query_id
        )

    def _get_cuml_transform_func(
        self, dataset: DataFrame, eval_metric_info: Optional[EvalMetricInfo] = None
    ) -> Tuple[
        _ConstructFunc,
        _TransformFunc,
        Optional[_EvaluateFunc],
    ]:
        self._cuml_params["algorithm"] = "brute"
        _, _transform_internal, _ = super()._get_cuml_transform_func(
            dataset, eval_metric_info
        )

        from sklearn.neighbors import NearestNeighbors as SKNN

        n_neighbors = self.getK()

        def _construct_sknn() -> SKNN:
            nn_object = SKNN(algorithm="brute", n_neighbors=n_neighbors)
            return nn_object

        return _construct_sknn, _transform_internal, None

    def _concate_pdf_batches(self) -> bool:
        return False
