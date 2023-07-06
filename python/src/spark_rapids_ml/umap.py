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

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol
from pyspark.sql import Column, DataFrame
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    Row,
    StringType,
    StructField,
    StructType,
)

from spark_rapids_ml.core import FitInputType

from .core import (
    CumlT,
    FitInputType,
    _ConstructFunc,
    _CumlEstimatorSupervised,
    _CumlModel,
    _EvaluateFunc,
    _TransformFunc,
    alias,
    param_alias,
    transform_evaluate,
)
from .params import HasFeaturesCols, P, _CumlClass, _CumlParams
from .utils import _ArrayOrder, _concat_and_free


class UMAPClass(_CumlClass):
    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        return {}

    def _get_cuml_params_default(self) -> Dict[str, Any]:
        return {
            "n_neighbors": 15,
            "n_components": 2,
            "metric": "euclidean",
            "n_epochs": None,
            "learning_rate": 1.0,
            "init": "spectral",
            "min_dist": 0.1,
            "spread": 1.0,
            "set_op_mix_ratio": 1.0,
            "local_connectivity": 1.0,
            "repulsion_strength": 1.0,
            "negative_sample_rate": 5,
            "transform_queue_size": 4.0,
            "a": None,
            "b": None,
            "hash_input": False,
            "random_state": None,
            "callback": None,
            "verbose": False,
            "output_type": None,
            "handle": None,
        }


class _UMAPCumlParams(_CumlParams, HasFeaturesCol, HasFeaturesCols, HasLabelCol):
    def __init__(self) -> None:
        super().__init__()
        self._setDefault()

    def getFeaturesCol(self) -> Union[str, List[str]]:  # type: ignore
        """
        Gets the value of :py:attr:`featuresCol` or :py:attr:`featuresCols`
        """
        if self.isDefined(self.featuresCols):
            return self.getFeaturesCols()
        elif self.isDefined(self.featuresCol):
            return self.getOrDefault("featuresCol")
        else:
            raise RuntimeError("featuresCol is not set")

    def setFeaturesCol(self: P, value: Union[str, List[str]]) -> P:
        """
        Sets the value of :py:attr:`featuresCol` or :py:attr:`featuresCols`. Used when input vectors are stored in a single column.
        """
        if isinstance(value, str):
            self.set_params(featuresCol=value)
        else:
            self.set_params(featuresCols=value)
        return self

    def setFeaturesCols(self: P, value: List[str]) -> P:
        """
        Sets the value of :py:attr:`featuresCols`. Used when input vectors are stored as multiple feature columns.
        """
        return self.set_params(featuresCols=value)

    def setLabelCol(self: P, value: str) -> P:
        """
        Sets the value of :py:attr:`labelCol`.
        """
        return self.set_params(labelCol=value)


class UMAP(UMAPClass, _CumlEstimatorSupervised, _UMAPCumlParams):

    """
    >>> from spark_rapids_ml.umap import UMAP
    >>> X, _ = (1000, 10, centers=42, cluster_std=0.1,
                dtype=np.float32, random_state=10)
    >>> df = spark.createDataFrame(X, ["features"])
    >>> local_model = UMAP().setFeaturesCol("features")
    >>> distributed_model = umap.fit(df)
    >>> embeddings = distributed_model.transform(df)
    >>> embeddings.show()
    """

    def __init__(self, sample_fraction: float = 1.0, **kwargs: Any) -> None:
        super().__init__()
        self.set_params(**kwargs)
        self.sample_fraction = sample_fraction

    def _create_pyspark_model(self, result: Row) -> "UMAPModel":
        return UMAPModel.from_row(result)

    def _fit(self, dataset: DataFrame) -> "UMAPModel":
        if self.sample_fraction < 1.0:
            data_subset = dataset.sample(
                withReplacement=False,
                fraction=self.sample_fraction,
                seed=self.cuml_params["random_state"],
            )
        else:
            data_subset = dataset

        input_num_workers = self.num_workers
        # Force to single partition, single worker
        self._num_workers = 1
        if data_subset.rdd.getNumPartitions() != 1:
            data_subset = data_subset.coalesce(1)

        pipelined_rdd = self._call_cuml_fit_func(
            dataset=data_subset,
            partially_collect=False,
            paramMaps=None,
        )
        rows = pipelined_rdd.collect()
        model = self._create_pyspark_model(rows[0])
        model._num_workers = input_num_workers

        self._copyValues(model)
        self._copy_cuml_params(model)  # type: ignore

        return model

    def _fit_array_order(self) -> _ArrayOrder:
        return "C"

    def _get_cuml_fit_func(
        self,
        dataset: DataFrame,
        extra_params: Optional[List[Dict[str, Any]]] = None,
    ) -> Callable[[FitInputType, Dict[str, Any]], Dict[str, Any],]:
        cls = self.__class__
        array_order = self._fit_array_order()

        def _cuml_fit(
            dfs: FitInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            from cuml.manifold import UMAP as CumlUMAP

            umap_object = CumlUMAP(
                **params[param_alias.cuml_init],
            )

            df_list = [x for (x, _, _) in dfs]
            if isinstance(df_list[0], pd.DataFrame):
                concated = pd.concat(df_list)
            else:
                concated = _concat_and_free(df_list, order=array_order)

            if dfs[0][1] is not None:
                # If labels are provided, call supervised fit
                label_list = [x for (_, x, _) in dfs]
                if isinstance(label_list[0], pd.DataFrame):
                    labels = pd.concat(label_list)
                else:
                    labels = _concat_and_free(label_list, order=array_order)
                local_model = umap_object.fit(concated, labels)
            else:
                # Call unsupervised fit
                local_model = umap_object.fit(concated)

            dtype = str(local_model.embedding_.dtype)

            return {
                "embedding_": [local_model.embedding_.tolist()],
                "n_cols": len(local_model.embedding_[0]),
                "dtype": dtype,
            }

        return _cuml_fit

    def _require_nccl_ucx(self) -> Tuple[bool, bool]:
        return (False, False)

    def _out_schema(self) -> Union[StructType, str]:
        return StructType(
            [
                StructField(
                    "embedding_",
                    ArrayType(ArrayType(FloatType()), False),
                    False,
                ),
                StructField("n_cols", IntegerType(), False),
                StructField("dtype", StringType(), False),
            ]
        )

    def _pre_process_data(
        self, dataset: DataFrame
    ) -> Tuple[
        List[Column], Optional[List[str]], int, Union[Type[FloatType], Type[DoubleType]]
    ]:
        (
            select_cols,
            multi_col_names,
            dimension,
            feature_type,
        ) = super(
            _CumlEstimatorSupervised, self
        )._pre_process_data(dataset)

        if self.getLabelCol() in dataset.schema.names:
            select_cols.append(self._pre_process_label(dataset, feature_type))

        return select_cols, multi_col_names, dimension, feature_type


class UMAPModel(_CumlModel, UMAPClass, _UMAPCumlParams):
    def __init__(self, embedding_: List[List[float]], n_cols: int, dtype: str) -> None:
        super(UMAPModel, self).__init__(
            embedding_=embedding_, n_cols=n_cols, dtype=dtype
        )
        self.embedding_ = embedding_

    @property
    def embedding(self) -> List[List[float]]:
        return self.embedding_

    def _get_cuml_transform_func(
        self, dataset: DataFrame, category: str = transform_evaluate.transform
    ) -> Tuple[_ConstructFunc, _TransformFunc, Optional[_EvaluateFunc],]:
        raise NotImplementedError("TODO")

    def _require_nccl_ucx(self) -> Tuple[bool, bool]:
        return (False, False)

    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        raise NotImplementedError("TODO")
