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
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vector, Vectors, VectorUDT, _convert_to_vector
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol
from pyspark.ml.util import MLReader, MLWriter
from pyspark.sql import Column, DataFrame
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    FloatType,
    LongType,
    IntegerType,
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
    _CumlCaller,
    _CumlEstimator,
    _CumlModel,
    _CumlModelWithPredictionCol,
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
            "sample_fraction": 1.0,
        }


class _UMAPCumlParams(_CumlParams, HasFeaturesCol, HasFeaturesCols):
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


class UMAP(UMAPClass, _CumlEstimator, _UMAPCumlParams):

    """
    >>> from spark_rapids_ml.umap import UMAP
    >>> X, y = (1000, 10, centers=42, cluster_std=0.1,
                dtype=np.float32, random_state=10)
    >>> data_df = spark.createDataFrame(pd.DataFrame(X))
    >>> gpu_umap = UMAP(n_neighbors=15, n_components=2, ...) # Create UMAP() instance with params
    >>> gpu_model = umap.fit(data_df)
    >>> gpu_model.transform(data_df).show()
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.set_params(**kwargs)
        self.num_workers = 1

    def _create_pyspark_model(self, result: Row) -> "UMAPModel":
        return UMAPModel.from_row(result)

    def _fit(self, dataset: DataFrame) -> "UMAPModel":
        
        def _estimate_dataset_memory(dataset: DataFrame):
            num_rows = dataset.count()
            num_columns = len(dataset.columns)
            dtype = dataset.schema[0].dataType
            if isinstance(dtype, (DoubleType, LongType)):
                size_per_element = 8
            else:
                size_per_element = 4
            memory_usage_bytes = num_rows * num_columns * size_per_element
            return memory_usage_bytes
        
        def _get_gpu_memory_usage():
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming a single GPU is available
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_memory = info.used
            pynvml.nvmlShutdown()
            return used_memory
    
        if self.cuml_params["sample_fraction"] < 1.0:
            data_subset = dataset.sample(
                withReplacement=False,
                fraction=self.cuml_params["sample_fraction"],
                seed=self.cuml_params["random_state"],
            )
        else:
            data_subset = dataset

        data_subset_memory = _estimate_dataset_memory(data_subset)
        gpu_memory = _get_gpu_memory_usage()
        if data_subset_memory > gpu_memory:
            raise RuntimeError(
                f"Dataset size ({data_subset_memory}) is larger than available GPU memory ({gpu_memory})."
            )

        # force to single partition, single worker
        if data_subset.rdd.getNumPartitions() != 1:
            data_subset = data_subset.coalesce(1)

        # Operate on single node
        pipelined_rdd = self._call_cuml_fit_func(
            dataset=data_subset,
            partially_collect=False,
            paramMaps=None,
        )
        rows = pipelined_rdd.collect()
        model = self._create_pyspark_model(rows[0])
        model._num_workers = (
            self._num_workers
        )  # TODO: Change num_workers for outgoing model

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
                # alias params to cuml names
                **params[param_alias.cuml_init],
            )
            df_list = [x for (x, _, _) in dfs]
            if isinstance(df_list[0], pd.DataFrame):
                concated = pd.concat(df_list)
            else:
                # features are either cp or np arrays here
                concated = _concat_and_free(df_list, order=array_order)
            local_model = umap_object.fit(concated)
            return {
                "embedding_": [local_model.embedding_.tolist()],
            }

        return _cuml_fit

    def _require_nccl_ucx(self) -> Tuple[bool, bool]:
        return (False, False)

    def _out_schema(self) -> Union[StructType, str]:
        return StructType(
            [
                StructField(
                    "embedding_",
                    ArrayType(ArrayType(DoubleType()), False),
                    False,
                ),
            ]
        )


class UMAPModel(_CumlModel, UMAPClass, _UMAPCumlParams):
    def __init__(self, embedding_) -> None:
        super().__init__()
        self.embedding_ = embedding_

    @property
    def embedding(self) -> Vector:
        """
        Model coefficients.
        """
        # TBD: for large enough dimension, SparseVector is returned. Need to find out how to match
        return Vectors.dense(self.embedding_)

    def _get_cuml_transform_func(
        self, dataset: DataFrame, category: str = transform_evaluate.transform
    ) -> Tuple[_ConstructFunc, _TransformFunc, Optional[_EvaluateFunc],]:
        driver_embedding = self.embedding

        def _cuml_transform(
            dfs: FitInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            from cuml.manifold import UMAP as CumlUMAP

            # set umap object to internal fitted model
            internal_model = CumlUMAP(
                **params[param_alias.cuml_init],
            )
            internal_model.embedding_ = driver_embedding

            return internal_model.transform(dfs)

        return _cuml_transform

    def _require_nccl_ucx(self) -> Tuple[bool, bool]:
        return (False, False)

    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        """
        The output schema of the model, which will be used to
        construct the returning pandas dataframe
        """
        ret_schema = "array<array<double>>"
        return ret_schema
