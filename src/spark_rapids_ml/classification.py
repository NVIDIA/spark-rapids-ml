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
import base64
import math
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cudf
import numpy as np
import pandas as pd
from pyspark import Row
from pyspark.ml.classification import _RandomForestClassifierParams
from pyspark.ml.param.shared import HasFeaturesCol
from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from spark_rapids_ml.core import (
    INIT_PARAMETERS_NAME,
    CumlInputType,
    CumlT,
    _CumlEstimatorSupervised,
    _CumlModelSupervised,
)
from spark_rapids_ml.params import HasFeaturesCols, _CumlClass


class RandomForestClassifierClass(_CumlClass):
    @classmethod
    def _cuml_cls(cls) -> List[type]:
        from cuml.ensemble.randomforest_common import BaseRandomForestModel

        return [BaseRandomForestModel]

    @classmethod
    def _param_excludes(cls) -> List[str]:
        return ["handle", "output_type", "n_streams"]

    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        return {
            "maxBins": "n_bins",
            "maxDepth": "max_depth",
            "numTrees": "n_estimators",
            "impurity": "split_criterion",
            "featureSubsetStrategy": "max_features",
            "bootstrap": "bootstrap",
            "seed": "random_state",
            "minInstancesPerNode": "min_samples_leaf",
        }

    @classmethod
    def _param_value_mapping(cls) -> Dict[str, Dict[str, Union[str, None]]]:
        return {
            "max_features": {
                "onethird": "0.3333",
                "all": "1.0",
                "auto": "auto",
                "sqrt": "sqrt",
                "log2": "log2",
            },
        }


class RandomForestClassifier(
    RandomForestClassifierClass,
    _CumlEstimatorSupervised,
    _RandomForestClassifierParams,
    HasFeaturesCol,
    HasFeaturesCols,
):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self.set_params(**kwargs)

    def getFeaturesCol(self) -> Union[str, List[str]]:  # type:ignore
        """
        Gets the value of :py:attr:`featuresCol` or :py:attr:`featuresCols`
        """
        if self.isDefined(self.featuresCols):
            return self.getFeaturesCols()
        elif self.isDefined(self.featuresCol):
            return self.getOrDefault("featuresCol")
        else:
            raise RuntimeError("featuresCol is not set")

    def setFeaturesCol(self, value: Union[str, List[str]]) -> "RandomForestClassifier":
        """
        Sets the value of :py:attr:`featuresCol` or :py:attr:`featureCols`.
        """
        if isinstance(value, str):
            self.set_params(featuresCol=value)
        else:
            self.set_params(featuresCols=value)
        return self

    def setFeaturesCols(self, value: List[str]) -> "RandomForestClassifier":
        """
        Sets the value of :py:attr:`featuresCols`.
        """
        return self.set_params(featuresCols=value)

    def setLabelCol(self, value: str) -> "RandomForestClassifier":
        """
        Sets the value of :py:attr:`labelCol`.
        """
        self._set(labelCol=value)  # type: ignore
        return self

    def setNumTrees(self, value: int) -> "RandomForestClassifier":
        """
        Sets the value of :py:attr:`numTrees`.
        """
        return self._set(numTrees=value)

    def _estimators_per_worker(self) -> List[int]:
        """Calculate the number of trees each task should train according to n_estimators"""
        n_estimators = self.cuml_params["n_estimators"]
        n_workers = self.getNumWorkers()
        if n_estimators < n_workers:
            raise ValueError("n_estimators cannot be lower than number of spark tasks.")

        n_est_per_worker = math.floor(n_estimators / n_workers)
        n_estimators_per_worker = [n_est_per_worker for i in range(n_workers)]
        remaining_est = n_estimators - (n_est_per_worker * n_workers)
        for i in range(remaining_est):
            n_estimators_per_worker[i] = n_estimators_per_worker[i] + 1
        return n_estimators_per_worker

    def _create_pyspark_model(self, result: Row) -> "RandomForestClassificationModel":
        return RandomForestClassificationModel.from_row(result)

    def _get_cuml_fit_func(
        self, dataset: DataFrame
    ) -> Callable[[CumlInputType, Dict[str, Any]], Dict[str, Any],]:
        n_estimators_per_worker = self._estimators_per_worker()

        def _rf_fit(
            dfs: CumlInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:

            from pyspark import BarrierTaskContext

            context = BarrierTaskContext.get()
            part_id = context.partitionId()

            from cuml import RandomForestClassifier

            rf_params = params[INIT_PARAMETERS_NAME]
            rf_params.pop("n_estimators")
            # Force n_streams=1 to avoid exception of running random forest
            # on the node with multi-gpus
            rf = RandomForestClassifier(
                n_estimators=n_estimators_per_worker[part_id],
                output_type="cudf",
                n_streams=1,
                **rf_params,
            )

            X_list = [item[0] for item in dfs]
            y_list = [item[1] for item in dfs]
            if isinstance(X_list[0], pd.DataFrame):
                X = pd.concat(X_list)
                y = pd.concat(y_list)
            else:
                # should be list of np.ndarrays here
                X = np.concatenate(X_list)
                y = np.concatenate(y_list)  # type: ignore

            # Fit a random forest classifier model on the dataset (X, y)
            rf.fit(X, y)

            # serialized_model is Dictionary type
            serialized_model = rf._get_serialized_model()
            pickled_model = pickle.dumps(serialized_model)
            msg = base64.b64encode(pickled_model).decode("utf-8")
            messages = context.allGather(msg)

            # concatenate the random forest in the worker0
            if part_id == 0:
                mod_bytes = [pickle.loads(base64.b64decode(i)) for i in messages]
                all_tl_mod_handles = [rf._tl_handle_from_bytes(i) for i in mod_bytes]
                rf._concatenate_treelite_handle(all_tl_mod_handles)

                from cuml.fil.fil import TreeliteModel

                for tl_handle in all_tl_mod_handles:
                    TreeliteModel.free_treelite_model(tl_handle)

                final_model_bytes = pickle.dumps(rf._get_serialized_model())
                final_model = base64.b64encode(final_model_bytes).decode("utf-8")
                return {
                    "treelite_model": [final_model],
                    "dtype": rf.dtype.name,
                    "n_cols": rf.n_cols,
                }
            else:
                return {}

        return _rf_fit

    def _out_schema(self) -> Union[StructType, str]:
        return StructType(
            [
                StructField("treelite_model", StringType(), False),
                StructField("n_cols", IntegerType(), False),
                StructField("dtype", StringType(), False),
            ]
        )

    def _enable_nccl(self) -> bool:
        return False


class RandomForestClassificationModel(
    RandomForestClassifierClass,
    _CumlModelSupervised,
    _RandomForestClassifierParams,
    HasFeaturesCol,
    HasFeaturesCols,
):
    def __init__(
        self,
        n_cols: int,
        dtype: str,
        treelite_model: str,
    ):
        super().__init__(dtype=dtype, n_cols=n_cols, treelite_model=treelite_model)
        self.treelite_model = treelite_model

    def getFeaturesCol(self) -> Union[str, List[str]]:  # type:ignore
        """
        Gets the value of :py:attr:`featuresCol` or :py:attr:`featuresCols`
        """
        if self.isDefined(self.featuresCols):
            return self.getFeaturesCols()
        elif self.isDefined(self.featuresCol):
            return self.getOrDefault("featuresCol")
        else:
            raise RuntimeError("featuresCol is not set")

    def setFeaturesCol(
        self, value: Union[str, List[str]]
    ) -> "RandomForestClassificationModel":
        """
        Sets the value of :py:attr:`featuresCol` or :py:attr:`featureCols`.
        """
        if isinstance(value, str):
            self.set_params(featuresCol=value)
        else:
            self.set_params(featuresCols=value)
        return self

    def setFeaturesCols(self, value: List[str]) -> "RandomForestClassificationModel":
        """
        Sets the value of :py:attr:`featuresCols`.
        """
        return self.set_params(featuresCols=value)

    def _get_cuml_transform_func(
        self, dataset: DataFrame
    ) -> Tuple[
        Callable[..., CumlT],
        Callable[[CumlT, Union[cudf.DataFrame, np.ndarray]], pd.DataFrame],
    ]:
        treelite_model = self.treelite_model

        def _construct_rf() -> CumlT:
            model = pickle.loads(base64.b64decode(treelite_model))
            from cuml import RandomForestClassifier as cuRf

            rf = cuRf()
            rf._concatenate_treelite_handle([rf._tl_handle_from_bytes(model)])

            return rf

        def _predict(rf: CumlT, pdf: Union[cudf.DataFrame, np.ndarray]) -> pd.Series:
            rf.update_labels = False
            ret = rf.predict(pdf)
            return pd.Series(ret)

        return _construct_rf, _predict
