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
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import cudf
import numpy as np
import pandas as pd
from pyspark.ml.linalg import Vector
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol
from pyspark.ml.tree import _DecisionTreeModel
from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from spark_rapids_ml.core import (
    CumlInputType,
    CumlT,
    _CumlEstimatorSupervised,
    _CumlModelSupervised,
    param_alias,
)
from spark_rapids_ml.params import HasFeaturesCols, P, _CumlClass, _CumlParams
from spark_rapids_ml.utils import _concat_and_free


class _RandomForestClass(_CumlClass):
    @classmethod
    def _cuml_cls(cls) -> List[type]:
        from cuml.ensemble.randomforest_common import BaseRandomForestModel

        return [BaseRandomForestModel]

    @classmethod
    def _param_excludes(cls) -> List[str]:
        return [
            "handle",
            "output_type",
            "accuracy_metric",
            "dtype",
            "criterion",
            "min_weight_fraction_leaf",
            "max_leaf_nodes",
            "min_impurity_split",
            "oob_score",
            "n_jobs",
            "warm_start",
            "class_weight",
        ]

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
            "minInfoGain": "",
            "maxMemoryInMB": "",
            "cacheNodeIds": "",
            "checkpointInterval": "",
            "subsamplingRate": "",
            "minWeightFractionPerNode": "",
            "weightCol": None,
            "leafCol": None,
        }

    @classmethod
    def _param_value_mapping(cls) -> Dict[str, Dict[str, Union[str, None]]]:
        return {
            "max_features": {
                "onethird": str(1 / 3.0),
                "all": "1.0",
                "auto": "auto",
                "sqrt": "sqrt",
                "log2": "log2",
            },
        }


class _RandomForestCumlParams(
    _CumlParams,
    HasFeaturesCol,
    HasFeaturesCols,
    HasLabelCol,
):
    def __init__(self) -> None:
        super().__init__()
        # restrict default seed to max value of 32-bit signed integer for CuML
        self._setDefault(seed=hash(type(self).__name__) & 0x07FFFFFFF)

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

    def setFeaturesCol(self: P, value: Union[str, List[str]]) -> P:
        """
        Sets the value of :py:attr:`featuresCol` or :py:attr:`featureCols`.
        """
        if isinstance(value, str):
            self.set_params(featuresCol=value)
        else:
            self.set_params(featuresCols=value)
        return self

    def setFeaturesCols(self: P, value: List[str]) -> P:
        """
        Sets the value of :py:attr:`featuresCols`.
        """
        return self.set_params(featuresCols=value)

    def setLabelCol(self: P, value: str) -> P:
        """
        Sets the value of :py:attr:`labelCol`.
        """
        self._set(labelCol=value)  # type: ignore
        return self

    def setSeed(self: P, value: int) -> P:
        if value > 0x07FFFFFFF:
            raise ValueError("cuML seed value must be a 32-bit integer.")
        return self.set_params(seed=value)


class _RandomForestEstimator(
    _CumlEstimatorSupervised,
    _RandomForestCumlParams,
):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self.set_params(**kwargs)
        if "n_streams" not in kwargs:
            # cuML will throw exception when running on a node with multi-gpus when n_streams > 0
            self._set_cuml_value("n_streams", 1)

    @abstractmethod
    def _is_classification(self) -> bool:
        """Indicate if it is regression or classification estimator"""
        raise NotImplementedError()

    def _estimators_per_worker(self) -> List[int]:
        """Calculate the number of trees each task should train according to n_estimators"""
        n_estimators = self.cuml_params["n_estimators"]
        n_workers = self.num_workers
        if n_estimators < n_workers:
            raise ValueError("n_estimators cannot be lower than number of spark tasks.")

        n_est_per_worker = math.floor(n_estimators / n_workers)
        n_estimators_per_worker = [n_est_per_worker for i in range(n_workers)]
        remaining_est = n_estimators - (n_est_per_worker * n_workers)
        for i in range(remaining_est):
            n_estimators_per_worker[i] = n_estimators_per_worker[i] + 1
        return n_estimators_per_worker

    def _get_cuml_fit_func(
        self, dataset: DataFrame
    ) -> Callable[[CumlInputType, Dict[str, Any]], Dict[str, Any],]:
        n_estimators_per_worker = self._estimators_per_worker()

        is_classification = self._is_classification()

        total_trees = self.cuml_params["n_estimators"]

        def _rf_fit(
            dfs: CumlInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            from pyspark import BarrierTaskContext

            context = BarrierTaskContext.get()
            part_id = context.partitionId()

            rf_params = params[param_alias.cuml_init]
            rf_params.pop("n_estimators")

            if rf_params["max_features"] == "auto":
                if total_trees == 1:
                    rf_params["max_features"] = 1.0
                else:
                    rf_params["max_features"] = (
                        "sqrt" if is_classification else (1 / 3.0)
                    )

            if is_classification:
                from cuml import RandomForestClassifier as cuRf
            else:
                from cuml import RandomForestRegressor as cuRf

            rf = cuRf(
                n_estimators=n_estimators_per_worker[part_id],
                output_type="cudf",
                **rf_params,
            )

            X_list = [item[0] for item in dfs]
            y_list = [item[1] for item in dfs]
            if isinstance(X_list[0], pd.DataFrame):
                X = pd.concat(X_list)
                y = pd.concat(y_list)
            else:
                # should be list of np.ndarrays here
                X = _concat_and_free(cast(List[np.ndarray], X_list))
                y = _concat_and_free(cast(List[np.ndarray], y_list))

            # Fit a random forest model on the dataset (X, y)
            rf.fit(X, y, convert_dtype=False)

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
                result = {
                    "treelite_model": [final_model],
                    "dtype": rf.dtype.name,
                    "n_cols": rf.n_cols,
                }
                if is_classification:
                    result["num_classes"] = rf.num_classes
                return result
            else:
                return {}

        return _rf_fit

    def _out_schema(self) -> Union[StructType, str]:
        fields = [
            StructField("treelite_model", StringType(), False),
            StructField("n_cols", IntegerType(), False),
            StructField("dtype", StringType(), False),
        ]
        if self._is_classification():
            fields.append(StructField("num_classes", IntegerType(), False))
        return StructType(fields)

    def _require_nccl_ucx(self) -> Tuple[bool, bool]:
        return False, False


class _RandomForestModel(
    _CumlModelSupervised,
    _RandomForestCumlParams,
):
    def __init__(
        self,
        n_cols: int,
        dtype: str,
        treelite_model: str,
        num_classes: int = -1,  # only for classification
    ):
        if self._is_classification():
            super().__init__(
                dtype=dtype,
                n_cols=n_cols,
                treelite_model=treelite_model,
                num_classes=num_classes,
            )
        else:
            super().__init__(dtype=dtype, n_cols=n_cols, treelite_model=treelite_model)

        self.treelite_model = treelite_model

    @property
    def featureImportances(self) -> Vector:
        """Estimate the importance of each feature."""
        raise NotImplementedError

    @property
    def getNumTrees(self) -> int:
        """Number of trees in ensemble."""
        return self.getOrDefault("numTrees")

    @property
    def toDebugString(self) -> str:
        """Full description of model."""
        raise NotImplementedError

    @property
    def totalNumNodes(self) -> int:
        """Total number of nodes, summed over all trees in the ensemble."""
        raise NotImplementedError

    @property
    def trees(self) -> List[_DecisionTreeModel]:
        """Trees in this ensemble. Warning: These have null parent Estimators."""
        raise NotImplementedError

    @property
    def treeWeights(self) -> List[float]:
        """Return the weights for each tree."""
        raise NotImplementedError

    def predict(self, value: Vector) -> float:
        """
        (Not supported) Predict label for the given features.
        """
        raise NotImplementedError

    def predictLeaf(self, value: Vector) -> float:
        """
        (Not supported) Predict the indices of the leaves corresponding to the feature vector.
        """
        raise NotImplementedError

    @abstractmethod
    def _is_classification(self) -> bool:
        """Indicate if it is regression or classification model"""
        raise NotImplementedError()

    def _get_cuml_transform_func(
        self, dataset: DataFrame
    ) -> Tuple[
        Callable[..., CumlT],
        Callable[[CumlT, Union[cudf.DataFrame, np.ndarray]], pd.DataFrame],
    ]:
        treelite_model = self.treelite_model

        is_classification = self._is_classification()

        def _construct_rf() -> CumlT:
            model = pickle.loads(base64.b64decode(treelite_model))

            if is_classification:
                from cuml import RandomForestClassifier as cuRf
            else:
                from cuml import RandomForestRegressor as cuRf

            rf = cuRf()
            rf._concatenate_treelite_handle([rf._tl_handle_from_bytes(model)])

            return rf

        def _predict(rf: CumlT, pdf: Union[cudf.DataFrame, np.ndarray]) -> pd.Series:
            rf.update_labels = False
            ret = rf.predict(pdf)
            return pd.Series(ret)

        # TBD: figure out why RF algo's warns regardless of what np array order is set
        return _construct_rf, _predict
