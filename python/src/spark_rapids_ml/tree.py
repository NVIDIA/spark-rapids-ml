#
# Copyright (c) 2025, NVIDIA CORPORATION.
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
import json
import math
import pickle
from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from pyspark.ml.classification import (
    DecisionTreeClassificationModel,
)
from pyspark.ml.classification import (
    RandomForestClassificationModel as SparkRandomForestClassificationModel,
)
from pyspark.ml.linalg import Vector
from pyspark.ml.param.shared import (
    HasFeaturesCol,
    HasLabelCol,
    Param,
    Params,
    TypeConverters,
)
from pyspark.ml.regression import (
    DecisionTreeRegressionModel,
)
from pyspark.ml.regression import (
    RandomForestRegressionModel as SparkRandomForestRegressionModel,
)
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from .core import (
    CumlT,
    FitInputType,
    TransformInputType,
    _ConstructFunc,
    _CumlEstimatorSupervised,
    _CumlModelWithPredictionCol,
    _EvaluateFunc,
    _TransformFunc,
    param_alias,
)
from .metrics import EvalMetricInfo
from .params import HasFeaturesCols, P, _CumlClass, _CumlParams
from .utils import (
    _concat_and_free,
    _get_spark_session,
    _str_or_numerical,
    java_uid,
    translate_trees,
)

if TYPE_CHECKING:
    import cupy as cp


class _RandomForestClass(_CumlClass):
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
    def _param_value_mapping(
        cls,
    ) -> Dict[str, Callable[[Any], Union[None, str, float, int]]]:
        def _tree_mapping(feature_subset: str) -> Union[None, str, float, int]:
            _maybe_numerical = _str_or_numerical(feature_subset)
            if isinstance(_maybe_numerical, int) or isinstance(_maybe_numerical, float):
                _numerical = _maybe_numerical
                return _numerical
            else:
                _str = _maybe_numerical
                _tree_string_mapping: Dict[str, Union[None, str, float, int]] = {
                    "onethird": 1 / 3.0,
                    "all": 1.0,
                    "auto": "auto",
                    "sqrt": "sqrt",
                    "log2": "log2",
                }
                return _tree_string_mapping.get(_str, None)

        return {
            "max_features": _tree_mapping,
        }

    def _get_cuml_params_default(self) -> Dict[str, Any]:
        return {
            "n_streams": 4,
            "n_estimators": 100,
            "max_depth": 16,
            "max_features": "sqrt",  # for classification, should be 1.0 for regressor, cuml is a little broken here
            "n_bins": 128,
            "bootstrap": True,
            "verbose": False,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_samples": 1.0,
            "max_leaves": -1,
            "min_impurity_decrease": 0.0,
            "random_state": None,
            "max_batch_size": 4096,
        }


class _RandomForestCumlParams(
    _CumlParams,
    HasFeaturesCol,
    HasFeaturesCols,
    HasLabelCol,
):

    n_streams = Param(
        Params._dummy(),
        "n_streams",
        "The n_streams parameter to use for cuml.",
        typeConverter=TypeConverters.toInt,
    )

    min_samples_split = Param(
        Params._dummy(),
        "min_samples_split",
        "The min_sample_split parameter to use for cuml.",
        typeConverter=TypeConverters.toInt,
    )

    max_samples = Param(
        Params._dummy(),
        "max_samples",
        "The max_samples parameter to use for cuml.",
        typeConverter=TypeConverters.toFloat,
    )

    max_leaves = Param(
        Params._dummy(),
        "max_leaves",
        "The max_leaves parameter to use for cuml.",
        typeConverter=TypeConverters.toInt,
    )

    min_impurity_decrease = Param(
        Params._dummy(),
        "min_impurity_decrease",
        "The min_impurity_decrease parameter to use for cuml.",
        typeConverter=TypeConverters.toFloat,
    )

    max_batch_size = Param(
        Params._dummy(),
        "max_batch_size",
        "The max_batch_size parameter to use for cuml.",
        typeConverter=TypeConverters.toInt,
    )

    def __init__(self) -> None:
        super().__init__()
        self._setDefault(n_streams=4)
        self._setDefault(min_samples_split=2)
        self._setDefault(max_samples=1.0)
        self._setDefault(max_leaves=-1)
        self._setDefault(min_impurity_decrease=0.0)
        self._setDefault(max_batch_size=4096)
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
            self._set_params(featuresCol=value)
        else:
            self._set_params(featuresCols=value)
        return self

    def setFeaturesCols(self: P, value: List[str]) -> P:
        """
        Sets the value of :py:attr:`featuresCols`.
        """
        return self._set_params(featuresCols=value)

    def setLabelCol(self: P, value: str) -> P:
        """
        Sets the value of :py:attr:`labelCol`.
        """
        self._set(labelCol=value)  # type: ignore
        return self

    def setPredictionCol(self: P, value: str) -> P:
        """
        Sets the value of :py:attr:`predictionCol`.
        """
        self._set(predictionCol=value)  # type: ignore
        return self


class _RandomForestEstimatorParams(_RandomForestCumlParams):
    def __init__(self) -> None:
        super().__init__()

    # include setters used only in estimator classes (classifier and regressor) here
    def setBootstrap(self: P, value: bool) -> P:
        """
        Sets the value of :py:attr:`bootstrap`.
        """
        return self._set_params(bootstrap=value)

    def setFeatureSubsetStrategy(self: P, value: str) -> P:
        """
        Sets the value of :py:attr:`featureSubsetStrategy`.
        """
        return self._set_params(featureSubsetStrategy=value)

    def setImpurity(self: P, value: str) -> P:
        """
        Sets the value of :py:attr:`impurity`.
        """
        return self._set_params(impurity=value)  # type: ignore

    def setMaxBins(self: P, value: int) -> P:
        """
        Sets the value of :py:attr:`maxBins`.
        """
        return self._set_params(maxBins=value)

    def setMaxDepth(self: P, value: int) -> P:
        """
        Sets the value of :py:attr:`maxDepth`.
        """
        return self._set_params(maxDepth=value)

    def setMinInstancesPerNode(self: P, value: int) -> P:
        """
        Sets the value of :py:attr:`minInstancesPerNode`.
        """
        return self._set_params(minInstancesPerNode=value)

    def setNumTrees(self: P, value: int) -> P:
        """
        Sets the value of :py:attr:`numTrees`.
        """
        return self._set_params(numTrees=value)

    def setSeed(self: P, value: int) -> P:
        """
        Sets the value of :py:attr:`seed`.
        """
        if value > 0x07FFFFFFF:
            raise ValueError("cuML seed value must be a 32-bit integer.")
        return self._set_params(seed=value)


class _RandomForestEstimator(
    _CumlEstimatorSupervised,
    _RandomForestEstimatorParams,
):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self._set_params(**kwargs)
        if "n_streams" not in kwargs:
            # cuML will throw exception when running on a node with multi-gpus when n_streams > 0
            self._set_cuml_value("n_streams", 1)

    @abstractmethod
    def _is_classification(self) -> bool:
        """Indicate if it is regression or classification estimator"""
        raise NotImplementedError()

    def _estimators_per_worker(self, n_estimators: int) -> List[int]:
        """Calculate the number of trees each task should train according to n_estimators"""
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
        self,
        dataset: DataFrame,
        extra_params: Optional[List[Dict[str, Any]]] = None,
    ) -> Callable[
        [FitInputType, Dict[str, Any]],
        Dict[str, Any],
    ]:
        # Each element of n_estimators_of_all_params is a list value which
        # is composed of n_estimators per worker.
        n_estimators_of_all_params: List[List[int]] = []
        total_trees: List[int] = []

        all_params = [{}] if extra_params is None else extra_params

        for params in all_params:
            num_trees = (
                self.cuml_params["n_estimators"]
                if "n_estimators" not in params
                else params["n_estimators"]
            )
            n_estimators_of_all_params.append(self._estimators_per_worker(num_trees))
            total_trees.append(num_trees)

        is_classification = self._is_classification()

        def _rf_fit(
            dfs: FitInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            X_list = [item[0] for item in dfs]
            y_list = [item[1] for item in dfs]
            if isinstance(X_list[0], pd.DataFrame):
                X = pd.concat(X_list)
                y = pd.concat(y_list)
            else:
                # features are either cp or np arrays here
                X = _concat_and_free(X_list)
                y = _concat_and_free(y_list)

            if is_classification:
                from cuml import RandomForestClassifier as cuRf
            else:
                from cuml import RandomForestRegressor as cuRf

            from pyspark import BarrierTaskContext

            context = BarrierTaskContext.get()
            part_id = context.partitionId()

            def _single_fit(rf: cuRf) -> Dict[str, Any]:
                # Fit a random forest model on the dataset (X, y)
                rf.fit(X, y, convert_dtype=False)

                # serialized_model is Dictionary type
                serialized_model = rf._get_serialized_model()
                pickled_model = pickle.dumps(serialized_model)
                msg = base64.b64encode(pickled_model).decode("utf-8")
                trees = rf.get_json()
                data = {"model_bytes": msg, "model_json": trees}
                messages = context.allGather(json.dumps(data))

                # concatenate the random forest in the worker0
                if part_id == 0:
                    mod_bytes = []
                    mod_jsons = []
                    for msg in messages:
                        data = json.loads(msg)
                        mod_bytes.append(
                            pickle.loads(base64.b64decode(data["model_bytes"]))
                        )
                        mod_jsons.append(data["model_json"])

                    all_tl_mod_handles = [
                        rf._tl_handle_from_bytes(i) for i in mod_bytes
                    ]

                    # tree concatenation raises a non-user friendly error if some workers didn't get all label values
                    try:
                        rf._concatenate_treelite_handle(all_tl_mod_handles)
                    except RuntimeError as err:
                        import traceback

                        exc_str = traceback.format_exc()
                        if "different num_class than the first model object" in exc_str:
                            raise RuntimeError(
                                "Some GPU workers did not receive all label values.  Rerun with fewer workers or shuffle input data."
                            )
                        else:
                            raise err

                    from cuml.fil.fil import TreeliteModel

                    for tl_handle in all_tl_mod_handles:
                        TreeliteModel.free_treelite_model(tl_handle)

                    final_model_bytes = pickle.dumps(rf._get_serialized_model())
                    final_model = base64.b64encode(final_model_bytes).decode("utf-8")
                    result = {
                        "treelite_model": final_model,
                        "dtype": rf.dtype.name,
                        "n_cols": rf.n_cols,
                        "model_json": mod_jsons,
                    }

                    if is_classification:
                        result["num_classes"] = rf.num_classes

                    return result
                else:
                    return {}

            rf_params = params[param_alias.cuml_init]
            fit_multiple_params = params[param_alias.fit_multiple_params]

            if len(fit_multiple_params) == 0:
                fit_multiple_params.append({})

            models = []
            for i in range(len(fit_multiple_params)):
                tmp_rf_params = rf_params.copy()
                tmp_rf_params.update(fit_multiple_params[i])
                tmp_rf_params.pop("n_estimators")

                if tmp_rf_params["max_features"] == "auto":
                    if total_trees[i] == 1:
                        tmp_rf_params["max_features"] = 1.0
                    else:
                        tmp_rf_params["max_features"] = (
                            "sqrt" if is_classification else (1 / 3.0)
                        )
                rf = cuRf(
                    n_estimators=n_estimators_of_all_params[i][part_id],
                    output_type="cudf",
                    **tmp_rf_params,
                )
                models.append(_single_fit(rf))
                del rf

            models_dict = {}

            if part_id == 0:
                for k in models[0].keys():
                    models_dict[k] = [m[k] for m in models]
            return models_dict

        return _rf_fit

    def _out_schema(self) -> Union[StructType, str]:
        fields = [
            StructField("treelite_model", StringType(), False),
            StructField("n_cols", IntegerType(), False),
            StructField("dtype", StringType(), False),
            StructField("model_json", ArrayType(StringType()), False),
        ]
        if self._is_classification():
            fields.append(StructField("num_classes", IntegerType(), False))

        return StructType(fields)

    def _require_nccl_ucx(self) -> Tuple[bool, bool]:
        return False, False

    def _enable_fit_multiple_in_single_pass(self) -> bool:
        return True


class _RandomForestModel(
    _CumlModelWithPredictionCol,
    _RandomForestCumlParams,
):
    def __init__(
        self,
        n_cols: int,
        dtype: str,
        treelite_model: Union[str, List[str]],
        model_json: Union[List[str], List[List[str]]] = [],  # type: ignore
        num_classes: int = -1,  # only for classification
    ):
        if self._is_classification():
            super().__init__(
                dtype=dtype,
                n_cols=n_cols,
                treelite_model=treelite_model,
                num_classes=num_classes,
                model_json=model_json,
            )
        else:
            super().__init__(
                dtype=dtype,
                n_cols=n_cols,
                treelite_model=treelite_model,
                model_json=model_json,
            )
        self._num_classes = num_classes
        self._model_json = model_json
        self._treelite_model = treelite_model

    def cpu(
        self,
    ) -> Union[SparkRandomForestRegressionModel, SparkRandomForestClassificationModel]:
        raise NotImplementedError()

    @property
    def featureImportances(self) -> Vector:
        """Estimate the importance of each feature."""
        return self.cpu().featureImportances

    @property
    def getNumTrees(self) -> int:
        """Number of trees in ensemble."""
        return self.getOrDefault("numTrees")

    @property
    def toDebugString(self) -> str:
        """Full description of model."""
        return self.cpu().toDebugString

    @property
    def totalNumNodes(self) -> int:
        """Total number of nodes, summed over all trees in the ensemble."""
        return self.cpu().totalNumNodes

    @property
    def trees(
        self,
    ) -> Union[
        List[DecisionTreeRegressionModel], List[DecisionTreeClassificationModel]
    ]:
        """Trees in this ensemble. Warning: These have null parent Estimators."""
        return self.cpu().trees

    @property
    def treeWeights(self) -> List[float]:
        """Return the weights for each tree."""
        return self.cpu().treeWeights

    def predict(self, value: Vector) -> float:
        """
        Predict label for the given features.
        """
        return self.cpu().predict(value)

    def predictLeaf(self, value: Vector) -> float:
        """
        Predict the indices of the leaves corresponding to the feature vector.
        """
        return self.cpu().predictLeaf(value)

    @abstractmethod
    def _is_classification(self) -> bool:
        """Indicate if it is regression or classification model"""
        raise NotImplementedError()

    def _convert_to_java_trees(self, impurity: str) -> Tuple[Any, List[Any]]:
        """Convert cuml trees to Java decision tree model"""
        sc = _get_spark_session().sparkContext
        assert sc._jvm is not None
        assert sc._gateway is not None

        # This function shouldn't be called for the multiple models scenario.
        model_json = cast(List[str], self._model_json)

        # Convert cuml trees to Spark trees
        trees = [
            translate_trees(sc, impurity, trees)
            for trees_json in model_json
            for trees in json.loads(trees_json)
        ]

        if self._is_classification():
            uid = java_uid(sc, "rfc")
            java_decision_tree_model_class = (
                sc._jvm.org.apache.spark.ml.classification.DecisionTreeClassificationModel
            )
            # Wrap the trees into Spark DecisionTreeClassificationModel
            decision_trees = [
                java_decision_tree_model_class(
                    uid, tree, self.numFeatures, self._num_classes
                )
                for tree in trees
            ]
        else:
            uid = java_uid(sc, "rfr")
            java_decision_tree_model_class = (
                sc._jvm.org.apache.spark.ml.regression.DecisionTreeRegressionModel
            )
            # Wrap the trees into Spark DecisionTreeClassificationModel
            decision_trees = [
                java_decision_tree_model_class(uid, tree, self.numFeatures)
                for tree in trees
            ]

        java_trees = sc._gateway.new_array(
            java_decision_tree_model_class, len(decision_trees)
        )
        for i in range(len(decision_trees)):
            java_trees[i] = decision_trees[i]

        return uid, java_trees

    def _get_cuml_transform_func(
        self, dataset: DataFrame, eval_metric_info: Optional[EvalMetricInfo] = None
    ) -> Tuple[
        _ConstructFunc,
        _TransformFunc,
        Optional[_EvaluateFunc],
    ]:
        treelite_model = self._treelite_model
        is_classification = self._is_classification()

        def _construct_rf() -> CumlT:
            if is_classification:
                from cuml import RandomForestClassifier as cuRf
            else:
                from cuml import RandomForestRegressor as cuRf

            rfs = []
            treelite_models = (
                treelite_model if isinstance(treelite_model, list) else [treelite_model]
            )
            for m in treelite_models:
                model = pickle.loads(base64.b64decode(m))
                rf = cuRf()
                rf._concatenate_treelite_handle([rf._tl_handle_from_bytes(model)])
                rfs.append(rf)

            return rfs

        if eval_metric_info:

            def _predict(rf: CumlT, pdf: TransformInputType) -> "cp.ndarray":
                rf.update_labels = False
                return rf.predict(pdf)

        else:

            def _predict(rf: CumlT, pdf: TransformInputType) -> pd.Series:
                rf.update_labels = False
                ret = rf.predict(pdf)
                return pd.Series(ret)

        # TBD: figure out why RF algo's warns regardless of what np array order is set
        return _construct_rf, _predict, None

    def _transform(self, dataset: DataFrame) -> DataFrame:
        df = super()._transform(dataset)
        return df.withColumn(
            self.getPredictionCol(), df[self.getPredictionCol()].cast("double")
        )

    @classmethod
    def _combine(
        cls: Type["_RandomForestModel"], models: List["_RandomForestModel"]  # type: ignore
    ) -> "_RandomForestModel":
        assert len(models) > 0 and all(isinstance(model, cls) for model in models)
        first_model = models[0]
        treelite_models = [model._treelite_model for model in models]
        model_jsons = [model._model_json for model in models]
        attrs = first_model._get_model_attributes()
        assert attrs is not None
        attrs["treelite_model"] = treelite_models
        attrs["model_json"] = model_jsons
        rf_model = cls(**attrs)
        first_model._copyValues(rf_model)
        first_model._copy_cuml_params(rf_model)
        return rf_model
