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

from abc import ABCMeta
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import pytest
from pyspark import Row, SparkConf, TaskContext
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCols, HasOutputCols
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType

from spark_rapids_ml.core import (
    CumlT,
    FitInputType,
    _ConstructFunc,
    _CumlEstimator,
    _CumlModel,
    _EvaluateFunc,
    _TransformFunc,
    param_alias,
)
from spark_rapids_ml.metrics import EvalMetricInfo
from spark_rapids_ml.params import _CumlClass, _CumlParams
from spark_rapids_ml.utils import PartitionDescriptor

from .utils import assert_params, get_default_cuml_parameters


class CumlDummy(object):
    """
    A dummy class to mimic a cuml python class
    """

    def __init__(self, a: float = 10.0, b: int = 20, k: int = 30, x: float = 40.0) -> None:  # type: ignore
        super().__init__()
        self.a = a  # alpha
        self.b = b  # dropped
        self.k = k  # k
        self.x = x  # extra, keep w/ default


class SparkRapidsMLDummyClass(_CumlClass):
    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        return {
            "alpha": "a",  # direct map, different names
            "beta": None,  # unmapped, raise error if defined on Spark side
            "gamma": "",  # unmapped, ignore value from Spark side
            "k": "k",  # direct map, same name
        }

    def _get_cuml_params_default(self) -> Dict[str, Any]:
        return {"a": 10.0, "k": 30, "x": 40.0}


class _SparkRapidsMLDummyParams(_CumlParams):
    """
    Params for Spark Dummy class
    """

    alpha = Param(
        Params._dummy(),  # type: ignore
        "alpha",
        "alpha dummy param",
        TypeConverters.toFloat,
    )

    beta = Param(
        Params._dummy(),  # type: ignore
        "beta",
        "beta dummy param ",
        TypeConverters.toInt,
    )

    gamma = Param(
        Params._dummy(),  # type: ignore
        "gamma",
        "gamma dummy param ",
        TypeConverters.toString,
    )

    k = Param(
        Params._dummy(),  # type: ignore
        "k",
        "k dummy param ",
        TypeConverters.toInt,
    )

    def __init__(self, *args: Any):
        super(_SparkRapidsMLDummyParams, self).__init__(*args)
        self._setDefault(
            alpha=1.0,
            # beta=2,         # leave undefined to test mapping to None
            gamma="three",
            k=4,
        )


class SparkRapidsMLDummy(
    SparkRapidsMLDummyClass,
    _CumlEstimator,
    _SparkRapidsMLDummyParams,
    HasInputCols,
    HasOutputCols,
):
    """
    PySpark estimator of CumlDummy
    """

    def __init__(
        self,
        m: int = 0,
        n: int = 0,
        partition_num: int = 0,
        runtime_check: bool = True,
        **kwargs: Any,
    ) -> None:
        #

        super().__init__()
        self._set_params(**kwargs)
        self.m = m
        self.n = n
        self.partition_num = partition_num
        self.runtime_check = runtime_check

    """
    PySpark estimator of CumlDummy
    """

    def setInputCols(self, value: List[str]) -> "SparkRapidsMLDummy":
        return self._set(inputCols=value)

    def setOutputCols(self, value: List[str]) -> "SparkRapidsMLDummy":
        return self._set(outputCols=value)

    def setAlpha(self, value: int) -> "SparkRapidsMLDummy":
        return self._set_params(**{"alpha": value})

    def setBeta(self, value: int) -> "SparkRapidsMLDummy":
        raise ValueError("Not supported")

    def setGamma(self, value: float) -> "SparkRapidsMLDummy":
        return self._set_params(**{"gamma": value})

    def setK(self, value: str) -> "SparkRapidsMLDummy":
        return self._set_params(**{"k": value})

    def _get_cuml_fit_func(
        self,
        dataset: DataFrame,
        extra_params: Optional[List[Dict[str, Any]]] = None,
    ) -> Callable[
        [FitInputType, Dict[str, Any]],
        Dict[str, Any],
    ]:
        num_workers = self.num_workers
        partition_num = self.partition_num
        m = self.m
        n = self.n

        # if the common framework tries to pickle the whole class,
        # it will throw exception since dataset is not picklable.
        self.test_pickle_dataframe = dataset

        runtime_check = self.runtime_check

        def _cuml_fit(
            dfs: FitInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            context = TaskContext.get()
            assert context is not None
            assert param_alias.handle in params
            assert param_alias.part_sizes in params
            assert param_alias.num_cols in params

            pd = PartitionDescriptor.build(
                params[param_alias.part_sizes], params[param_alias.num_cols]
            )

            assert param_alias.cuml_init in params
            init_params = params[param_alias.cuml_init]
            dummy = CumlDummy(**init_params)

            if runtime_check:
                assert pd.rank == context.partitionId()
                assert len(pd.parts_rank_size) == partition_num
                assert pd.m == m
                assert pd.n == n

                assert init_params == {"a": 100, "k": 4, "x": 40.0}
                assert dummy.a == 100
                assert dummy.b == 20
                assert dummy.k == 4
                assert dummy.x == 40.0

            import time

            # sleep for 1 sec to bypass https://issues.apache.org/jira/browse/SPARK-40932
            time.sleep(1)

            return {
                "dtype": np.dtype(np.float32).name,
                "n_cols": n,
                "model_attribute_a": [1024],
                "model_attribute_b": "hello dummy",
            }

        return _cuml_fit

    def _out_schema(self) -> Union[StructType, str]:
        return (
            "dtype string, n_cols int, model_attribute_a int, model_attribute_b string"
        )

    def _create_pyspark_model(self, result: Row) -> "SparkRapidsMLDummyModel":
        assert result.dtype == np.dtype(np.float32).name
        assert result.n_cols == self.n
        assert result.model_attribute_a == 1024
        assert result.model_attribute_b == "hello dummy"
        return SparkRapidsMLDummyModel._from_row(result)

    def _pyspark_class(self) -> Optional[ABCMeta]:
        return None


class SparkRapidsMLDummyModel(
    SparkRapidsMLDummyClass,
    _CumlModel,
    _SparkRapidsMLDummyParams,
    HasInputCols,
    HasOutputCols,
):
    """
    PySpark model of CumlDummy
    """

    def __init__(
        self,
        dtype: str,
        n_cols: int,
        model_attribute_a: int,
        model_attribute_b: str,
        not_used: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dtype=dtype,
            n_cols=n_cols,
            model_attribute_a=model_attribute_a,
            model_attribute_b=model_attribute_b,
        )  # type: ignore
        self.model_attribute_a = model_attribute_a
        self.model_attribute_b = model_attribute_b
        self._set_params(**kwargs)

    def setInputCols(self, value: List[str]) -> "SparkRapidsMLDummyModel":
        return self._set(inputCols=value)

    def setOutputCols(self, value: List[str]) -> "SparkRapidsMLDummyModel":
        return self._set(outputCols=value)

    def _get_cuml_transform_func(
        self, dataset: DataFrame, eval_metric_info: Optional[EvalMetricInfo] = None
    ) -> Tuple[
        _ConstructFunc,
        _TransformFunc,
        Optional[_EvaluateFunc],
    ]:
        model_attribute_a = self.model_attribute_a

        # if the common framework tries to pickle the whole class,
        # it will throw exception since dataset is not picklable.
        self.test_pickle_dataframe = dataset
        output_cols = self.getInputCols()

        def _construct_dummy() -> CumlT:
            dummy = CumlDummy(a=101, b=102, k=103)
            return dummy

        def _dummy_transform(
            dummy: CumlT, df: Union[pd.DataFrame, np.ndarray]
        ) -> pd.DataFrame:
            assert dummy.a == 101
            assert dummy.b == 102
            assert dummy.k == 103

            assert model_attribute_a == 1024
            if isinstance(df, pd.DataFrame):
                col_mapper = dict(zip(df.columns, output_cols))
                return df.rename(columns=col_mapper)
            else:
                # TODO: implement when adding single column test
                raise NotImplementedError()

        return _construct_dummy, _dummy_transform, None

    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        return input_schema


def _test_input_setter_getter(est_class: Any) -> None:
    if est_class().hasParam("inputCol"):
        assert est_class(inputCol="features").getInputCol() == "features"
        assert est_class(inputCol=["f1", "f2"]).getInputCols() == ["f1", "f2"]
        assert est_class(inputCols=["f1", "f2"]).getInputCols() == ["f1", "f2"]

        assert est_class().setInputCol("features").getInputCol() == "features"
        assert est_class().setInputCol(["f1", "f2"]).getInputCols() == ["f1", "f2"]
        assert est_class().setInputCols(["f1", "f2"]).getInputCols() == ["f1", "f2"]

    else:
        assert est_class().hasParam("featuresCol")

        assert est_class(featuresCol="features").getFeaturesCol() == "features"
        assert est_class(featuresCol=["f1", "f2"]).getFeaturesCols() == ["f1", "f2"]
        assert est_class(featuresCols=["f1", "f2"]).getFeaturesCols() == ["f1", "f2"]

        assert est_class().setFeaturesCol("features").getFeaturesCol() == "features"
        assert est_class().setFeaturesCol(["f1", "f2"]).getFeaturesCols() == [
            "f1",
            "f2",
        ]
        assert est_class().setFeaturesCols(["f1", "f2"]).getFeaturesCols() == [
            "f1",
            "f2",
        ]


def _test_est_copy(
    Estimator: Type[_CumlEstimator],
    input_spark_params: Dict[str, Any],
    cuml_params_update: Optional[Dict[str, Any]],
) -> None:
    """
    This tests the copy() function of an estimator object.
    For Spark-specific parameters (e.g. enable_sparse_data_optim in LogisticRegression), set cuml_params_update to None.
    """

    est = Estimator()
    copy_params = {getattr(est, p): input_spark_params[p] for p in input_spark_params}
    est_copy = est.copy(copy_params)

    # handle Spark-Rapids-ML-only params
    if cuml_params_update is None:
        for param in input_spark_params:
            assert est_copy.getOrDefault(param) == input_spark_params[param]
        return

    res_cuml_params = est.cuml_params.copy()
    res_cuml_params.update(cuml_params_update)
    assert (
        est.cuml_params != res_cuml_params
    ), "please modify cuml_params_update because it does not change the default estimator.cuml_params"
    assert est_copy.cuml_params == res_cuml_params

    # test init function
    est_init = Estimator(**input_spark_params)
    assert est_init.cuml_params == res_cuml_params


def _test_model_copy(
    gpu_model: Params,
    cpu_model: Params,
    input_spark_params: Dict[str, Any],
) -> None:
    """
    This tests the copy() function of a model object.
    """

    gpu_attrs = {
        getattr(gpu_model, p): input_spark_params[p] for p in input_spark_params
    }
    gpu_model_copy = gpu_model.copy(gpu_attrs)

    cpu_attrs = {
        getattr(cpu_model, p): input_spark_params[p] for p in input_spark_params
    }
    cpu_model_copy = cpu_model.copy(cpu_attrs)

    for p in input_spark_params:
        assert gpu_model_copy.getOrDefault(p) == input_spark_params[p]
        assert gpu_model_copy.getOrDefault(p) == cpu_model_copy.getOrDefault(p)
    return


def test_default_cuml_params() -> None:
    cuml_params = get_default_cuml_parameters([CumlDummy], ["b"])
    spark_params = SparkRapidsMLDummy()._get_cuml_params_default()
    assert cuml_params == spark_params


def test_dummy_params(gpu_number: int, tmp_path: str) -> None:
    # Default constructor
    default_spark_params = {
        "alpha": 1.0,  # a
        # "beta": 2,            # should raise exception if defined
        "gamma": "three",  # should be ignored
        "k": 4,  # k
    }
    default_cuml_params = {
        "a": 1.0,  # default value for Spark 'alpha'
        # "b": 20               # should be dropped
        "k": 4,  # default value for Spark 'k'
        "x": 40.0,  # default value for cuML
    }
    default_dummy = SparkRapidsMLDummy()
    assert_params(default_dummy, default_spark_params, default_cuml_params)

    # Spark constructor (with ignored param "gamma")
    spark_params = {"alpha": 2.0, "gamma": "test", "k": 1}
    spark_dummy = SparkRapidsMLDummy(
        m=0, n=0, partition_num=0, runtime_check=True, **spark_params
    )
    expected_spark_params = default_spark_params.copy()
    expected_spark_params.update(spark_params)
    expected_cuml_params = default_cuml_params.copy()
    expected_cuml_params.update({"a": 2.0, "k": 1})
    assert_params(spark_dummy, expected_spark_params, expected_cuml_params)

    # cuML constructor
    cuml_params = {"a": 1.1, "k": 2, "x": 3.3}
    cuml_dummy = SparkRapidsMLDummy(
        m=0, n=0, partition_num=0, runtime_check=True, **cuml_params
    )
    expected_spark_params = default_spark_params.copy()
    expected_spark_params.update(
        {
            "alpha": 1.1,
            "k": 2,
        }
    )
    expected_cuml_params = default_cuml_params.copy()
    expected_cuml_params.update(cuml_params)
    assert_params(cuml_dummy, expected_spark_params, expected_cuml_params)

    # Estimator persistence
    path = tmp_path + "/dummy_tests"
    estimator_path = f"{path}/dummy_estimator"
    cuml_dummy.write().overwrite().save(estimator_path)
    loaded_dummy = SparkRapidsMLDummy.load(estimator_path)
    assert_params(loaded_dummy, expected_spark_params, expected_cuml_params)

    # Spark constructor (with error param "beta")
    spark_params = {"alpha": 2.0, "beta": 0, "k": 1}
    with pytest.raises(ValueError, match="Spark Param 'beta' is not supported by cuML"):
        spark_dummy = SparkRapidsMLDummy(
            m=0, n=0, partition_num=0, runtime_check=True, **spark_params
        )

    # cuML constructor (with unsupported param "b")
    cuml_params = {"a": 1.1, "b": 0, "k": 2, "x": 3.3}
    with pytest.raises(ValueError, match="Unsupported param 'b'"):
        cuml_dummy = SparkRapidsMLDummy(
            m=0, n=0, partition_num=0, runtime_check=True, **cuml_params
        )

    # test the parameter copy
    dummy = SparkRapidsMLDummy()
    dummy2 = dummy.copy({dummy.alpha: 1111})
    assert dummy.getOrDefault(dummy.alpha) == 1
    assert dummy.cuml_params["a"] == 1
    assert dummy2.getOrDefault(dummy.alpha) == 1111
    assert dummy2.cuml_params["a"] == 1111


def test_dummy(gpu_number: int, tmp_path: str) -> None:
    data = [
        [1.0, 4.0, 4.0, 4.0],
        [2.0, 2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0, 2.0],
        [3.0, 3.0, 3.0, 2.0],
        [5.0, 2.0, 1.0, 3.0],
    ]
    m = len(data)
    n = len(data[0])
    input_cols = ["c1", "c2", "c3", "c4"]

    max_records_per_batch = 1

    def assert_estimator(dummy: SparkRapidsMLDummy) -> None:
        assert dummy.getInputCols() == input_cols
        assert dummy.cuml_params == {"a": 100, "k": 4, "x": 40.0}
        assert dummy.num_workers == gpu_number

    def ceiling_division(n: int, d: int) -> int:
        return -(n // -d)

    # Generate estimator
    dummy = SparkRapidsMLDummy(
        inputCols=input_cols,
        a=100,
        num_workers=gpu_number,
        partition_num=ceiling_division(m, max_records_per_batch),
        m=m,
        n=n,
    )

    assert_estimator(dummy)

    # Estimator persistence
    path = tmp_path + "/dummy_tests"
    estimator_path = f"{path}/dummy_estimator"
    dummy.write().overwrite().save(estimator_path)
    dummy_loaded = SparkRapidsMLDummy.load(estimator_path)
    assert_estimator(dummy_loaded)

    def assert_model(model: SparkRapidsMLDummyModel) -> None:
        assert model.model_attribute_a == 1024
        assert model.model_attribute_b == "hello dummy"
        assert model.cuml_params == {"a": 100, "k": 4, "x": 40.0}
        assert model.num_workers == gpu_number

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_records_per_batch)}
    from .sparksession import CleanSparkSession

    # Estimator fit and get a model
    with CleanSparkSession(conf) as spark:
        df = spark.sparkContext.parallelize(data).toDF(input_cols)
        model: SparkRapidsMLDummyModel = dummy.fit(df)
        assert_model(model)
        # Model persistence
        model_path = f"{path}/dummy_model"
        model.write().overwrite().save(model_path)
        model_loaded = SparkRapidsMLDummyModel.load(model_path)
        assert_model(model_loaded)

        dummy2 = dummy.copy()
        assert dummy2.cuml_params["a"] == 100
        with pytest.raises(
            Exception,
            match="assert {'a': 9876.0, 'k': 4, 'x': 40.0} == {'a': 100, 'k': 4, 'x': 40.0}",
        ):
            dummy2.fit(df, {dummy2.alpha: 9876.0})
        assert dummy2.cuml_params["a"] == 100
        assert dummy2.getOrDefault(dummy2.alpha) == 100

        dummy3 = SparkRapidsMLDummy(
            inputCols=input_cols,
            a=100,
            num_workers=gpu_number,
            partition_num=ceiling_division(m, max_records_per_batch),
            m=m,
            n=n,
            runtime_check=False,  # don't assert on the runtime.
        )
        model3 = dummy3.fit(df, {dummy3.alpha: 9876.0})
        assert dummy3.cuml_params["a"] == 100
        assert dummy3.getOrDefault(dummy3.alpha) == 100
        assert model3.cuml_params["a"] == 9876.0
        assert model3.getOrDefault(model3.alpha) == 9876.0

        # Transform the training dataset with a clean spark
        with CleanSparkSession() as clean_spark:
            test_df = clean_spark.sparkContext.parallelize(data, m).toDF(input_cols)
            transformed_df = model.transform(test_df)

            ret = transformed_df.collect()
            assert len(ret) == m

            # Compare data
            for x, y in zip(ret, data):
                for i in range(n):
                    assert x[i] == y[i]


def test_num_workers_validation() -> None:
    from .sparksession import CleanSparkSession

    with CleanSparkSession() as spark:
        data = [
            [1.0, 4.0, 4.0, 4.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 2.0],
            [3.0, 3.0, 3.0, 2.0],
            [5.0, 2.0, 1.0, 3.0],
        ]
        m = len(data)
        n = len(data[0])
        input_cols = ["c1", "c2", "c3", "c4"]

        df = spark.sparkContext.parallelize(data).toDF(input_cols)

        dummy = SparkRapidsMLDummy(
            inputCols=input_cols,
            a=100,
            num_workers=55,
            partition_num=1,
            m=m,
            n=n,
        )

        with pytest.raises(
            ValueError,
            match="The num_workers \(55\) should be less than or equal to spark default parallelism",
        ):
            dummy.fit(df)


def test_stage_level_scheduling() -> None:
    dummy = SparkRapidsMLDummy()

    standalone_conf = (
        SparkConf()
        .setMaster("spark://foo")
        .set("spark.executor.cores", "12")
        .set("spark.task.cpus", "1")
        .set("spark.executor.resource.gpu.amount", "1")
        .set("spark.task.resource.gpu.amount", "0.08")
    )

    # the correct configurations should not skip stage-level scheduling
    assert not dummy._skip_stage_level_scheduling("3.4.0", standalone_conf)
    assert not dummy._skip_stage_level_scheduling("3.4.1", standalone_conf)
    assert not dummy._skip_stage_level_scheduling("3.5.0", standalone_conf)
    assert not dummy._skip_stage_level_scheduling("3.5.1", standalone_conf)

    # spark version < 3.4.0
    assert dummy._skip_stage_level_scheduling("3.3.0", standalone_conf)

    # spark.executor.cores is not set
    bad_conf = (
        SparkConf()
        .setMaster("spark://foo")
        .set("spark.task.cpus", "1")
        .set("spark.executor.resource.gpu.amount", "1")
        .set("spark.task.resource.gpu.amount", "0.08")
    )
    assert dummy._skip_stage_level_scheduling("3.4.0", bad_conf)

    # spark.executor.cores=1
    bad_conf = (
        SparkConf()
        .setMaster("spark://foo")
        .set("spark.executor.cores", "1")
        .set("spark.task.cpus", "1")
        .set("spark.executor.resource.gpu.amount", "1")
        .set("spark.task.resource.gpu.amount", "0.08")
    )
    assert dummy._skip_stage_level_scheduling("3.4.0", bad_conf)

    # spark.executor.resource.gpu.amount is not set
    bad_conf = (
        SparkConf()
        .setMaster("spark://foo")
        .set("spark.executor.cores", "12")
        .set("spark.task.cpus", "1")
        .set("spark.task.resource.gpu.amount", "0.08")
    )
    assert dummy._skip_stage_level_scheduling("3.4.0", bad_conf)

    # spark.executor.resource.gpu.amount>1
    bad_conf = (
        SparkConf()
        .setMaster("spark://foo")
        .set("spark.executor.cores", "12")
        .set("spark.task.cpus", "1")
        .set("spark.executor.resource.gpu.amount", "2")
        .set("spark.task.resource.gpu.amount", "0.08")
    )
    assert dummy._skip_stage_level_scheduling("3.4.0", bad_conf)

    # spark.task.resource.gpu.amount is not set
    bad_conf = (
        SparkConf()
        .setMaster("spark://foo")
        .set("spark.executor.cores", "12")
        .set("spark.task.cpus", "1")
        .set("spark.executor.resource.gpu.amount", "1")
    )
    assert not dummy._skip_stage_level_scheduling("3.4.0", bad_conf)

    # spark.task.resource.gpu.amount=1
    bad_conf = (
        SparkConf()
        .setMaster("spark://foo")
        .set("spark.executor.cores", "12")
        .set("spark.task.cpus", "1")
        .set("spark.executor.resource.gpu.amount", "1")
        .set("spark.task.resource.gpu.amount", "1")
    )
    assert dummy._skip_stage_level_scheduling("3.4.0", bad_conf)

    # For Yarn and K8S
    for mode in ["yarn", "k8s://"]:
        for gpu_amount in ["0.08", "0.2", "1.0"]:
            conf = (
                SparkConf()
                .setMaster(mode)
                .set("spark.executor.cores", "12")
                .set("spark.task.cpus", "1")
                .set("spark.executor.resource.gpu.amount", "1")
                .set("spark.task.resource.gpu.amount", gpu_amount)
            )
            assert dummy._skip_stage_level_scheduling("3.3.0", conf)
            assert dummy._skip_stage_level_scheduling("3.4.0", conf)
            assert dummy._skip_stage_level_scheduling("3.4.1", conf)
            assert dummy._skip_stage_level_scheduling("3.5.0", conf)

            # This will be fixed when spark 4.0.0 is released.
            if gpu_amount == "1.0":
                assert dummy._skip_stage_level_scheduling("3.5.1", conf)
            else:
                # Starting from 3.5.1+, stage-level scheduling is working for Yarn and K8s
                assert not dummy._skip_stage_level_scheduling("3.5.1", conf)
