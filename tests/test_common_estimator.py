#
# Copyright (c) 2022, NVIDIA CORPORATION.
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

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, Union

import cudf
import numpy as np
import pandas as pd
import pytest
from pyspark import Row, TaskContext
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCols, HasOutputCols
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType

from spark_rapids_ml.core import (
    INIT_PARAMETERS_NAME,
    CumlInputType,
    CumlT,
    _CumlEstimator,
    _CumlModel,
)
from spark_rapids_ml.params import _CumlClass
from spark_rapids_ml.utils import PartitionDescriptor
from tests.utils import assert_params


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


class SparkCumlDummyClass(_CumlClass):
    @classmethod
    def _cuml_cls(cls) -> List[type]:
        return [CumlDummy]

    @classmethod
    def _param_mapping(cls) -> Mapping[str, Optional[str]]:
        return {
            "alpha": "a",  # direct map, different names
            "beta": None,  # unmapped, raise error if defined on Spark side
            "gamma": "",  # unmapped, ignore value from Spark side
            "k": "k",  # direct map, same name
        }

    @classmethod
    def _param_excludes(cls) -> List[str]:
        return ["b"]  # dropped from CuML side


class _SparkDummyParams(Params):
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
        super(_SparkDummyParams, self).__init__(*args)
        self._setDefault(
            alpha=1.0,
            # beta=2,         # leave undefined to test mapping to None
            gamma="three",
            k=4,
        )


class SparkCumlDummy(
    SparkCumlDummyClass, _CumlEstimator, _SparkDummyParams, HasInputCols, HasOutputCols
):
    """
    PySpark estimator of CumlDummy
    """

    def __init__(
        self, m: int = 0, n: int = 0, partition_num: int = 0, **kwargs: Any
    ) -> None:
        #

        super().__init__()
        self.set_params(**kwargs)
        self.m = m
        self.n = n
        self.partition_num = partition_num

    def setInputCols(self, value: List[str]) -> "SparkCumlDummy":
        return self._set(inputCols=value)

    def setOutputCols(self, value: List[str]) -> "SparkCumlDummy":
        return self._set(outputCols=value)

    def setAlpha(self, value: int) -> "SparkCumlDummy":
        return self.set_params({"alpha": value})

    def setBeta(self, value: int) -> "SparkCumlDummy":
        raise ValueError("Not supported")

    def setGamma(self, value: float) -> "SparkCumlDummy":
        return self.set_params({"gamma": value})

    def setK(self, value: str) -> "SparkCumlDummy":
        return self.set_params({"k": value})

    def _get_cuml_fit_func(
        self, dataset: DataFrame
    ) -> Callable[[CumlInputType, Dict[str, Any]], Dict[str, Any],]:
        num_workers = self.getNumWorkers()
        partition_num = self.partition_num
        m = self.m
        n = self.n

        # if the common framework tries to pickle the whole class,
        # it will throw exception since dataset is not picklable.
        self.test_pickle_dataframe = dataset

        def _cuml_fit(
            dfs: CumlInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            context = TaskContext.get()
            assert context is not None
            assert "handle" in params
            assert "part_sizes" in params
            assert "n" in params

            pd = PartitionDescriptor.build(params["part_sizes"], params["n"])

            assert pd.rank == context.partitionId()
            assert len(pd.parts_rank_size) == partition_num
            assert pd.m == m
            assert pd.n == n

            assert INIT_PARAMETERS_NAME in params
            init_params = params[INIT_PARAMETERS_NAME]
            assert init_params == {"a": 100, "k": 4, "x": 40.0}
            dummy = CumlDummy(**init_params)
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

    def _create_pyspark_model(self, result: Row) -> "SparkCumlDummyModel":
        assert result.dtype == np.dtype(np.float32).name
        assert result.n_cols == self.n
        assert result.model_attribute_a == 1024
        assert result.model_attribute_b == "hello dummy"
        return SparkCumlDummyModel.from_row(result)


class SparkCumlDummyModel(
    SparkCumlDummyClass, _CumlModel, _SparkDummyParams, HasInputCols, HasOutputCols
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
        self.set_params(**kwargs)

    def setInputCols(self, value: List[str]) -> "SparkCumlDummy":
        return self._set(inputCols=value)

    def setOutputCols(self, value: List[str]) -> "SparkCumlDummy":
        return self._set(outputCols=value)

    def _get_cuml_transform_func(
        self, dataset: DataFrame
    ) -> Tuple[
        Callable[..., CumlT],
        Callable[[CumlT, Union[cudf.DataFrame, np.ndarray]], pd.DataFrame],
    ]:
        model_attribute_a = self.model_attribute_a

        # if the common framework tries to pickle the whole class,
        # it will throw exception since dataset is not picklable.
        self.test_pickle_dataframe = dataset

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
                return df
            else:
                # TODO: implement when adding single column test
                raise NotImplementedError()

        return _construct_dummy, _dummy_transform

    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        return input_schema


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
        "x": 40.0,  # default value for CuML
    }
    default_dummy = SparkCumlDummy()
    assert_params(default_dummy, default_spark_params, default_cuml_params)

    # Spark constructor (with ignored param "gamma")
    spark_params = {"alpha": 2.0, "gamma": "test", "k": 1}
    spark_dummy = SparkCumlDummy(m=0, n=0, partition_num=0, **spark_params)
    expected_spark_params = default_spark_params.copy()
    expected_spark_params.update(spark_params)
    expected_cuml_params = default_cuml_params.copy()
    expected_cuml_params.update({"a": 2.0, "k": 1})
    assert_params(spark_dummy, expected_spark_params, expected_cuml_params)

    # CuML constructor
    cuml_params = {"a": 1.1, "k": 2, "x": 3.3}
    cuml_dummy = SparkCumlDummy(m=0, n=0, partition_num=0, **cuml_params)
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
    loaded_dummy = SparkCumlDummy.load(estimator_path)
    assert_params(loaded_dummy, expected_spark_params, expected_cuml_params)

    # Spark constructor (with error param "beta")
    spark_params = {"alpha": 2.0, "beta": 0, "k": 1}
    with pytest.raises(ValueError, match="Spark Param 'beta' is not supported by CuML"):
        spark_dummy = SparkCumlDummy(m=0, n=0, partition_num=0, **spark_params)

    # CuML constructor (with unsupported param "b")
    cuml_params = {"a": 1.1, "b": 0, "k": 2, "x": 3.3}
    with pytest.raises(ValueError, match="Unsupported param 'b'"):
        cuml_dummy = SparkCumlDummy(m=0, n=0, partition_num=0, **cuml_params)


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

    def assert_estimator(dummy: SparkCumlDummy) -> None:
        assert dummy.getInputCols() == input_cols
        assert dummy.cuml_params == {"a": 100, "k": 4, "x": 40.0}

    def ceiling_division(n: int, d: int) -> int:
        return -(n // -d)

    # Generate estimator
    dummy = SparkCumlDummy(
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
    dummy_loaded = SparkCumlDummy.load(estimator_path)
    assert_estimator(dummy_loaded)

    def assert_model(model: SparkCumlDummyModel) -> None:
        assert model.model_attribute_a == 1024
        assert model.model_attribute_b == "hello dummy"
        assert model.cuml_params == {"a": 100, "k": 4, "x": 40.0}

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_records_per_batch)}
    from .sparksession import CleanSparkSession

    # Estimator fit and get a model
    with CleanSparkSession(conf) as spark:
        df = spark.sparkContext.parallelize(data).toDF(input_cols)
        model: SparkCumlDummyModel = dummy.fit(df)
        assert_model(model)
        # Model persistence
        model_path = f"{path}/dummy_model"
        model.write().overwrite().save(model_path)
        model_loaded = SparkCumlDummyModel.load(model_path)
        assert_model(model_loaded)

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
