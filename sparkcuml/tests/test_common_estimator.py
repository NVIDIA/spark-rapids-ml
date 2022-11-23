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

from typing import Any, Callable, Dict, List, Tuple, Type, Union

import cudf
import numpy as np
import pandas as pd
from pyspark import Row, TaskContext
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType

from sparkcuml.core import (
    INIT_PARAMETERS_NAME,
    CumlT,
    _CumlEstimator,
    _CumlModel,
    _set_pyspark_cuml_cls_param_attrs,
)
from sparkcuml.utils import PartitionDescriptor


class CumlDummy(object):
    """
    A dummy class to mimic a cuml python class
    """

    def __init__(self, a=1, b=2, c=3) -> None:  # type: ignore
        super().__init__()
        self.a = a
        self.b = b
        self.c = c


class SparkCumlDummyModel(_CumlModel):
    """
    PySpark model of CumlDummy
    """

    def __init__(
        self, model_attribute_a: int, model_attribute_b: str, not_used: int = 1
    ) -> None:
        super().__init__(
            model_attribute_a=model_attribute_a, model_attribute_b=model_attribute_b
        )
        self.model_attribute_a = model_attribute_a
        self.model_attribute_b = model_attribute_b

    def _get_cuml_transform_func(
        self, dataset: DataFrame
    ) -> Tuple[
        Callable[..., CumlT],
        Callable[[CumlT, Union[cudf.DataFrame, np.ndarray]], pd.DataFrame],
    ]:
        model_attribute_a = self.model_attribute_a

        def _construct_dummy() -> CumlT:
            dummy = CumlDummy(a=101, b=102, c=103)
            return dummy

        def _dummy_transform(
            dummy: CumlT, df: Union[pd.DataFrame, np.ndarray]
        ) -> pd.DataFrame:
            assert dummy.a == 101
            assert dummy.b == 102
            assert dummy.c == 103

            assert model_attribute_a == 1024
            if isinstance(df, pd.DataFrame):
                return df
            else:
                # TODO: implement when adding single column test
                raise NotImplementedError()

        return _construct_dummy, _dummy_transform

    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        return input_schema


class SparkCumlDummy(_CumlEstimator):
    """
    PySpark estimator of CumlDummy
    """

    def __init__(
        self, m: int = 0, n: int = 0, partition_num: int = 0, **kwargs: Any
    ) -> None:
        super().__init__()
        self.set_params(**kwargs)
        self.m = m
        self.n = n
        self.partition_num = partition_num

    def _get_cuml_fit_func(
        self, dataset: DataFrame
    ) -> Callable[
        [Union[List[cudf.DataFrame], List[np.ndarray]], Dict[str, Any]], Dict[str, Any]
    ]:
        num_workers = self.get_num_workers()
        partition_num = self.partition_num
        m = self.m
        n = self.n

        def _cuml_fit(
            df: Union[List[cudf.DataFrame], List[np.ndarray]], params: Dict[str, Any]
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
            assert init_params["a"] == 100
            assert "b" not in init_params
            assert init_params["c"] == 3
            dummy = CumlDummy(**init_params)
            assert dummy.a == 100
            assert dummy.b == 2
            assert dummy.c == 3

            import time

            # sleep for 1 sec to bypass https://issues.apache.org/jira/browse/SPARK-40932
            time.sleep(1)

            return {"model_attribute_a": [1024], "model_attribute_b": "hello dummy"}

        return _cuml_fit

    def _out_schema(self) -> Union[StructType, str]:
        return "model_attribute_a int, model_attribute_b string"

    def _create_pyspark_model(self, result: Row) -> "SparkCumlDummyModel":
        assert result.model_attribute_a == 1024
        assert result.model_attribute_b == "hello dummy"
        return SparkCumlDummyModel.from_row(result)

    @classmethod
    def _cuml_cls(cls) -> Type:
        return CumlDummy

    @classmethod
    def _not_supported_param(cls) -> List[str]:
        """
        For some reason, spark cuml may not support all the parameters.
        In that case, we need to explicitly exclude them.
        """
        return ["b"]


_set_pyspark_cuml_cls_param_attrs(SparkCumlDummy, SparkCumlDummyModel)


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
        assert dummy.getOrDefault(dummy.a) == 100  # type: ignore
        assert not dummy.hasParam("b")
        assert dummy.getOrDefault(dummy.c) == 3  # type: ignore

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
        assert model.getOrDefault(model.a) == 100  # type: ignore
        assert not model.hasParam("b")
        assert model.getOrDefault(model.c) == 3  # type: ignore

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_records_per_batch)}
    from sparkcuml.tests.sparksession import CleanSparkSession

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

            # Compare data.
            for x, y in zip(ret, data):
                for i in range(n):
                    assert x[i] == y[i]
