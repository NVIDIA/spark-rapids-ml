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

from typing import Union, Any

import cudf
from pyspark import Row, TaskContext
from pyspark.sql.types import StructType

from sparkcuml.core import _CumlEstimator, _CumlModel, INIT_PARAMETERS_NAME
from sparkcuml.utils import _set_pyspark_cuml_cls_param_attrs


class CumlDummy(object):
    """
    A dummy class to mimic a cuml python class
    """

    def __init__(self, a=1, b=2, c=3):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c


class SparkCumlDummyModel(_CumlModel):
    """
    PySpark model of CumlDummy
    """

    def __init__(self):
        super().__init__()


class SparkCumlDummy(_CumlEstimator):
    """
    PySpark estimator of CumlDummy
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.set_params(**kwargs)

    def _fit_internal(self, df: list[cudf.DataFrame], **kwargs) -> dict[str, Any]:
        # assert len(kwargs) == 3
        assert kwargs["rank"] == TaskContext.get().partitionId()
        assert "handle" in kwargs
        assert INIT_PARAMETERS_NAME in kwargs
        init_params = kwargs[INIT_PARAMETERS_NAME]
        assert init_params["a"] == 100
        assert init_params["c"] == 3
        dummy = CumlDummy(**init_params)
        assert dummy.a == 100
        assert dummy.b == 2
        assert dummy.c == 3

        return {
            "dummy": [1024],
        }

    def _out_schema(self) -> Union[StructType, str]:
        return "dummy int"

    def _create_pyspark_model(self, result: Row):
        assert result.dummy == 1024
        return SparkCumlDummyModel()

    @classmethod
    def _cuml_cls(cls):
        return CumlDummy

    @classmethod
    def _not_supported_param(cls) -> list[str]:
        """
        For some reason, spark cuml may not support all the parameters.
        In that case, we need to explicitly exclude them.
        """
        return ["b"]


_set_pyspark_cuml_cls_param_attrs(SparkCumlDummy, SparkCumlDummyModel)


def test_dummy(spark):
    data = [[1.0, 4.0, 4.0, 4.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 2.0],
            [3.0, 3.0, 3.0, 2.0],
            ]

    rdd = spark.sparkContext.parallelize(data)
    input_cols = ["c1", "c2", "c3", "c4"]
    df = rdd.toDF(input_cols)
    df.show()

    dummy = SparkCumlDummy(inputCols=input_cols, a=100)
    assert dummy.getInputCols() == input_cols
    assert dummy.getOrDefault(dummy.a) == 100
    assert not dummy.hasParam("b")
    assert dummy.getOrDefault(dummy.c) == 3
    model = dummy.fit(df)
