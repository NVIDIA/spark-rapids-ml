#
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

from typing import Dict, Optional

import numpy as np

from spark_rapids_ml.utils import (
    _concat_and_free,
    _get_default_params_from_func,
    _unsupported_methods_attributes,
)


def test_get_default_params_from_func() -> None:
    def dummy_func(a=1, b=2, c=3, d=4) -> None:  # type: ignore
        pass

    params = _get_default_params_from_func(dummy_func, ["c"])
    assert "c" not in params
    assert len(params) == 3
    assert params["a"] == 1
    assert params["d"] == 4


def test_concat_and_free() -> None:
    a = np.array([[0.0, 1.0], [2.0, 3.0]], order="F")
    arr_list = [a, a]
    concat = _concat_and_free(arr_list, order="C")
    assert len(arr_list) == 0
    assert concat.flags["C_CONTIGUOUS"]
    assert not concat.flags["F_CONTIGUOUS"]

    a = np.array([[0.0, 1.0], [2.0, 3.0]], order="C")
    arr_list = [a, a]
    concat = _concat_and_free(arr_list)
    assert len(arr_list) == 0
    assert not concat.flags["C_CONTIGUOUS"]
    assert concat.flags["F_CONTIGUOUS"]


def test_unsupported_methods_attributes() -> None:
    a = 1
    assert _unsupported_methods_attributes(a) == set()

    class A:
        @classmethod
        def _param_mapping(cls) -> Dict[str, Optional[str]]:
            return {"param1": "param2", "param3": None, "param4": ""}

        @classmethod
        def unsupported_method(cls) -> None:
            """Unsupported."""
            pass

        def unsupported_function(self) -> None:
            """Unsupported."""
            pass

        @classmethod
        def supported_method(cls) -> None:
            """supported"""
            pass

        def supported_function(self) -> None:
            """supported"""
            pass

    assert _unsupported_methods_attributes(A) == set(
        [
            "param3",
            "getParam3",
            "setParam3",
            "param4",
            "getParam4",
            "setParam4",
            "unsupported_method",
            "unsupported_function",
        ]
    )


def test_clean_sparksession() -> None:
    from .sparksession import CleanSparkSession

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(1)}
    # Clean SparkSession with extra conf
    with CleanSparkSession(conf) as spark:
        assert spark.conf.get("spark.sql.execution.arrow.maxRecordsPerBatch") == "1"

    # Clean SparkSession
    with CleanSparkSession() as spark:
        assert spark.conf.get("spark.sql.execution.arrow.maxRecordsPerBatch") == "10000"

    # Test Nested SparkSession
    with CleanSparkSession(conf) as spark:
        assert spark.conf.get("spark.sql.execution.arrow.maxRecordsPerBatch") == "1"

        # Nested SparkSession will reset the conf
        with CleanSparkSession() as spark:
            assert (
                spark.conf.get("spark.sql.execution.arrow.maxRecordsPerBatch")
                == "10000"
            )

        # The conf has been reset.
        assert spark.conf.get("spark.sql.execution.arrow.maxRecordsPerBatch") == "10000"
