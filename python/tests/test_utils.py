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

from spark_rapids_ml.utils import _get_default_params_from_func


def test_get_default_params_from_func() -> None:
    def dummy_func(a=1, b=2, c=3, d=4) -> None:  # type: ignore
        pass

    params = _get_default_params_from_func(dummy_func, ["c"])
    assert "c" not in params
    assert len(params) == 3
    assert params["a"] == 1
    assert params["d"] == 4


def test_clean_sparksession() -> None:
    from .sparksession import CleanSparkSession

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(1)}
    # Clean SparkSession with extra conf
    with CleanSparkSession(conf) as spark:
        assert int(spark.conf.get("spark.sql.execution.arrow.maxRecordsPerBatch")) == 1

    # Clean SparkSession
    with CleanSparkSession() as spark:
        assert (
            int(spark.conf.get("spark.sql.execution.arrow.maxRecordsPerBatch")) == 10000
        )

    # Test Nested SparkSession
    with CleanSparkSession(conf) as spark:
        assert int(spark.conf.get("spark.sql.execution.arrow.maxRecordsPerBatch")) == 1

        # Nested SparkSession will reset the conf
        with CleanSparkSession() as spark:
            assert (
                int(spark.conf.get("spark.sql.execution.arrow.maxRecordsPerBatch"))
                == 10000
            )

        # The conf has been reset.
        assert (
            int(spark.conf.get("spark.sql.execution.arrow.maxRecordsPerBatch")) == 10000
        )
