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

import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Generator, List

import cupy
import pytest
from pyspark.sql import SparkSession

dir_path = os.path.dirname(os.path.realpath(__file__))

gpu_discovery_script_path = f"{dir_path}/discover_gpu.sh"


def _get_devices() -> List[str]:
    """This works only if driver is the same machine of worker."""
    completed = subprocess.run(gpu_discovery_script_path, stdout=subprocess.PIPE)
    assert completed.returncode == 0, "Failed to execute discovery script."
    msg = completed.stdout.decode("utf-8")
    result = json.loads(msg)
    addresses = result["addresses"]
    return addresses


_gpu_number = min(len(_get_devices()), cupy.cuda.runtime.getDeviceCount())
# We restrict the max gpu numbers to use
_gpu_number = _gpu_number if _gpu_number < 4 else 4


@pytest.fixture
def gpu_number() -> int:
    return _gpu_number


@pytest.fixture
def tmp_path() -> Generator[str, None, None]:
    path = tempfile.mkdtemp(prefix="spark_rapids_ml_tests_")
    yield path
    shutil.rmtree(path)


_default_conf = {
    "spark.master": f"local[{_gpu_number}]",
    "spark.python.worker.reuse": "false",
    "spark.driver.host": "127.0.0.1",
    "spark.task.maxFailures": "1",
    "spark.driver.memory": "32g",
    "spark.sql.execution.pyspark.udf.simplifiedTraceback.enabled": "false",
    "spark.sql.pyspark.jvmStacktrace.enabled": "true",
    "spark.sql.execution.arrow.pyspark.enabled": "true",
}


def _get_spark() -> SparkSession:
    builder = SparkSession.builder.appName(name="spark-rapids-ml python tests")
    for k, v in _default_conf.items():
        builder.config(k, v)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    logging.getLogger("pyspark").setLevel(logging.WARN)
    return spark


_spark = _get_spark()


def get_spark_i_know_what_i_am_doing() -> SparkSession:
    """
    Get the current SparkSession.
    This should almost never be called directly instead you should call
    with_spark_session for spark_session.
    This is to guarantee that the session and it's config is setup in a repeatable way.
    """
    return _spark


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark as slow to run")
    config.addinivalue_line("markers", "compat: mark as compatibility test")


def pytest_collection_modifyitems(
    config: pytest.Config, items: List[pytest.Item]
) -> None:
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
