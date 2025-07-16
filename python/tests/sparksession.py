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
from typing import Any, Dict

from pyspark.sql import SparkSession

from .conftest import _default_conf, get_spark_i_know_what_i_am_doing


# sparksession.py is copied from spark-rapids
def _from_scala_map(scala_map) -> Dict[str, Any]:  # type: ignore
    ret = {}
    # The value we get is a scala map, not a java map, so we need to jump through some hoops
    keys = scala_map.keys().iterator()  # type: ignore
    while keys.hasNext():  # type: ignore
        key = keys.next()  # type: ignore
        ret[key] = scala_map.get(key).get()  # type: ignore
    return ret  # type: ignore


_spark = get_spark_i_know_what_i_am_doing()
# Have to reach into a private member to get access to the API we need
_orig_conf = _from_scala_map(_spark.conf._jconf.getAll())  # type: ignore
_orig_conf_keys = _orig_conf.keys()  # type: ignore


class CleanSparkSession:
    """
    A context manager to auto reset spark conf.
    """

    def __init__(self, conf: Dict[str, Any] = {}) -> None:
        self.conf = conf
        self.spark = _spark

    def __enter__(self) -> SparkSession:
        self._reset_spark_session_conf()
        self._set_all_confs(self.conf)
        return self.spark

    def __exit__(self, *args: Any) -> None:
        self._reset_spark_session_conf()

    def _set_all_confs(self, conf: Dict[str, Any]) -> None:
        newconf = _default_conf.copy()
        newconf.update(conf)
        for key, value in newconf.items():
            if self.spark.conf.get(key, None) != value:
                self.spark.conf.set(key, value)

    def _reset_spark_session_conf(self) -> None:
        """Reset all of the configs for a given spark session."""
        self._set_all_confs(_orig_conf)
        # Have to reach into a private member to get access to the API we need
        current_keys = _from_scala_map(self.spark.conf._jconf.getAll()).keys()  # type: ignore
        for key in current_keys:
            if key not in _orig_conf_keys:
                self.spark.conf.unset(key)
