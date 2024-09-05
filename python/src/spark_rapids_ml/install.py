#
# Copyright (c) 2024, NVIDIA CORPORATION.
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

import importlib
import sys
import types
from typing import Any

_accelerated_attributes = {
    "feature": ["PCA", "PCAModel"],
    "clustering": ["KMeans", "KMeansModel"],
    "classification": [
        "LogisticRegression",
        "LogisticRegressionModel",
        "RandomForestClassifier",
        "RandomForestClassificationModel",
    ],
    "regression": [
        "LinearRegression",
        "LinearRegressionModel" "RandomForestRegressor",
        "RandomForestClassificationModel",
    ],
    "tuning": ["CrossValidator"],
}


_rapids_modules = {
    module_name: importlib.import_module(f"spark_rapids_ml.{module_name}")
    for module_name in _accelerated_attributes.keys()
}
_pyspark_modules = {
    module_name: importlib.import_module(f"pyspark.ml.{module_name}")
    for module_name in _accelerated_attributes.keys()
}


def _set_pyspark_mod_getattr(mod_name: str) -> None:
    proxy_module = types.ModuleType(f"pyspark.ml.${mod_name}")

    def _getattr(attr: str) -> Any:
        frame = sys._getframe()
        assert frame.f_back
        calling_path = frame.f_back.f_code.co_filename
        if any(
            [
                (
                    f"pyspark/ml/{mod_name}" in calling_path
                    or f"spark_rapids_ml/{mod_name}" in calling_path
                )
                for mod_name in _accelerated_attributes.keys()
            ]
        ) or (attr not in _accelerated_attributes[mod_name]):
            try:
                attr_val = getattr(_pyspark_modules[mod_name], attr)
            except:
                raise AttributeError(f"No attribute '{attr}'")

            return attr_val
        else:
            return getattr(_rapids_modules[mod_name], attr)

    setattr(proxy_module, "__getattr__", _getattr)
    sys.modules[f"pyspark.ml.{mod_name}"] = proxy_module


for mod_name in _accelerated_attributes.keys():
    _set_pyspark_mod_getattr(mod_name)
