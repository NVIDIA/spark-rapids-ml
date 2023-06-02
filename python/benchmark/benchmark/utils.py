# Copyright (c) 2007-2023 The scikit-learn developers. All rights reserved.
# Modifications copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
import inspect
import numbers
from distutils.util import strtobool
from time import time
from typing import Any, Callable, Dict, List

import numpy as np
from pyspark.sql import SparkSession


class WithSparkSession(object):
    def __init__(self, confs: List[str], shutdown: bool = True) -> None:
        builder = SparkSession.builder
        for conf in confs:
            key, value = conf.split("=")
            builder = builder.config(key, value)
        self.spark = builder.getOrCreate()
        self.shutdown = shutdown

    def __enter__(self) -> SparkSession:
        return self.spark

    def __exit__(self, *args: Any) -> None:
        if self.shutdown:
            print("stopping spark session")
            self.spark.stop()


def with_benchmark(phrase: str, action: Callable) -> Any:
    start = time()
    result = action()
    end = time()
    print("-" * 100)
    duration = round(end - start, 2)
    print("{}: {} seconds".format(phrase, duration))
    print("-" * 100)
    return result, duration


def inspect_default_params_from_func(
    func: Callable, unsupported_set: List[str] = []
) -> Dict[str, Any]:
    """
    Returns a dictionary of parameters and their default value of function fn.
    Only the parameters with a default value will be included.
    """
    sig = inspect.signature(func)
    filtered_params_dict = {}
    for parameter in sig.parameters.values():
        # Remove parameters without a default value and those in the unsupported_set
        if (
            parameter.default is not parameter.empty
            and parameter.default is not None
            and parameter.name not in unsupported_set
        ):
            filtered_params_dict[parameter.name] = parameter.default
    return filtered_params_dict


def to_bool(literal: str) -> bool:
    return bool(strtobool(literal))


def check_random_state(seed: int) -> np.random.RandomState:
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )
