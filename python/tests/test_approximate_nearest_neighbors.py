from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
from _pytest.logging import LogCaptureFixture
from pyspark.sql import DataFrame
from sklearn.datasets import make_blobs

from spark_rapids_ml.core import alias
from spark_rapids_ml.knn import (
    ApproximateNearestNeighbors,
    ApproximateNearestNeighborsModel,
)

from .sparksession import CleanSparkSession
from .test_nearest_neighbors import (
    NNEstimator,
    NNModel,
    func_test_example_no_id,
    func_test_example_with_id,
)
from .utils import (
    array_equal,
    create_pyspark_dataframe,
    get_default_cuml_parameters,
    idfn,
    pyspark_supported_feature_types,
)


def test_default_cuml_params() -> None:
    from cuml import NearestNeighbors as CumlNearestNeighbors

    cuml_params = get_default_cuml_parameters(
        [CumlNearestNeighbors],
        [
            "handle",
            "metric",
            "p",
            "algo_params",
            "metric_expanded",
            "metric_params",
            "output_type",
        ],
    )

    spark_params = ApproximateNearestNeighbors()._get_cuml_params_default()
    cuml_params["algorithm"] = "ivfflat"  # change cuml default 'auto' to 'ivfflat'
    assert cuml_params == spark_params


@pytest.mark.parametrize(
    "algo_and_params", [("brute", None), ("ivfflat", {"nlist": 1, "nprobe": 2})]
)
@pytest.mark.parametrize(
    "func_test", [func_test_example_no_id, func_test_example_with_id]
)
def test_example(
    algo_and_params: Tuple[str, Optional[dict[str, Any]]],
    func_test: Callable[[NNEstimator, str], Tuple[NNEstimator, NNModel]],
    gpu_number: int,
    tmp_path: str,
) -> None:
    algorithm = algo_and_params[0]
    algo_params = algo_and_params[1]

    gpu_knn = ApproximateNearestNeighbors(algorithm=algorithm, algo_params=algo_params)
    gpu_knn, gpu_model = func_test(tmp_path, gpu_knn)  # type: ignore

    for obj in [gpu_knn, gpu_model]:
        assert obj._cuml_params["algorithm"] == algorithm
        assert obj._cuml_params["algo_params"] == algo_params
