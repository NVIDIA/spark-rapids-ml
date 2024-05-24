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
import argparse
import sys

from benchmark.bench_approximate_nearest_neighbors import (
    BenchmarkApproximateNearestNeighbors,
)
from benchmark.bench_dbscan import BenchmarkDBSCAN
from benchmark.bench_kmeans import BenchmarkKMeans
from benchmark.bench_linear_regression import BenchmarkLinearRegression
from benchmark.bench_logistic_regression import BenchmarkLogisticRegression
from benchmark.bench_nearest_neighbors import BenchmarkNearestNeighbors
from benchmark.bench_pca import BenchmarkPCA
from benchmark.bench_random_forest import (
    BenchmarkRandomForestClassifier,
    BenchmarkRandomForestRegressor,
)
from benchmark.bench_umap import BenchmarkUMAP


class BenchmarkRunner:
    def __init__(self) -> None:
        registered_algorithms = {
            "approximate_nearest_neighbors": BenchmarkApproximateNearestNeighbors,
            "dbscan": BenchmarkDBSCAN,
            "kmeans": BenchmarkKMeans,
            "knn": BenchmarkNearestNeighbors,
            "linear_regression": BenchmarkLinearRegression,
            "pca": BenchmarkPCA,
            "random_forest_classifier": BenchmarkRandomForestClassifier,
            "random_forest_regressor": BenchmarkRandomForestRegressor,
            "logistic_regression": BenchmarkLogisticRegression,
            "umap": BenchmarkUMAP,
        }
        algorithms = "\n    ".join(registered_algorithms.keys())
        parser = argparse.ArgumentParser(
            description="Benchmark Spark Rapids ML algorithms",
            usage=f"""benchmark_runner.py <algorithm> [<args>]

        Supported algorithms are:
        {algorithms}
        """,
        )
        parser.add_argument("algorithm", help="benchmark the ML algorithms")
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])

        if args.algorithm not in registered_algorithms:
            print("Unrecognized algorithm: ", args.algorithm)
            parser.print_help()
            exit(1)

        self._runner: BenchmarkBase = registered_algorithms[args.algorithm](  # type: ignore
            sys.argv[2:]
        )

    def run(self) -> None:
        self._runner.run()


if __name__ == "__main__":
    """
    There're two ways to do the benchmark.

    1.
        python benchmark_runner.py [linear_regression] \
            --num_gpus=2 \
            --train_path=xxx \
            --spark_confs="spark.master=local[12]" \

    2.
        spark-submit --master local[12] benchmark_runner.py --num_gpus=2 --train_path=xxx
    """
    BenchmarkRunner().run()
