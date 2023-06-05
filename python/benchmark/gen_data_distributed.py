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

import random
from abc import abstractmethod
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from gen_data import DataGenBase, DefaultDataGen, main
from pyspark.sql import DataFrame, SparkSession
from scipy import linalg
from sklearn.datasets import (
    make_blobs,
    make_classification,
    make_low_rank_matrix,
    make_regression,
)

from benchmark.utils import check_random_state, inspect_default_params_from_func


class DataGenBaseMeta(DataGenBase):
    """Base class datagen with meta info support"""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def gen_dataframe_and_meta(
        self, spark: SparkSession
    ) -> Tuple[DataFrame, List[str], np.ndarray]:
        raise NotImplementedError()

    def gen_dataframe(self, spark: SparkSession) -> Tuple[DataFrame, List[str]]:
        df, feature_cols, _ = self.gen_dataframe_and_meta(spark)
        return df, feature_cols


class BlobsDataGen(DataGenBaseMeta):
    """Generate random dataset using distributed calls to sklearn.datasets.make_blobs,
    which creates blobs for benchmarking unsupervised clustering algorithms (e.g. KMeans)
    """

    def __init__(self, argv: List[Any]) -> None:
        super().__init__()
        self._parse_arguments(argv)

    def _supported_extra_params(self) -> Dict[str, Any]:
        params = inspect_default_params_from_func(
            make_blobs, ["n_samples", "n_features", "return_centers"]
        )
        # must replace the None to the correct type
        params["centers"] = int
        params["random_state"] = int

        return params

    def gen_dataframe_and_meta(
        self, spark: SparkSession
    ) -> Tuple[DataFrame, List[str], np.ndarray]:
        dtype = self.dtype
        params = self.extra_params

        if "random_state" not in params:
            # for reproducible dataset.
            params["random_state"] = 1

        print(f"Passing {params} to make_blobs")

        rows = self.num_rows
        cols = self.num_cols
        assert self.args is not None
        num_partitions = self.args.output_num_files

        # Set num_partitions to Spark's default if output_num_files is not provided.
        if num_partitions is None:
            num_partitions = spark.sparkContext.defaultParallelism

        # Produce partition seeds for reproducibility.
        random.seed(params["random_state"])
        seed_maxval = 100 * num_partitions
        partition_seeds = random.sample(range(1, seed_maxval), num_partitions)

        partition_sizes = [rows // num_partitions] * num_partitions
        partition_sizes[-1] += rows % num_partitions

        # Generate centers upfront.
        _, _, centers = make_blobs(
            n_samples=0, n_features=cols, **params, return_centers=True
        )

        # Update params for partition-specific calls.
        params["centers"] = centers
        del params["random_state"]

        # UDF to distribute make_blobs() calls across partitions. Each partition
        # produces an equal fraction of the total samples around the predefined centers.
        def make_blobs_udf(iter: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
            for pdf in iter:
                partition_index = pdf.iloc[0][0]
                n_partition_samples = partition_sizes[partition_index]
                data, labels = make_blobs(
                    n_samples=n_partition_samples,
                    n_features=cols,
                    **params,
                    random_state=partition_seeds[partition_index],
                )
                data = np.concatenate(
                    (
                        data.astype(dtype),
                        labels.reshape(n_partition_samples, 1).astype(dtype),
                    ),
                    axis=1,
                )
                yield pd.DataFrame(data=data)

        label_col = "label"
        self.schema.append(f"{label_col} {self.pyspark_type}")

        return (
            (
                spark.range(
                    0, num_partitions, numPartitions=num_partitions
                ).mapInPandas(make_blobs_udf, schema=",".join(self.schema))
            ),
            self.feature_cols,
            centers,
        )


class LowRankMatrixDataGen(DataGenBase):
    """Generate random dataset using a distributed version of sklearn.datasets.make_low_rank_matrix,
    which creates large low rank matrices for benchmarking dimensionality reduction algos like pca
    """

    def __init__(self, argv: List[Any]) -> None:
        super().__init__()
        self._parse_arguments(argv)

    def _supported_extra_params(self) -> Dict[str, Any]:
        params = inspect_default_params_from_func(
            make_low_rank_matrix, ["n_samples", "n_features"]
        )
        # must replace the None to the correct type
        params["random_state"] = int
        return params

    def gen_dataframe(self, spark: SparkSession) -> Tuple[DataFrame, List[str]]:
        dtype = self.dtype
        params = self.extra_params

        if "random_state" not in params:
            # for reproducible dataset.
            params["random_state"] = 1

        print(f"Passing {params} to make_low_rank_matrix")

        rows = self.num_rows
        cols = self.num_cols
        assert self.args is not None
        num_partitions = self.args.output_num_files
        generator = check_random_state(params["random_state"])
        n = min(rows, cols)
        # If params not provided, set to defaults.
        tail_strength = params.get("tail_strength", 0.5)
        effective_rank = params.get("effective_rank", 10)

        partition_sizes = [rows // num_partitions] * num_partitions
        partition_sizes[-1] += rows % num_partitions
        # Check sizes to ensure QR decomp produces a matrix of the correct dimension.
        for size in partition_sizes:
            assert (
                size >= cols
            ), f"Num samples per partition ({size}) must be >= num_features ({cols}); \
                                    decrease output_num_files to <= {rows // cols}"

        # Generate U, S, V, the SVD decomposition of the output matrix.
        # S and V are generated upfront, U is generated across partitions.
        singular_ind = np.arange(n, dtype=np.float64)
        low_rank = (1 - params["tail_strength"]) * np.exp(
            -1.0 * (singular_ind / params["effective_rank"]) ** 2
        )
        tail = tail_strength * np.exp(-0.1 * singular_ind / effective_rank)
        s = np.identity(n) * (low_rank + tail)
        # compute V upfront
        v, _ = linalg.qr(
            generator.standard_normal(size=(cols, n)),
            mode="economic",
            check_finite=False,
        )

        # UDF for distributed generation of U and the resultant product U*S*V.T
        def make_matrix_udf(iter: Iterable[pd.Series]) -> Iterable[pd.DataFrame]:
            for pdf in iter:
                partition_index = pdf.iloc[0][0]
                n_partition_rows = partition_sizes[partition_index]
                u, _ = linalg.qr(
                    generator.standard_normal(size=(n_partition_rows, n)),
                    mode="economic",
                    check_finite=False,
                )
                # Include partition-wise normalization to ensure overall unit norm.
                u *= np.sqrt(1 / num_partitions)
                mat = np.dot(np.dot(u, s), v.T)
                yield pd.DataFrame(data=mat)

        return (
            (
                spark.range(
                    0, num_partitions, numPartitions=num_partitions
                ).mapInPandas(make_matrix_udf, schema=",".join(self.schema))
            ),
            self.feature_cols,
        )


if __name__ == "__main__":
    """
    See gen_data.main for more info.
    """

    registered_data_gens = {
        "blobs": BlobsDataGen,
        "default": DefaultDataGen,
        "low_rank_matrix": LowRankMatrixDataGen,
    }

    main(registered_data_gens=registered_data_gens, repartition=False)
