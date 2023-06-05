#
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
import random
import sys
from abc import abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from gen_data import DataGenBase, DefaultDataGen, dtype_to_pyspark_type
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import array
from sklearn.datasets import (
    make_blobs,
    make_classification,
    make_low_rank_matrix,
    make_regression,
)

from benchmark.utils import WithSparkSession, inspect_default_params_from_func, to_bool


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
        # TODO: set num_partitions if output_num_files is not provided
        # get default parallelism from spark context, get spark context from spark session
        # spark.sparkContext.defaultParallelism
        num_partitions = self.args.output_num_files

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
        def make_blobs_udf(iter: Iterator[pd.DataFrame]) -> pd.DataFrame:
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


if __name__ == "__main__":
    """
    python gen_data.py [regression|blobs|low_rank_matrix|default|classification] \
        --num_rows 5000 \
        --num_cols 3000 \
        --dtype "float64" \
        --output_dir "./5k_2k_float64.parquet" \
        --spark_confs "spark.master=local[*]" \
        --spark_confs "spark.driver.memory=128g"
    """

    registered_data_gens = {
        "blobs": BlobsDataGen,
        """
        "regression": RegressionDataGen,
        "classification": ClassificationDataGen,
        "low_rank_matrix": LowRankMatrixDataGen,
        """
        "default": DefaultDataGen,
    }

    parser = argparse.ArgumentParser(
        description="Generate random dataset.",
        usage="""gen_data.py <type> [<args>]

    Supported types are:
       blobs                 Generate random blobs datasets using sklearn's make_blobs
       regression            Generate random regression datasets using sklearn's make_regression
       classification        Generate random classification datasets using sklearn's make_classification
       low_rank_matrix       Generate random dataset using sklearn's make_low_rank_matrix
       default               Generate default dataset using pyspark RandomRDDs.uniformVectorRDD
    """,
    )
    parser.add_argument("type", help="Generate random dataset")
    # parse_args defaults to [1:] for args, but you need to
    # exclude the rest of the args too, or validation will fail
    args = parser.parse_args(sys.argv[1:2])

    if args.type not in registered_data_gens:
        print("Unrecognized type: ", args.type)
        parser.print_help()
        exit(1)

    data_gen = registered_data_gens[args.type](sys.argv[2:])  # type: ignore

    assert data_gen.args is not None
    args = data_gen.args

    with WithSparkSession(args.spark_confs, shutdown=(not args.no_shutdown)) as spark:
        df, feature_cols = data_gen.gen_dataframe(spark)

        if args.feature_type == "array":
            df = df.withColumn("feature_array", array(*feature_cols)).drop(
                *feature_cols
            )
        elif args.feature_type == "vector":
            from pyspark.ml.feature import VectorAssembler

            df = (
                VectorAssembler()
                .setInputCols(feature_cols)
                .setOutputCol("feature_array")
                .transform(df)
                .drop(*feature_cols)
            )

        def write_files(dataframe: DataFrame, path: str) -> None:
            # TODO: only need to repartition for DefaultDataGen
            if args.output_num_files is not None:
                dataframe = dataframe.repartition(args.output_num_files)

            writer = dataframe.write
            if args.overwrite:
                writer = writer.mode("overwrite")
            writer.parquet(path)

        if args.train_fraction is not None:
            train_df, eval_df = df.randomSplit(
                [args.train_fraction, 1 - args.train_fraction], seed=1
            )
            write_files(train_df, f"{args.output_dir}/train")
            write_files(eval_df, f"{args.output_dir}/eval")

        else:
            write_files(df, args.output_dir)

        df.printSchema()

        print("gen_data finished")
