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
from abc import abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pyspark.mllib.random import RandomRDDs
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import array
from sklearn.datasets import (
    make_blobs,
    make_classification,
    make_low_rank_matrix,
    make_regression,
)

from benchmark.utils import WithSparkSession, inspect_default_params_from_func, to_bool


def dtype_to_pyspark_type(dtype: Union[np.dtype, str]) -> str:
    """Convert np.dtype to the corresponding pyspark type"""
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        return "float"
    elif dtype == np.float64:
        return "double"
    else:
        raise RuntimeError("Unsupported dtype, found ", dtype)


class DataGen(object):
    """DataGen interface"""

    @abstractmethod
    def gen_dataframe(self, spark: SparkSession) -> Tuple[DataFrame, List[str]]:
        raise NotImplementedError()


class DataGenBase(DataGen):
    """Base class datagen"""

    def __init__(self) -> None:
        # Global parameters
        self._parser = argparse.ArgumentParser()
        self._parser.add_argument(
            "--num_rows",
            type=int,
            default=100,
            help="total number of rows. default to 100",
        )
        self._parser.add_argument(
            "--num_cols",
            type=int,
            default=30,
            help="total number of columns. default to 30",
        )
        self._parser.add_argument(
            "--dtype",
            type=str,
            choices=["float64", "float32"],
            default="float32",
            help="the data type, default to float32",
        )
        self._parser.add_argument(
            "--feature_type",
            type=str,
            choices=["array", "vector", "multi_cols"],
            default="multi_cols",
            help="array - 1 column with ArrayType<dtype>, vector - 1 column with VectorUDT type, multi_cols: multiple columns with dtype. Default to multiple",
        )
        self._parser.add_argument(
            "--output_dir", type=str, required=True, help="the dataset output directory"
        )
        self._parser.add_argument(
            "--output_num_files", type=int, help="the number of files to be generated"
        )
        self._parser.add_argument(
            "--overwrite", action="store_true", help="if overwrite the output directory"
        )
        self._parser.add_argument(
            "--spark_confs",
            action="append",
            default=[],
            help="the optional spark configurations",
        )
        self._parser.add_argument(
            "--no_shutdown",
            action="store_true",
            help="do not stop spark session when finished",
        )

        def _restrict_train_size(x: float) -> float:
            # refer to https://stackoverflow.com/a/12117065/1928940
            try:
                x = float(x)
            except ValueError:
                raise argparse.ArgumentTypeError(f"{x} is not a floating-point literal")

            if x < 0.0 or x > 1.0:
                raise argparse.ArgumentTypeError(f"{x} is not in range [0.0, 1.0]")

            return x

        self._parser.add_argument(
            "--train_fraction",
            type=_restrict_train_size,  # type: ignore
            help="the value should be between 0.0 and 1.0 and represent "
            "the proportion of the dataset to include in the train split",
        )

        self._add_extra_parameters()

        self.args_: Optional[argparse.Namespace] = None

    def _add_extra_parameters(self) -> None:
        self.supported_extra_params = self._supported_extra_params()
        for name, value in self.supported_extra_params.items():
            if name == "effective_rank":
                help_msg = "The approximate number of singular vectors required to explain most of the data by linear combinations, refer to sklearn.datasets.make_low_rank_matrix()"
            elif name == "random_state":
                help_msg = "seed for random feature generation"
            elif name == "use_gpu":
                help_msg = "boolean for whether to use gpu processing and cupy library"
            elif name == "logistic_regression":
                help_msg = "boolean for whether the regression model is linear (continuous label) or logistic (binary label)"
            elif name == "density":
                help_msg = "the density ratio for the sparse feature matrix"
            elif name == "redundant_cols":
                help_msg = "the number of extra columns in the sparse matrix that is linear combination of original feature matrix, does not change rank"
            elif name == "n_informative":
                help_msg = "the number of non-zero weights in the regression model"
            elif name == "n_targets":
                help_msg = (
                    "the number of target labels to get from the regression model"
                )
            elif name == "bias":
                help_msg = "the bias parameter of the linear/logistic model"
            elif name == "noise":
                help_msg = "the strength of random noise by random sampling from a normal distribution centered at this value"
            elif name == "shuffle":
                help_msg = "boolean for whether the shuffle the rows and cols of the feature matrix"
            elif name == "tail_strength":
                help_msg = "tail strength for random low rank feature matrix generation, refer to sklearn.datasets.make_low_rank_matrix()"
            elif name == "density_curve":
                help_msg = "Specify columns wise density curve, support Linear or Exponential. The density of the generated matrix will have a density growing linearly/exponentially from the first to the last column. \
                            Argument density would not be used to represent the max density in the curve"
            else:
                help_msg = ""

            # Support multiple biases
            if name == "bias" or name == "density":
                self._parser.add_argument(
                    "--" + name, nargs="+", type=float, help=help_msg
                )
                continue

            if value is None:
                raise RuntimeError("Must convert None value to the correct type")
            elif type(value) is bool or value is bool:
                self._parser.add_argument("--" + name, type=to_bool, help=help_msg)
            elif type(value) is type:
                # value is already type
                self._parser.add_argument("--" + name, type=value, help=help_msg)
            else:
                # get the type from the value
                self._parser.add_argument("--" + name, type=type(value), help=help_msg)

    def _supported_extra_params(self) -> Dict[str, Any]:
        """Function to inspect the specific function to get the parameters and values"""
        return {}

    def _parse_arguments(self, argv: List[Any]) -> None:
        """Subclass must call this function in __init__"""
        self.args_ = self._parser.parse_args(argv)

        self.num_rows = self.args_.num_rows
        self.num_cols = self.args_.num_cols
        self.dtype = np.dtype(self.args_.dtype)

        self.pyspark_type = dtype_to_pyspark_type(self.dtype)
        self.feature_cols: List[str] = [f"c{i}" for i in range(self.num_cols)]
        self.schema = [f"{c} {self.pyspark_type}" for c in self.feature_cols]

        self.extra_params = {
            k: v
            for k, v in vars(self.args_).items()
            if k in self.supported_extra_params and v is not None
        }

    @property
    def args(self) -> Optional[argparse.Namespace]:
        return self.args_


class DefaultDataGen(DataGenBase):
    """Generate default dataset only containing features"""

    def __init__(self, argv: List[Any]) -> None:
        super().__init__()
        self._parse_arguments(argv)

    def _supported_extra_params(self) -> Dict[str, Any]:
        params = inspect_default_params_from_func(RandomRDDs.uniformVectorRDD, [])
        # must replace the None to the correct type
        params["numPartitions"] = int
        params["seed"] = int

        return params

    def gen_dataframe(self, spark: SparkSession) -> Tuple[DataFrame, List[str]]:
        params = self.extra_params

        if "seed" not in params:
            # for reproducible dataset.
            params["seed"] = 1

        print(f"Passing {params} to RandomRDDs.uniformVectorRDD")

        rdd = RandomRDDs.uniformVectorRDD(
            spark.sparkContext, self.num_rows, self.num_cols, **params
        ).map(
            lambda nparray: nparray.tolist()  # type: ignore
        )

        return (
            spark.createDataFrame(rdd, schema=",".join(self.schema)),
            self.feature_cols,
        )


class BlobsDataGen(DataGenBase):
    """Generate random dataset using sklearn.datasets.make_blobs,
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

    def gen_dataframe(self, spark: SparkSession) -> Tuple[DataFrame, List[str]]:
        "More information about the implementation can be found in RegressionDataGen."

        dtype = self.dtype
        params = self.extra_params

        if "random_state" not in params:
            # for reproducible dataset.
            params["random_state"] = 1

        print(f"Passing {params} to make_blobs")

        rows = self.num_rows
        cols = self.num_cols

        def make_blobs_udf(iter: Iterator[pd.Series]) -> pd.DataFrame:
            data, _ = make_blobs(n_samples=rows, n_features=cols, **params)
            data = data.astype(dtype)
            yield pd.DataFrame(data=data)

        return (
            spark.range(0, self.num_rows, 1, 1).mapInPandas(
                make_blobs_udf, schema=",".join(self.schema)  # type: ignore
            )
        ), self.feature_cols


class LowRankMatrixDataGen(DataGenBase):
    """Generate random dataset using sklearn.datasets.make_low_rank_matrix,
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
        "More information about the implementation can be found in RegressionDataGen."

        dtype = self.dtype

        params = self.extra_params

        if "random_state" not in params:
            # for reproducible dataset.
            params["random_state"] = 1

        rows = self.num_rows
        cols = self.num_cols

        print(f"Passing {params} to make_low_rank_matrix")

        def make_matrix_udf(iter: Iterator[pd.Series]) -> pd.DataFrame:
            data = make_low_rank_matrix(n_samples=rows, n_features=cols, **params)
            data = data.astype(dtype)
            yield pd.DataFrame(data=data)

        return (
            spark.range(0, self.num_rows, 1, 1).mapInPandas(
                make_matrix_udf, schema=",".join(self.schema)  # type: ignore
            )
        ), self.feature_cols


class RegressionDataGen(DataGenBase):
    """Generate regression dataset including features and label."""

    def __init__(self, argv: List[Any]) -> None:
        super().__init__()
        self._parse_arguments(argv)

    def _supported_extra_params(self) -> Dict[str, Any]:
        params = inspect_default_params_from_func(
            make_regression, ["n_samples", "n_features", "coef"]
        )
        # must replace the None to the correct type
        params["effective_rank"] = int
        params["random_state"] = int
        return params

    def gen_dataframe(self, spark: SparkSession) -> Tuple[DataFrame, List[str]]:
        num_cols = self.num_cols
        dtype = self.dtype

        params = self.extra_params

        if "random_state" not in params:
            # for reproducible dataset.
            params["random_state"] = 1

        print(f"Passing {params} to make_regression")

        def make_regression_udf(iter: Iterator[pd.Series]) -> pd.DataFrame:
            """Pandas udf to call make_regression of sklearn to generate regression dataset"""
            total_rows = 0
            for pdf in iter:
                total_rows += pdf.shape[0]
            # here we iterator all batches of a single partition to get total rows.
            # use 10% of num_cols for number of informative features, following ratio for defaults
            X, y = make_regression(n_samples=total_rows, n_features=num_cols, **params)
            data = np.concatenate(
                (X.astype(dtype), y.reshape(total_rows, 1).astype(dtype)), axis=1
            )
            del X
            del y
            yield pd.DataFrame(data=data)

        label_col = "label"
        self.schema.append(f"{label_col} {self.pyspark_type}")

        # Each make_regression calling will return regression dataset with different coef.
        # So force to only 1 task to generate the regression dataset, which may cause OOM
        # and perf issue easily. I tested this script can generate 100, 000, 000 * 30
        # matrix without issues with 60g executor memory, which, I think, is really enough
        # to do the perf test.
        return (
            spark.range(0, self.num_rows, 1, 1).mapInPandas(
                make_regression_udf, schema=",".join(self.schema)  # type: ignore
            )
        ), self.feature_cols


class ClassificationDataGen(DataGenBase):
    """Generate classification dataset including features and label."""

    def __init__(self, argv: List[Any]) -> None:
        super().__init__()
        self._parse_arguments(argv)

    def _supported_extra_params(self) -> Dict[str, Any]:
        params = inspect_default_params_from_func(
            make_classification, ["n_samples", "n_features", "weights"]
        )
        # must replace the None to the correct type
        params["random_state"] = int
        return params

    def gen_dataframe(self, spark: SparkSession) -> Tuple[DataFrame, List[str]]:
        num_cols = self.num_cols
        dtype = self.dtype

        params = self.extra_params

        if "random_state" not in params:
            # for reproducible dataset.
            params["random_state"] = 1

        print(f"Passing {params} to make_classification")

        def make_classification_udf(iter: Iterator[pd.Series]) -> pd.DataFrame:
            """Pandas udf to call make_classification of sklearn to generate classification dataset"""
            total_rows = 0
            for pdf in iter:
                total_rows += pdf.shape[0]
            # here we iterator all batches of a single partition to get total rows.
            X, y = make_classification(
                n_samples=total_rows, n_features=num_cols, **params
            )
            data = np.concatenate(
                (X.astype(dtype), y.reshape(total_rows, 1).astype(dtype)), axis=1
            )
            del X
            del y
            yield pd.DataFrame(data=data)

        label_col = "label"
        self.schema.append(f"{label_col} {self.pyspark_type}")

        # Each make_regression calling will return regression dataset with different coef.
        # So force to only 1 task to generate the regression dataset, which may cause OOM
        # and perf issue easily. I tested this script can generate 100, 000, 000 * 30
        # matrix without issues with 60g executor memory, which, I think, is really enough
        # to do the perf test.
        return (
            spark.range(0, self.num_rows, 1, 1).mapInPandas(
                make_classification_udf, schema=",".join(self.schema)  # type: ignore
            )
        ), self.feature_cols


def main(registered_data_gens: Dict[str, Any], repartition: bool) -> None:
    """
    python gen_data.py [regression|blobs|low_rank_matrix|default|classification] \
        --num_rows 5000 \
        --num_cols 3000 \
        --dtype "float64" \
        --output_dir "./5k_2k_float64.parquet" \
        --spark_confs "spark.master=local[*]" \
        --spark_confs "spark.driver.memory=128g"
    """

    parser = argparse.ArgumentParser(
        description="Generate random dataset.",
        usage="""python gen_data_distributed.py <type> [<args>]

Supported types are:
    blobs                 Generate random blobs datasets using sklearn's make_blobs
    regression            Generate random regression datasets using sklearn's make_regression
    classification        Generate random classification datasets using sklearn's make_classification
    low_rank_matrix       Generate random dataset using sklearn's make_low_rank_matrix
    sparse_regression     Generate random sparse regression datasets stored as sparse vectors
    default               Generate default dataset using pyspark RandomRDDs.uniformVectorRDD

Example:
python gen_data_distributed.py [regression|blobs|low_rank_matrix|default|classification|sparse_regression] \\
    --feature_type array \\
    --num_rows 5000 \\
    --num_cols 3000 \\
    --dtype "float64" \\
    --output_num_files 100 \\
    --overwrite \\
    --output_dir "./5k_3k_float64.parquet" \\
    --spark_confs "spark.master=local[*]" \\
    --spark_confs "spark.driver.memory=128g"
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

    # Must repartition for default.
    if args.type == "default":
        repartition = True

    model = args.type
    assert data_gen.args is not None
    args = data_gen.args

    with WithSparkSession(args.spark_confs, shutdown=(not args.no_shutdown)) as spark:
        df, feature_cols = data_gen.gen_dataframe(spark)

        if args.feature_type == "array":
            df = df.withColumn("feature_array", array(*feature_cols)).drop(
                *feature_cols
            )
        elif args.feature_type == "vector" and model != "sparse_regression":
            from pyspark.ml.feature import VectorAssembler

            df = (
                VectorAssembler()
                .setInputCols(feature_cols)
                .setOutputCol("feature_array")
                .transform(df)
                .drop(*feature_cols)
            )

        def write_files(dataframe: DataFrame, path: str) -> None:
            if args.output_num_files is not None and repartition:
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


if __name__ == "__main__":
    registered_data_gens = {
        "blobs": BlobsDataGen,
        "regression": RegressionDataGen,
        "classification": ClassificationDataGen,
        "low_rank_matrix": LowRankMatrixDataGen,
        "default": DefaultDataGen,
    }

    main(registered_data_gens=registered_data_gens, repartition=True)
