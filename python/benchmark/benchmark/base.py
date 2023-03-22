#
# Copyright (c) 2023, NVIDIA CORPORATION.
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
import pprint
import subprocess
from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import array_to_vector
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col

from .utils import WithSparkSession, to_bool, with_benchmark


class BenchmarkBase:
    """Based class for benchmarking.

    This class handles command line argument parsing and execution of the benchmark.
    """

    _parser: argparse.ArgumentParser
    _args: argparse.Namespace
    _class_params: Dict[str, Any]

    def __init__(self, argv: List[Any]) -> None:
        """Parses command line arguments for the class under test."""
        print("=" * 100)
        print(self.__class__.__name__)
        print("=" * 100)

        # common params for all benchmark classes
        self._parser = argparse.ArgumentParser()
        self._parser.add_argument(
            "--num_gpus",
            type=int,
            default=1,
            help="number of GPUs to use. If num_gpus > 0, will run with the number of dataset partitions equal to num_gpus.",
        )
        self._parser.add_argument(
            "--num_cpus",
            type=int,
            default=6,
            help="number of CPUs to use",
        )
        self._parser.add_argument(
            "--num_runs",
            type=int,
            default=1,
            help="number of benchmark iterations (for cold/warm runs)",
        )
        self._parser.add_argument(
            "--report_path", type=str, default="", help="path to save benchmark results"
        )
        self._parser.add_argument(
            "--train_path",
            action="append",
            default=[],
            required=True,
            help="path to parquet data for training",
        )
        self._parser.add_argument(
            "--transform_path",
            action="append",
            default=[],
            help="path to parquet data for transform",
        )
        self._parser.add_argument("--spark_confs", action="append", default=[])
        self._parser.add_argument(
            "--no_shutdown",
            action="store_true",
            help="do not stop spark session when finished",
        )

        self._add_class_arguments()
        self._add_extra_arguments()
        self._parse_arguments(argv)

    def _add_extra_arguments(self) -> None:
        """Add command line arguments for the benchmarking environment."""
        pass

    def _add_class_arguments(self) -> None:
        """
        Add command line arguments for the parameters to be supplied to the class under test.

        The default implementation automatically constructs arguments from the dictionary returned
        from the :py:func:`_supported_class_params()` method.
        """
        for name, value in self._supported_class_params().items():
            (value, help) = value if isinstance(value, tuple) else (value, None)
            help = "PySpark parameter" if help is None else help
            if value is None:
                raise RuntimeError("Must convert None value to the correct type")
            elif type(value) is type:
                # value is already type
                self._parser.add_argument("--" + name, type=value, help=help)
            elif type(value) is bool:
                self._parser.add_argument(
                    "--" + name, type=to_bool, default=value, help=help
                )
            else:
                # get the type from the value
                self._parser.add_argument(
                    "--" + name, type=type(value), default=value, help=help
                )

    def _supported_class_params(self) -> Dict[str, Any]:
        """
        Return a dictionary of parameter names to values/types for the class under test.

        These parameters will be exposed as command line arguments.
        """
        return {}

    def _parse_arguments(self, argv: List[Any]) -> None:
        """Parse all command line arguments, separating out the parameters for the class under test."""
        pp = pprint.PrettyPrinter()

        self._args = self._parser.parse_args(argv)
        print("command line arguments:")
        pp.pprint(vars(self._args))

        supported_class_params = self._supported_class_params()
        self._class_params = {
            k: v
            for k, v in vars(self._args).items()
            if k in supported_class_params and v is not None
        }
        print("\nclass params:")
        pp.pprint(self._class_params)
        print()

    @property
    def args(self) -> argparse.Namespace:
        """Return all parsed command line arguments."""
        return self._args

    @property
    def class_params(self) -> Dict[str, Any]:
        return self._class_params

    def git_revision(self) -> str:
        rev = "unknown"
        try:
            rev = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
                .decode("ascii")
                .strip()
            )
        except Exception:
            pass
        return rev

    def input_dataframe(
        self, spark: SparkSession, *paths: str
    ) -> Tuple[DataFrame, Union[str, List[str]], str]:
        """Return a Spark DataFrame for benchmarking, along with the input and label column names."""
        assert self._args is not None

        df = spark.read.parquet(*paths)

        # Label column label is "label" which is hardcoded by gen_data.py
        label_col = "label"
        features_col = [c for c in df.schema.names if c != label_col]
        features_col = features_col[0] if len(features_col) == 1 else features_col  # type: ignore

        selected_cols = []
        if self._args.num_gpus == 0:
            # convert to vector for CPU Spark, since it only supports vector feature types
            if label_col in df.schema.names:
                selected_cols.append(col(label_col))

            if any(["array" in t[1] for t in df.dtypes]):
                # Array Type
                selected_cols.append(
                    array_to_vector(col(features_col)).alias("features")  # type: ignore
                )
                features_col = "features"  # type: ignore
            elif not any(["vector" in t[1] for t in df.dtypes]):
                # multi-cols
                selected_cols.append(col("features"))
                df = (
                    VectorAssembler()
                    .setInputCols(features_col)
                    .setOutputCol("features")
                    .transform(df)
                    .drop(*features_col)
                )
                features_col = "features"  # type: ignore
            else:
                # Vector Type
                selected_cols = []  # just use original df

        train_df = df.select(*selected_cols) if len(selected_cols) > 0 else df
        return train_df, features_col, label_col

    def run(self) -> None:
        """Runs benchmarks for the class under test and"""
        assert self._args is not None

        run_results = []
        git_revision = self.git_revision()
        with WithSparkSession(
            self._args.spark_confs, shutdown=(not self._args.no_shutdown)
        ) as spark:
            for _ in range(self._args.num_runs):
                train_df, features_col, label_col = self.input_dataframe(
                    spark, *self._args.train_path
                )

                transform_df: Optional[DataFrame] = None
                if len(self._args.transform_path) > 0:
                    transform_df, _, _ = self.input_dataframe(
                        spark, *self._args.transform_path
                    )

                benchmark_results, benchmark_time = with_benchmark(
                    "benchmark time: ",
                    lambda: self.run_once(
                        spark, train_df, features_col, transform_df, label_col
                    ),
                )
                results = {
                    "datetime": datetime.now().isoformat(),
                    "git_hash": git_revision,
                    "benchmark_time": benchmark_time,
                }
                results.update(benchmark_results)
                run_results.append(results)

        # dictionary results
        print("-" * 100)
        print("Results (python dictionary):")
        for i, results in enumerate(run_results):
            print(f"{i}: {results}")

        # tabular results
        print("-" * 100)
        print("Results (pandas DataFrame):")
        report_pdf = pd.DataFrame(run_results)
        print(report_pdf)
        print("-" * 100)

        # save results to disk
        if self._args.report_path != "":
            report_pdf.to_csv(self._args.report_path, mode="a")

    @abstractmethod
    def run_once(
        self,
        spark: SparkSession,
        train_df: DataFrame,
        features_col: Union[str, List[str]],
        transform_df: Optional[DataFrame],
        label_col: Optional[str],
    ) -> Dict[str, Any]:
        """Run a single iteration of benchmarks for the class under test, returning a summary of
        timing and/or scoring results in dictionary form."""
        raise NotImplementedError
