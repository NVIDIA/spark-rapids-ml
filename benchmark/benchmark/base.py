import argparse
import inspect
import pprint
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import array_to_vector
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col

from spark_rapids_ml.core import _CumlEstimator

from .utils import WithSparkSession, to_bool, with_benchmark


class BenchmarkBase:
    """Based class for benchmarking.

    This class handles command line argument parsing and execution of the benchmark.

    Attributes
    ----------
    test_cls : Type[_CumlEstimator]
        Class under test, which should be a subclass of _CumlEstimator.

    unsupported_params: List[str]
        List of Spark ML Params and/or cuML parameters which should not be exposed as command line arguments.
    """

    test_cls: Type[_CumlEstimator]
    unsupported_params: List[str] = []

    _parser: argparse.ArgumentParser
    _args: Optional[argparse.Namespace] = None
    _spark_args: List[str] = []
    _cuml_args: List[str] = []

    def __init__(self, argv: List[Any]) -> None:
        """Parses command line arguments for the class under test."""
        print("=" * 100)
        print(f"Benchmarking: {self.test_cls}")

        self._parser = argparse.ArgumentParser()
        self._parser.add_argument(
            "--num_gpus",
            type=int,
            default=1,
            help="number of GPUs to use. If num_gpus > 0, spark-rapids-ml will run with the number of dataset partitions equal to num_gpus.",
        )
        self._parser.add_argument(
            "--num_cpus",
            type=int,
            default=6,
            help="number of CPUs to use. If num_cpus > 0, spark will run and with the number of dataset partitions to num_cpus.",
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

        # add command line arguments for class under test
        self._add_spark_arguments()
        self._add_cuml_arguments()

        self.add_arguments()
        self._parse_arguments(argv)

    def _add_spark_arguments(self) -> None:
        """
        Adds parser arguments for the Spark ML Params of the class under test.

        To exclude an ML Param, just add it to `unsupported_params`.
        """
        # create a default instance to extract params
        instance = self.test_cls()

        for param in instance.params:
            if param.name not in self.unsupported_params:
                name = param.name
                default = (
                    instance.getOrDefault(name) if instance.hasDefault(name) else None
                )
                value_type = inspect.signature(param.typeConverter).return_annotation
                inspect.signature(param.typeConverter).return_annotation
                help = param.doc
                self._spark_args.append(name)
                self._parser.add_argument(
                    "--" + name, type=value_type, default=default, help=help
                )

    def _add_cuml_arguments(self) -> None:
        """
        Adds parser arguments for the unmapped cuML parameters of the class under test.

        cuML parameters that are mapped to Spark ML Params must be set via the Spark Param name.
        To exclude a parameter, just add it to `unsupported_params`.
        """
        # create a default instance to extract params
        instance = self.test_cls()
        param_mapping = instance._param_mapping().values()
        param_excludes = instance._param_excludes()

        for k, v in instance.cuml_params.items():
            if (
                k not in self.unsupported_params
                and k not in param_mapping
                and k not in param_excludes
            ):
                name = k
                default = v
                value_type = type(v)
                help = "cuML parameter"
                self._cuml_args.append(name)
                self._parser.add_argument(
                    "--" + name, type=value_type, default=default, help=help
                )

    def _parse_arguments(self, argv: List[Any]) -> None:
        """Parse command line arguments while separating out the parameters intended for the class under test."""
        pp = pprint.PrettyPrinter()

        self._args = self._parser.parse_args(argv)
        print("\ncommand line arguments:")
        pp.pprint(vars(self._args))

        self._spark_params = {
            k: v
            for k, v in vars(self._args).items()
            if k in self._spark_args and v is not None
        }
        print("\nspark_params:")
        pp.pprint(self._spark_params)

        self._cuml_params = {
            k: v
            for k, v in vars(self._args).items()
            if k in self._cuml_args and v is not None
        }
        print("\ncuml_params:")
        pp.pprint(self._cuml_params)
        print()

    @property
    def args(self) -> Optional[argparse.Namespace]:
        """Return all parsed command line arguments."""
        return self._args

    @property
    def spark_params(self) -> Dict[str, Any]:
        """Return parsed command line arguments intended for Spark ML Params of the class under test."""
        return self._spark_params.copy()

    @property
    def cuml_params(self) -> Dict[str, Any]:
        """Return parsed command line arguments intended for cuML parameters of the class under test."""
        return self._cuml_params.copy()

    @property
    def spark_cuml_params(self) -> Dict[str, Any]:
        """Return all parsed command line arguments indended for the class under test."""
        params = self._spark_params.copy()
        params.update(self._cuml_params)
        return params

    def add_arguments(self) -> None:
        """Add additional command line parser arguments."""
        pass

    def train_df(
        self, spark: SparkSession
    ) -> Tuple[DataFrame, Union[str, List[str]], str]:
        """Return a Spark DataFrame for benchmarking, along with the input and label column names."""
        assert self._args is not None

        df = spark.read.parquet(*self._args.train_path)

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
                    array_to_vector(col(features_col[0])).alias("features")
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
        with WithSparkSession(
            self._args.spark_confs, shutdown=(not self._args.no_shutdown)
        ) as spark:
            for _ in range(self._args.num_runs):
                df, features_col, label_col = self.train_df(spark)
                results, benchmark_time = with_benchmark(
                    "benchmark time: ",
                    lambda: self.run_once(spark, df, features_col, label_col),
                )
                results["benchmark_time"] = benchmark_time
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
            report_pdf.to_csv(self._args.report_path, mode="a")

    @abstractmethod
    def run_once(
        self,
        spark: SparkSession,
        df: DataFrame,
        features_col: Union[str, List[str]],
        label_col: Optional[str],
    ) -> Dict[str, Any]:
        """Run a single iteration of benchmarks for the class under test, returning a summary of
        timing and/or scoring results in dictionary form."""
        raise NotImplementedError
