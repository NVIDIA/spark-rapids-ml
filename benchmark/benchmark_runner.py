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
import datetime
import sys
from abc import abstractmethod
from distutils.util import strtobool
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import array_to_vector
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, sum

from benchmark.utils import (
    WithSparkSession,
    inspect_default_params_from_func,
    to_bool,
    with_benchmark,
)


def _to_bool(literal: str) -> bool:
    return bool(strtobool(literal))


class BenchmarkBase:
    def __init__(self, argv: List[Any]) -> None:
        """Add common parameters"""
        self._parser = argparse.ArgumentParser()
        self._parser.add_argument(
            "--gpu_workers",
            type=int,
            default=0,
            help="if gpu_workers > 0, How many gpu workers to be used for Spark Rapids ML"
            "else do the benchmark on Spark ML library ",
        )
        self._parser.add_argument(
            "--train_path",
            action="append",
            default=[],
            required=True,
            help="Input parquet format data path used for training",
        )
        self._parser.add_argument(
            "--transform_path",
            action="append",
            default=[],
            help="Input parquet format data path used for transform",
        )
        self._parser.add_argument("--spark_confs", action="append", default=[])
        self._parser.add_argument(
            "--no_shutdown",
            action="store_true",
            help="do not stop spark session when finished",
        )
        self.args_: Optional[argparse.Namespace] = None

        self._add_extra_parameters()

        self._parse_arguments(argv)

    def _add_extra_parameters(self) -> None:

        self.supported_extra_params = self._supported_extra_params()

        for name, value in self.supported_extra_params.items():
            (value, help) = value if isinstance(value, tuple) else (value, None)
            help = "PySpark parameter" if help is None else help
            if value is None:
                raise RuntimeError("Must convert None value to the correct type")
            elif type(value) is type:
                # value is already type
                self._parser.add_argument("--" + name, type=value, help=help)
            elif type(value) is bool:
                self._parser.add_argument("--" + name, type=to_bool, help=help)
            else:
                # get the type from the value
                self._parser.add_argument("--" + name, type=type(value), help=help)

    def _supported_extra_params(self) -> Dict[str, Any]:
        """Function to inspect the specific function to get the parameters and values or types"""
        return {}

    def _parse_arguments(self, argv: List[Any]) -> None:
        self.args_ = self._parser.parse_args(argv)

        self.extra_params = {
            k: v
            for k, v in vars(self.args_).items()
            if k in self.supported_extra_params and v is not None
        }

    @property
    def args(self) -> Optional[argparse.Namespace]:
        return self.args_

    @abstractmethod
    def run(
        self,
        df: DataFrame,
        features_col: Union[str, List[str]],
        label_name: Optional[str],
    ) -> None:
        raise NotImplementedError()


class BenchmarkLinearRegression(BenchmarkBase):
    def _supported_extra_params(self) -> Dict[str, Any]:
        from pyspark.ml.regression import LinearRegression

        params = inspect_default_params_from_func(
            LinearRegression, ["featuresCol", "labelCol", "predictionCol", "weightCol"]
        )
        return params

    def run(
        self,
        df: DataFrame,
        features_col: Union[str, List[str]],
        label_name: Optional[str],
    ) -> None:

        assert label_name is not None
        assert self.args is not None

        params = self.extra_params
        print(f"Passing {params} to LinearRegression")

        if self.args.gpu_workers > 0:
            from spark_rapids_ml.regression import LinearRegression

            lr = LinearRegression(
                num_workers=self.args.gpu_workers, verbose=7, **params
            )
            benchmark_string = "Spark Rapids ML LinearRegression training:"
        else:
            from pyspark.ml.regression import LinearRegression

            lr = LinearRegression(**params)
            benchmark_string = "Spark ML LinearRegression training:"

        lr.setFeaturesCol(features_col)
        lr.setLabelCol(label_name)

        model = with_benchmark(benchmark_string, lambda: lr.fit(df))

        # placeholder try block till hasSummary is supported in gpu model
        try:
            if model.hasSummary:
                print(f"total iterations: {model.summary.totalIterations}")
                print(f"objective history: {model.summary.objectiveHistory}")
        except:
            print("model does not have hasSummary attribute")

        df_with_preds = model.transform(df)

        # model does not yet have col getters setters and uses default value for prediction col
        prediction_col = model.getOrDefault(model.predictionCol)

        # run a simple dummy computation to trigger transform. count is short
        # circuited due to pandas_udf used internally
        with_benchmark(
            "Spark ML LinearRegression transform:",
            lambda: df_with_preds.agg(sum(prediction_col)).collect(),
        )

        # compute prediction mse on training data
        from pyspark.ml.evaluation import RegressionEvaluator

        evaluator = (
            RegressionEvaluator()
            .setPredictionCol(prediction_col)
            .setLabelCol(label_name)
        )
        rmse = evaluator.evaluate(df_with_preds)

        coefficients = np.array(model.coefficients)
        coefs_l1 = np.sum(np.abs(coefficients))
        coefs_l2 = np.sum(coefficients**2)

        l2_penalty_factor = 0.5 * lr.getRegParam() * (1.0 - lr.getElasticNetParam())
        l1_penalty_factor = lr.getRegParam() * lr.getElasticNetParam()
        full_objective = (
            0.5 * (rmse**2)
            + coefs_l2 * l2_penalty_factor
            + coefs_l1 * l1_penalty_factor
        )

        # note: results for spark ML and spark rapids ml will currently match in all regularization
        # cases only if features and labels were standardized in the original dataset.  Otherwise,
        # they will match only if regParam = 0 or elastNetParam = 1.0 (aka Lasso)
        print(
            f"RMSE: {rmse}, coefs l1: {coefs_l1}, coefs l2^2: {coefs_l2}, "
            f"full_objective: {full_objective}, intercept: {model.intercept}"
        )


class BenchmarkRandomForestClassifier(BenchmarkBase):
    def _supported_extra_params(self) -> Dict[str, Any]:
        from pyspark.ml.classification import RandomForestClassifier

        # pyspark paramters
        params = inspect_default_params_from_func(
            RandomForestClassifier,
            [
                "featuresCol",
                "labelCol",
                "predictionCol",
                "probabilityCol",
                "rawPredictionCol",
                "weightCol",
                "leafCol",
            ],
        )
        # must replace the None to the correct type
        params["seed"] = int

        # cuML paramters
        params["n_streams"] = (
            int,
            "cuML parameters, number of parallel streams used for forest building",
        )
        params["max_batch_size"] = (
            int,
            "cuML parameters, maximum number of nodes that can be processed in a given batch",
        )
        return params

    def run(
        self,
        df: DataFrame,
        features_col: Union[str, List[str]],
        label_name: Optional[str],
    ) -> None:
        assert label_name is not None
        assert self.args is not None

        params = self.extra_params
        print(f"Passing {params} to RandomForestClassifier")

        if self.args.gpu_workers > 0:
            from spark_rapids_ml.classification import RandomForestClassifier

            rfc = RandomForestClassifier(
                num_workers=self.args.gpu_workers, verbose=7, **params
            )
            benchmark_string = "Spark Rapids ML RandomForestClassifier"
        else:
            from pyspark.ml.classification import RandomForestClassifier

            rfc = RandomForestClassifier(**params)
            benchmark_string = "Spark ML RandomForestClassifier"

        rfc.setFeaturesCol(features_col)
        rfc.setLabelCol(label_name)

        model = with_benchmark(f"{benchmark_string} training:", lambda: rfc.fit(df))

        df_with_preds = model.transform(df)

        # model does not yet have col getters setters and uses default value for prediction col
        prediction_col = model.getOrDefault(model.predictionCol)

        # run a simple dummy computation to trigger transform. count is short
        # circuited due to pandas_udf used internally
        with_benchmark(
            f"{benchmark_string} transform:",
            lambda: df_with_preds.agg(sum(prediction_col)).collect(),
        )

        df_with_preds = df_with_preds.select(
            col(prediction_col).cast("double").alias(prediction_col), label_name
        )

        if model.numClasses == 2:
            # binary classification
            evaluator: Union[
                BinaryClassificationEvaluator, MulticlassClassificationEvaluator
            ] = (
                BinaryClassificationEvaluator()
                .setRawPredictionCol(prediction_col)
                .setLabelCol(label_name)
            )
        else:
            evaluator = (
                MulticlassClassificationEvaluator()
                .setPredictionCol(prediction_col)
                .setLabelCol(label_name)
            )

        accuracy = evaluator.evaluate(df_with_preds)

        print(f"{benchmark_string} accuracy: {accuracy}")


class BenchmarkRandomForestRegressor(BenchmarkBase):
    def _supported_extra_params(self) -> Dict[str, Any]:
        from pyspark.ml.regression import RandomForestRegressor

        params = inspect_default_params_from_func(
            RandomForestRegressor,
            ["featuresCol", "labelCol", "predictionCol", "weightCol", "leafCol"],
        )
        # must replace the None to the correct type
        params["seed"] = int

        # cuML paramters
        params["n_streams"] = (
            int,
            "cuML parameters, number of parallel streams used for forest building",
        )
        params["max_batch_size"] = (
            int,
            "cuML parameters, maximum number of nodes that can be processed in a given batch",
        )
        return params

    def run(
        self,
        df: DataFrame,
        features_col: Union[str, List[str]],
        label_name: Optional[str],
    ) -> None:
        assert label_name is not None
        assert self.args is not None

        params = self.extra_params
        print(f"Passing {params} to RandomForestRegressor")

        if self.args.gpu_workers > 0:
            from spark_rapids_ml.regression import RandomForestRegressor

            rf = RandomForestRegressor(
                num_workers=self.args.gpu_workers, verbose=7, **params
            )
            benchmark_string = "Spark Rapids ML RandomForestRegressor"
        else:
            from pyspark.ml.regression import RandomForestRegressor

            rf = RandomForestRegressor(**params)
            benchmark_string = "Spark ML RandomForestRegressor"

        rf.setFeaturesCol(features_col)
        rf.setLabelCol(label_name)

        model = with_benchmark(f"{benchmark_string} training:", lambda: rf.fit(df))

        df_with_preds = model.transform(df)

        # model does not yet have col getters setters and uses default value for prediction col
        prediction_col = model.getOrDefault(model.predictionCol)

        # run a simple dummy computation to trigger transform. count is short
        # circuited due to pandas_udf used internally
        with_benchmark(
            f"{benchmark_string} transform:",
            lambda: df_with_preds.agg(sum(prediction_col)).collect(),
        )

        # compute prediction mse on training data
        from pyspark.ml.evaluation import RegressionEvaluator

        evaluator = (
            RegressionEvaluator()
            .setPredictionCol(prediction_col)
            .setLabelCol(label_name)
        )
        rmse = evaluator.evaluate(df_with_preds)

        print(f"{benchmark_string} RMSE: {rmse}")


class BenchmarkRunner:
    def __init__(self) -> None:
        registered_algorithms = {
            "linear_regression": BenchmarkLinearRegression,
            "random_forest_classifier": BenchmarkRandomForestClassifier,
            "random_forest_regressor": BenchmarkRandomForestRegressor,
        }

        parser = argparse.ArgumentParser(
            description="Benchmark Spark CUML",
            usage="""benchmark_runner.py <algorithm> [<args>]

        Supported algorithms are:
           linear_regression             Benchmark linear regression
           random_forest_classifier      Benchmark RandomForestClassifier
           random_forest_regressor       Benchmark RandomForestRegressor
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
        args = self._runner.args

        assert args is not None

        with WithSparkSession(
            args.spark_confs, shutdown=(not args.no_shutdown)
        ) as spark:

            df = spark.read.parquet(*args.train_path)

            # Label column label is "label" which is hardcoded by gen_data.py
            label_name = "label"
            features_col = [c for c in df.schema.names if c != label_name]

            selected_cols = []
            if args.gpu_workers == 0:
                # Spark ml only supports vector feature type.
                if label_name in df.schema.names:
                    selected_cols.append(col(label_name))

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

            with_benchmark(
                "Total running time: ",
                lambda: self._runner.run(train_df, features_col, label_name),
            )


if __name__ == "__main__":
    """
    There're two ways to do the benchmark.

    1.
        python benchmark_runner.py [linear_regression] \
            --gpu_workers=2 \
            --train_path=xxx \
            --spark_confs="spark.master=local[12]" \

    2.
        spark-submit --master local[12] benchmark_runner.py -gpu_workers=2 --train_path=xxx
    """

    print(f"invoked time: {datetime.datetime.now()}")

    BenchmarkRunner().run()
