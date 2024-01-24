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
from typing import Any, Dict, List, Optional, Union

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, sum

from .base import BenchmarkBase
from .utils import inspect_default_params_from_func, with_benchmark


class BenchmarkRandomForestClassifier(BenchmarkBase):
    def _supported_class_params(self) -> Dict[str, Any]:
        from pyspark.ml.classification import RandomForestClassifier

        # pyspark paramters
        params = inspect_default_params_from_func(
            RandomForestClassifier.__init__,
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

    def run_once(
        self,
        spark: SparkSession,
        train_df: DataFrame,
        features_col: Union[str, List[str]],
        transform_df: Optional[DataFrame],
        label_col: Optional[str],
    ) -> Dict[str, Any]:
        assert label_col is not None
        assert self.args is not None

        params = self.class_params
        print(f"Passing {params} to RandomForestClassifier")

        if self.args.num_gpus > 0:
            from spark_rapids_ml.classification import RandomForestClassifier

            rfc = RandomForestClassifier(
                num_workers=self.args.num_gpus, verbose=self.args.verbose, **params
            )
            benchmark_string = "Spark Rapids ML RandomForestClassifier"
        else:
            from pyspark.ml.classification import (
                RandomForestClassifier as SparkRandomForestClassifier,
            )

            rfc = SparkRandomForestClassifier(**params)  # type: ignore[assignment]
            benchmark_string = "Spark ML RandomForestClassifier"

        rfc.setFeaturesCol(features_col)
        rfc.setLabelCol(label_col)

        model, training_time = with_benchmark(
            f"{benchmark_string} training:", lambda: rfc.fit(train_df)
        )

        eval_df = train_df if transform_df is None else transform_df

        df_with_preds = model.transform(eval_df)

        # model does not yet have col getters setters and uses default value for prediction col
        prediction_col = model.getOrDefault(model.predictionCol)
        probability_col = model.getOrDefault(model.probabilityCol)

        # run a simple dummy computation to trigger transform. count is short
        # circuited due to pandas_udf used internally
        _, transform_time = with_benchmark(
            f"{benchmark_string} transform:",
            lambda: df_with_preds.agg(sum(prediction_col)).collect(),
        )

        if model.numClasses == 2:
            # binary classification
            evaluator: Union[
                BinaryClassificationEvaluator, MulticlassClassificationEvaluator
            ] = (
                BinaryClassificationEvaluator()
                .setRawPredictionCol(probability_col)
                .setLabelCol(label_col)
            )
        else:
            evaluator = (
                MulticlassClassificationEvaluator()
                .setPredictionCol(prediction_col)
                .setLabelCol(label_col)
            )

        accuracy = evaluator.evaluate(df_with_preds)

        print(f"{benchmark_string} accuracy: {accuracy}")

        results = {
            "training_time": training_time,
            "transform_time": transform_time,
            "accuracy": accuracy,
        }

        return results


class BenchmarkRandomForestRegressor(BenchmarkBase):
    def _supported_class_params(self) -> Dict[str, Any]:
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

    def run_once(
        self,
        spark: SparkSession,
        train_df: DataFrame,
        features_col: Union[str, List[str]],
        transform_df: Optional[DataFrame],
        label_col: Optional[str],
    ) -> Dict[str, Any]:
        assert label_col is not None

        params = self.class_params
        print(f"Passing {params} to RandomForestRegressor")

        if self.args.num_gpus > 0:
            from spark_rapids_ml.regression import RandomForestRegressor

            rf = RandomForestRegressor(
                num_workers=self.args.num_gpus, verbose=self.args.verbose, **params
            )
            benchmark_string = "Spark Rapids ML RandomForestRegressor"
        else:
            from pyspark.ml.regression import (
                RandomForestRegressor as SparkRandomForestRegressor,
            )

            rf = SparkRandomForestRegressor(**params)  # type: ignore[assignment]
            benchmark_string = "Spark ML RandomForestRegressor"

        rf.setFeaturesCol(features_col)
        rf.setLabelCol(label_col)

        model, training_time = with_benchmark(
            f"{benchmark_string} training:", lambda: rf.fit(train_df)
        )

        eval_df = train_df if transform_df is None else transform_df

        df_with_preds = model.transform(eval_df)

        # model does not yet have col getters setters and uses default value for prediction col
        prediction_col = model.getOrDefault(model.predictionCol)

        # run a simple dummy computation to trigger transform. count is short
        # circuited due to pandas_udf used internally
        _, transform_time = with_benchmark(
            f"{benchmark_string} transform:",
            lambda: df_with_preds.agg(sum(prediction_col)).collect(),
        )

        # compute prediction mse on training data
        from pyspark.ml.evaluation import RegressionEvaluator

        evaluator = (
            RegressionEvaluator()
            .setPredictionCol(prediction_col)
            .setLabelCol(label_col)
        )
        rmse = evaluator.evaluate(df_with_preds)

        print(f"{benchmark_string} RMSE: {rmse}")

        results = {
            "training_time": training_time,
            "transform_time": transform_time,
            "rmse": rmse,
        }
        return results
