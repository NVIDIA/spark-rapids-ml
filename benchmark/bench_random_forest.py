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
from typing import List, Optional, Union

from pyspark.ml.classification import RandomForestClassifier as SparkRandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.regression import RandomForestRegressor as SparkRandomForestRegressor
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, sum
from spark_rapids_ml.classification import RandomForestClassifier
from spark_rapids_ml.regression import RandomForestRegressor

from benchmark.base import BenchmarkBase
from benchmark.utils import with_benchmark

class BenchmarkRandomForestClassifier(BenchmarkBase):
    test_cls = RandomForestClassifier
    unsupported_params = test_cls._param_excludes() + [
        "featuresCol",
        "labelCol",
        "predictionCol",
        "probabilityCol",
        "rawPredictionCol",
        "weightCol",
        "leafCol",
    ]

    def run_once(
        self,
        spark: SparkSession,
        df: DataFrame,
        features_col: Union[str, List[str]],
        label_col: Optional[str],
    ) -> None:
        assert label_col is not None
        assert self.args is not None

        results = {}
        if self.args.num_gpus > 0:
            params = self.spark_cuml_params
            print(f"Passing {params} to RandomForestClassifier")

            rfc = RandomForestClassifier(
                num_workers=self.args.num_gpus, **params
            )
            benchmark_string = "Spark Rapids ML RandomForestClassifier"
        else:
            params = self.spark_params
            print(f"Passing {params} to SparkRandomForestClassifier")

            rfc = SparkRandomForestClassifier(**params)
            benchmark_string = "Spark ML RandomForestClassifier"

        rfc.setFeaturesCol(features_col)
        rfc.setLabelCol(label_col)

        model, training_time = with_benchmark(f"{benchmark_string} training:", lambda: rfc.fit(df))

        df_with_preds = model.transform(df)

        # model does not yet have col getters setters and uses default value for prediction col
        prediction_col = model.getOrDefault(model.predictionCol)

        # run a simple dummy computation to trigger transform. count is short
        # circuited due to pandas_udf used internally
        _, transform_time = with_benchmark(
            f"{benchmark_string} transform:",
            lambda: df_with_preds.agg(sum(prediction_col)).collect(),
        )

        df_with_preds = df_with_preds.select(
            col(prediction_col).cast("double").alias(prediction_col), label_col
        )

        if model.numClasses == 2:
            # binary classification
            evaluator: Union[
                BinaryClassificationEvaluator, MulticlassClassificationEvaluator
            ] = (
                BinaryClassificationEvaluator()
                .setRawPredictionCol(prediction_col)
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
    test_cls = RandomForestRegressor
    unsupported_params = [
        "featuresCol",
        "labelCol",
        "predictionCol",
        "weightCol",
        "leafCol",
    ]

    def run_once(
        self,
        spark: SparkSession,
        df: DataFrame,
        features_col: Union[str, List[str]],
        label_col: Optional[str],
    ) -> None:
        assert label_col is not None
        assert self.args is not None

        if self.args.num_gpus > 0:
            params = self.spark_cuml_params
            print(f"Passing {params} to RandomForestRegressor")
            rf = RandomForestRegressor(
                num_workers=self.args.num_gpus, **params
            )
            benchmark_string = "Spark Rapids ML RandomForestRegressor"
        else:
            params = self.spark_params
            print(f"Passing {params} to SparkRandomForestRegressor")

            rf = SparkRandomForestRegressor(**params)
            benchmark_string = "Spark ML RandomForestRegressor"

        rf.setFeaturesCol(features_col)
        rf.setLabelCol(label_col)

        model, training_time = with_benchmark(f"{benchmark_string} training:", lambda: rf.fit(df))

        df_with_preds = model.transform(df)

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