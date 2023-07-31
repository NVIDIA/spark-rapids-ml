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
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import sum

from .base import BenchmarkBase
from .utils import inspect_default_params_from_func, with_benchmark


class BenchmarkLogisticRegression(BenchmarkBase):
    def _supported_class_params(self) -> Dict[str, Any]:
        from pyspark.ml.classification import LogisticRegression 

        params = inspect_default_params_from_func(
            LogisticRegression.__init__,
            ["featuresCol", "labelCol", "predictionCol", "weightCol"]
        )
        return params

    def run_once(
        self,
        spark: SparkSession,
        train_df: DataFrame,
        features_col: Union[str, List[str]],
        transform_df: Optional[DataFrame],
        label_name: Optional[str],
    ) -> Dict[str, Any]:
        assert label_name is not None

        params = self.class_params
        print(f"Passing {params} to LogisticRegression")

        if self.args.num_gpus > 0:
            from spark_rapids_ml.classification import LogisticRegression

            lr = LogisticRegression(
                num_workers=self.args.num_gpus, **params
            )
            benchmark_string = "Spark Rapids ML LogisticRegression training"
        else:
            from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression

            lr = SparkLogisticRegression(**params)  # type: ignore[assignment]
            benchmark_string = "Spark ML LogisticRegression training"

        lr.setFeaturesCol(features_col)
        lr.setLabelCol(label_name)

        model, fit_time = with_benchmark(benchmark_string, lambda: lr.fit(train_df))

        # placeholder try block till hasSummary is supported in gpu model
        try:
            if model.hasSummary:
                print(f"total iterations: {model.summary.totalIterations}")
                print(f"objective history: {model.summary.objectiveHistory}")
        except:
            print("model does not have hasSummary attribute")

        eval_df = train_df if transform_df is None else transform_df

        df_with_preds = model.transform(eval_df)

        # model does not yet have col getters setters and uses default value for prediction col
        prediction_col = model.getOrDefault(model.predictionCol)

        # run a simple dummy computation to trigger transform. count is short
        # circuited due to pandas_udf used internally
        _, transform_time = with_benchmark(
            "Spark ML LogisticRegression transform",
            lambda: df_with_preds.agg(sum(prediction_col)).collect(),
        )

        from pyspark.ml.evaluation import (
            BinaryClassificationEvaluator,
            MulticlassClassificationEvaluator,
        )
        # TODO: support multiple classes
        # binary classification
        evaluator: Union[
            BinaryClassificationEvaluator, MulticlassClassificationEvaluator
        ] = (
            BinaryClassificationEvaluator()
            .setRawPredictionCol(prediction_col)
            .setLabelCol(label_name)
        )

        accuracy = evaluator.evaluate(df_with_preds)

        print(f"{benchmark_string} accuracy: {accuracy}")

        results = {
            "fit_time": fit_time,
            "transform_time": transform_time,
            "accuracy": accuracy,
            "num_gpus": self.args.num_gpus,
            "num_cpus": self.args.num_cpus,
            "train_path": self.args.train_path,
            "maxIter": params["maxIter"],
            "tol": params["tol"],
            "regParam": params["regParam"],
            "standardization": params["standardization"],
        }

        return results