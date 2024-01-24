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

import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import sum

from .base import BenchmarkBase
from .utils import inspect_default_params_from_func, with_benchmark


class BenchmarkLinearRegression(BenchmarkBase):
    def _supported_class_params(self) -> Dict[str, Any]:
        from pyspark.ml.regression import LinearRegression

        params = inspect_default_params_from_func(
            LinearRegression.__init__,
            ["featuresCol", "labelCol", "predictionCol", "weightCol"],
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
        print(f"Passing {params} to LinearRegression")

        if self.args.num_gpus > 0:
            from spark_rapids_ml.regression import LinearRegression

            lr = LinearRegression(
                num_workers=self.args.num_gpus, verbose=self.args.verbose, **params
            )
            benchmark_string = "Spark Rapids ML LinearRegression training"
        else:
            from pyspark.ml.regression import LinearRegression as SparkLinearRegression

            lr = SparkLinearRegression(**params)  # type: ignore[assignment]
            benchmark_string = "Spark ML LinearRegression training"

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
            "Spark ML LinearRegression transform",
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

        results = {
            "fit_time": fit_time,
            "transform_time": transform_time,
            "RMSE": rmse,
            "coefs_l1": coefs_l1,
            "coefs_l2": coefs_l2,
            "full_objective": full_objective,
            "intercept": model.intercept,
        }
        return results
