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
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import array, col, sum
from pyspark.sql.types import DoubleType, StructField, StructType

from .base import BenchmarkBase
from .utils import inspect_default_params_from_func, with_benchmark


class BenchmarkKMeans(BenchmarkBase):
    def _supported_class_params(self) -> Dict[str, Any]:
        from pyspark.ml.clustering import KMeans

        params = inspect_default_params_from_func(
            KMeans.__init__,
            [
                "distanceMeasure",
                "featuresCol",
                "labelCol",
                "predictionCol",
                "probabilityCol",
                "rawPredictionCol",
                "weightCol",
                "leafCol",
            ],
        )
        params["seed"] = 1
        return params

    def _add_extra_arguments(self) -> None:
        self._parser.add_argument(
            "--no_cache",
            action="store_true",
            default=False,
            help="whether to enable dataframe repartition, cache and cout outside fit function",
        )

    def score(
        self,
        centers: np.ndarray,
        transformed_df: DataFrame,
        features_col: str,
        prediction_col: str,
    ) -> float:
        """Computes the sum of squared euclidean distances between vectors in the features_col
        of transformed_df and the vector in centers having the corresponding index value in prediction_col.
        This is the objective function being optimized by the kmeans algorithm.  It is also referred to as inertia.

        Parameters
        ----------
        centers
            KMeans computed center/centroid vectors.
        transformed_df
            KMeansModel transformed data.
        features_col
            Name of features column.
            Note: this column is assumed to be of pyspark sql 'array' type.
        prediction_col
            Name of prediction column (index of nearest centroid, as computed by KMeansModel.transform)

        Returns
        -------
        float
            The computed inertia score, per description above.

        """

        sc = transformed_df.rdd.context
        centers_bc = sc.broadcast(centers)

        def partition_score_udf(
            pdf_iter: Iterator[pd.DataFrame],
        ) -> Iterator[pd.DataFrame]:
            local_centers = centers_bc.value.astype(np.float64)
            partition_score = 0.0
            for pdf in pdf_iter:
                input_vecs = np.array(list(pdf[features_col]), dtype=np.float64)
                predictions = list(pdf[prediction_col])
                center_vecs = local_centers[predictions, :]
                partition_score += np.sum((input_vecs - center_vecs) ** 2)
            yield pd.DataFrame({"partition_score": [partition_score]})

        total_score = (
            transformed_df.mapInPandas(
                partition_score_udf,  # type: ignore
                StructType([StructField("partition_score", DoubleType(), True)]),
            )
            .agg(sum("partition_score").alias("total_score"))
            .toPandas()
        )  # type: ignore
        total_score = total_score["total_score"][0]  # type: ignore
        return total_score

    def run_once(
        self,
        spark: SparkSession,
        train_df: DataFrame,
        features_col: Union[str, List[str]],
        transform_df: Optional[DataFrame],
        label_name: Optional[str],
    ) -> Dict[str, Any]:
        num_gpus = self.args.num_gpus
        num_cpus = self.args.num_cpus
        no_cache = self.args.no_cache
        train_path = self.args.train_path

        func_start_time = time.time()

        first_col = train_df.dtypes[0][0]
        first_col_type = train_df.dtypes[0][1]
        is_array_col = True if "array" in first_col_type else False
        is_vector_col = True if "vector" in first_col_type else False
        is_single_col = is_array_col or is_vector_col
        if not is_single_col:
            input_cols = [c for c in train_df.schema.names]
        output_col = "cluster_idx"

        if num_gpus > 0:
            from spark_rapids_ml.clustering import KMeans

            assert num_cpus <= 0
            if not no_cache:

                def gpu_cache_df(df: DataFrame) -> DataFrame:
                    df = df.repartition(num_gpus).cache()
                    df.count()
                    return df

                train_df, prepare_time = with_benchmark(
                    "prepare dataset", lambda: gpu_cache_df(train_df)
                )

            params = self.class_params
            print(f"Passing {params} to KMeans")

            gpu_estimator = KMeans(
                num_workers=num_gpus, verbose=self.args.verbose, **params
            ).setPredictionCol(output_col)

            if is_single_col:
                gpu_estimator = gpu_estimator.setFeaturesCol(first_col)
            else:
                gpu_estimator = gpu_estimator.setFeaturesCols(input_cols)

            gpu_model, fit_time = with_benchmark(
                "gpu fit", lambda: gpu_estimator.fit(train_df)
            )

            transformed_df = gpu_model.setPredictionCol(output_col).transform(train_df)
            # count doesn't trigger compute so do something not too compute intensive
            _, transform_time = with_benchmark(
                "gpu transform", lambda: transformed_df.agg(sum(output_col)).collect()
            )

            total_time = round(time.time() - func_start_time, 2)
            print(f"gpu total time: {total_time} sec")

            df_for_scoring = transformed_df
            feature_col = first_col
            if not is_single_col:
                feature_col = "features_array"
                df_for_scoring = transformed_df.select(
                    array(*input_cols).alias("features_array"), output_col
                )
            elif is_vector_col:
                df_for_scoring = transformed_df.select(
                    vector_to_array(col(feature_col)).alias(feature_col), output_col
                )

            cluster_centers = gpu_model.cluster_centers_

        if num_cpus > 0:
            from pyspark.ml.clustering import KMeans as SparkKMeans

            assert num_gpus <= 0
            if is_array_col:
                vector_df = train_df.select(
                    array_to_vector(train_df[first_col]).alias(first_col)
                )
            elif not is_vector_col:
                vector_assembler = VectorAssembler(outputCol="features").setInputCols(
                    input_cols
                )
                vector_df = vector_assembler.transform(train_df).drop(*input_cols)
                first_col = "features"
            else:
                vector_df = train_df

            if not no_cache:

                def cpu_cache_df(df: DataFrame) -> DataFrame:
                    df = df.cache()
                    df.count()
                    return df

                vector_df, prepare_time = with_benchmark(
                    "prepare dataset", lambda: cpu_cache_df(vector_df)
                )

            params = self.class_params
            print(f"Passing {params} to KMeans")

            cpu_estimator = (
                SparkKMeans(**params)
                .setFeaturesCol(first_col)
                .setPredictionCol(output_col)
            )

            cpu_model, fit_time = with_benchmark(
                "cpu fit", lambda: cpu_estimator.fit(vector_df)
            )

            print(
                f"spark ML: iterations: {cpu_model.summary.numIter}, inertia: {cpu_model.summary.trainingCost}"
            )

            def cpu_transform(df: DataFrame) -> None:
                transformed_df = cpu_model.transform(df)
                transformed_df.agg(sum(output_col)).collect()
                return transformed_df

            transformed_df, transform_time = with_benchmark(
                "cpu transform", lambda: cpu_transform(vector_df)
            )

            total_time = time.time() - func_start_time
            print(f"cpu total took: {total_time} sec")

            feature_col = first_col
            df_for_scoring = transformed_df.select(
                vector_to_array(col(feature_col)).alias(feature_col), output_col
            )
            cluster_centers = cpu_model.clusterCenters()

        # either cpu or gpu mode is run, not both in same run
        score = self.score(
            np.array(cluster_centers), df_for_scoring, feature_col, output_col
        )
        # note: seems that inertia matches score at iterations-1
        print(f"score: {score}")

        result = {
            "fit_time": fit_time,
            "transform_time": transform_time,
            "total_time": total_time,
            "score": score,
            "k": self.args.k,
            "maxIter": self.args.maxIter,
            "tol": self.args.tol,
            "num_gpus": num_gpus,
            "num_cpus": num_cpus,
            "no_cache": no_cache,
            "train_path": train_path,
        }

        return result
