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
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, sum
from pyspark.sql.types import DoubleType, StructField, StructType

from .base import BenchmarkBase
from .utils import inspect_default_params_from_func, with_benchmark


class BenchmarkPCA(BenchmarkBase):
    def _supported_class_params(self) -> Dict[str, Any]:
        from pyspark.ml.feature import PCA

        params = inspect_default_params_from_func(
            PCA.__init__,
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
        params["k"] = int
        return params

    def _add_extra_arguments(self) -> None:
        self._parser.add_argument(
            "--no_cache",
            action="store_true",
            default=False,
            help="whether to enable dataframe repartition, cache and cout outside fit function",
        )

    def score(
        self, pc_vectors: np.ndarray, transformed_df: DataFrame, transformed_col: str
    ) -> Tuple[float, float]:
        """Computes a measure of orthonormality of the pc_vectors: maximum absolute deviation from 1 from all norms
        and absolute deviation from 0 of all dot products between the pc_vectors.  This should be very small.

        Also computes the sum of squares of the transformed_col vectors.  The larger the better.

        PCA projection should have been performed on mean removed input for the second metric to be relevant.

        Parameters
        ----------
        pc_vectors
            principal component vectors.
            pc_vectors.shape assumed to be (dim, k).
        transformed_df
            PCAModel transformed data.
        transformed_col
            Name of column with the PCA transformed data.
            Note: This column is expected to be of pyspark sql 'array' type.

        Returns
        -------
        Tuple[float, float]
            The components of the returned tuple are respectively the orthonormality score and
            the sum of squares of the transformed vectors.

        """

        pc_vectors = np.array(pc_vectors, dtype=np.float64)
        k = pc_vectors.shape[1]
        pc_vectors_self_prod = np.matmul(np.transpose(pc_vectors), pc_vectors)
        orthonormality_score = np.max(np.abs(np.eye(k) - pc_vectors_self_prod))

        def partition_score_udf(pdf_iter: Iterator[pd.DataFrame]) -> Iterator[float]:
            partition_score = 0.0
            for pdf in pdf_iter:
                transformed_vecs = np.array(
                    list(pdf[transformed_col]), dtype=np.float64
                )
                partition_score += np.sum(transformed_vecs**2)
            yield pd.DataFrame({"partition_score": [partition_score]})

        total_score = (
            transformed_df.mapInPandas(
                partition_score_udf,  # type: ignore
                StructType([StructField("partition_score", DoubleType(), True)]),
            )
            .agg(sum("partition_score").alias("total_score"))
            .toPandas()
        )
        total_score = total_score["total_score"][0]  # type: ignore
        return orthonormality_score, total_score

    def run_once(
        self,
        spark: SparkSession,
        train_df: DataFrame,
        features_col: Union[str, List[str]],
        transform_df: Optional[DataFrame],
        label_name: Optional[str],
    ) -> Dict[str, Any]:
        n_components = self.args.k
        num_gpus = self.args.num_gpus
        num_cpus = self.args.num_cpus
        no_cache = self.args.no_cache

        func_start_time = time.time()

        first_col = train_df.dtypes[0][0]
        first_col_type = train_df.dtypes[0][1]
        is_array_col = True if "array" in first_col_type else False
        is_vector_col = True if "vector" in first_col_type else False
        is_single_col = is_array_col or is_vector_col

        if not is_single_col:
            input_cols = [c for c in train_df.schema.names]

        if num_gpus > 0:
            from spark_rapids_ml.feature import PCA

            assert num_cpus <= 0
            if not no_cache:

                def gpu_cache_df(df: DataFrame) -> DataFrame:
                    df = df.repartition(num_gpus).cache()
                    df.count()
                    return df

                train_df, prepare_time = with_benchmark(
                    "prepare session and dataset", lambda: gpu_cache_df(train_df)
                )

            params = self.class_params
            print(f"Passing {params} to PCA")

            output_col = "pca_features"
            gpu_pca = (
                PCA(num_workers=num_gpus, verbose=self.args.verbose, **params)
                .setInputCol(features_col)
                .setOutputCol(output_col)
            )

            gpu_model, fit_time = with_benchmark(
                "gpu fit", lambda: gpu_pca.fit(train_df)
            )

            def gpu_transform(df: DataFrame) -> DataFrame:
                transformed_df = gpu_model.transform(df)
                transformed_df.select((col(output_col)[0]).alias("zero")).agg(
                    sum("zero")
                ).collect()
                return transformed_df

            transformed_df, transform_time = with_benchmark(
                "gpu transform", lambda: gpu_transform(train_df)
            )

            total_time = round(time.time() - func_start_time, 2)
            print(f"gpu total took: {total_time} sec")

            # spark ml does not remove the mean in the transformed features, so do that here
            # needed for scoring
            feature_col = output_col
            df_for_scoring = transformed_df.select(
                array_to_vector(col(output_col)).alias(output_col + "_vec")
            )
            standard_scaler = (
                StandardScaler()
                .setWithStd(False)
                .setWithMean(True)
                .setInputCol(output_col + "_vec")
                .setOutputCol(output_col + "_mean_removed")
            )
            scaler_model = standard_scaler.fit(df_for_scoring)
            df_for_scoring = (
                scaler_model.transform(df_for_scoring)
                .drop(output_col + "_vec")
                .select(
                    vector_to_array(col(output_col + "_mean_removed")).alias(
                        feature_col
                    )
                )
            )

            pc_for_scoring = gpu_model.pc.toArray()

        if num_cpus > 0:
            from pyspark.ml.feature import PCA as SparkPCA

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

            output_col = "pca_features"

            params = self.class_params
            print(f"Passing {params} to SparkPCA")

            cpu_pca = SparkPCA(**params).setInputCol(first_col).setOutputCol(output_col)

            cpu_model, fit_time = with_benchmark(
                "cpu fit", lambda: cpu_pca.fit(vector_df)
            )

            def cpu_transform(df: DataFrame) -> None:
                transformed_df = cpu_model.transform(df)
                transformed_df.select(
                    (vector_to_array(col(output_col))[0]).alias("zero")
                ).agg(sum("zero")).collect()
                return transformed_df

            transformed_df, transform_time = with_benchmark(
                "cpu transform", lambda: cpu_transform(vector_df)
            )

            total_time = round(time.time() - func_start_time, 2)
            print(f"cpu total took: {total_time} sec")

            # spark ml does not remove the mean in the transformed features, so do that here
            # needed for scoring
            standard_scaler = (
                StandardScaler()
                .setWithStd(False)
                .setWithMean(True)
                .setInputCol(output_col)
                .setOutputCol(output_col + "_mean_removed")
            )

            scaler_model = standard_scaler.fit(transformed_df)
            transformed_df = scaler_model.transform(transformed_df).drop(output_col)

            feature_col = output_col + "_mean_removed"
            pc_for_scoring = cpu_model.pc.toArray()
            df_for_scoring = transformed_df.select(
                vector_to_array(col(feature_col)).alias(feature_col)
            )

        orthonormality, variance = self.score(
            pc_for_scoring, df_for_scoring, feature_col
        )
        print(f"orthonormality score: {orthonormality}, variance score {variance}")

        report_dict = {
            "fit": fit_time,
            "transform": transform_time,
            "total": total_time,
            "orthonormality": orthonormality,
            "variance": variance,
            "k": self.args.k,
            "num_gpus": self.args.num_gpus,
            "num_cpus": self.args.num_cpus,
            "no_cache": self.args.no_cache,
            "train_path": self.args.train_path,
        }

        if not no_cache:
            report_dict["prepare"] = prepare_time

        return report_dict
