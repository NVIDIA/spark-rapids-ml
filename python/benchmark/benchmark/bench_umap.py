#
# Copyright (c) 2024, NVIDIA CORPORATION.
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
import pprint
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pandas import DataFrame as PandasDataFrame
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import array, col, sum

from benchmark.base import BenchmarkBase
from benchmark.utils import inspect_default_params_from_func, with_benchmark


class BenchmarkUMAP(BenchmarkBase):
    def _supported_class_params(self) -> Dict[str, Any]:
        """Note: only needed for Spark PCA on CPU."""
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

    def _parse_arguments(self, argv: List[Any]) -> None:
        """Override to set class params based on cpu or gpu run (umap or pca)"""
        pp = pprint.PrettyPrinter()

        self._args = self._parser.parse_args(argv)
        print("command line arguments:")
        pp.pprint(vars(self._args))

        if self._args.num_cpus > 0:
            supported_class_params = self._supported_class_params()
        else:
            supported_class_params = {}
        self._class_params = {
            k: v
            for k, v in vars(self._args).items()
            if k in supported_class_params and v is not None
        }
        print("\nclass params:")
        pp.pprint(self._class_params)
        print()

    def _add_extra_arguments(self) -> None:
        self._parser.add_argument(
            "--no_cache",
            action="store_true",
            default=False,
            help="whether to enable dataframe repartition, cache and cout outside fit function",
        )

    def score(
        self, transformed_df: DataFrame, data_col: str, transformed_col: str
    ) -> float:
        """Computes the trustworthiness score, a measure of the extent to which the local structure
        of the dataset is retained in the embedding of the UMAP model (or the projection in the case of PCA).
        Score is in the range of [0, 1].

        Parameters
        ----------
        transformed_df
            Model transformed data.
        data_col
            Name of column with the input data.
            Note: This column is expected to be of pyspark sql 'array' type.
        transformed_col
            Name of column with the transformed data.
            Note: This column is expected to be of pyspark sql 'array' type.

        Returns
        -------
        float
            The trustworthiness score of the transformed data.

        """
        from cuml.metrics import trustworthiness

        pdf: PandasDataFrame = transformed_df.toPandas()
        embedding = np.array(pdf[transformed_col].to_list())
        input = np.array(pdf[data_col].to_list()).astype(np.float32)
        score = trustworthiness(input, embedding, n_neighbors=15)

        return score

    def run_once(
        self,
        spark: SparkSession,
        train_df: DataFrame,
        features_col: Union[str, List[str]],
        transform_df: Optional[DataFrame],
        label_name: Optional[str],
    ) -> Dict[str, Any]:
        """
        This function evaluates the runtimes of Spark Rapids ML UMAP and Spark PCA and returns the
        trustworthiness score of the model projections. The primary purpose is to help understand GPU behavior
        and performance.
        """
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
            from spark_rapids_ml.umap import UMAP, UMAPModel

            assert num_cpus <= 0
            if not no_cache:

                def gpu_cache_df(df: DataFrame) -> DataFrame:
                    df = df.repartition(num_gpus).cache()
                    df.count()
                    return df

                train_df, prepare_time = with_benchmark(
                    "prepare dataset", lambda: gpu_cache_df(train_df)
                )

            gpu_estimator = UMAP(
                num_workers=num_gpus,
                verbose=self.args.verbose,
            )

            if is_single_col:
                gpu_estimator = gpu_estimator.setFeaturesCol(first_col)
            else:
                gpu_estimator = gpu_estimator.setFeaturesCols(input_cols)

            gpu_model, fit_time = with_benchmark(
                "gpu fit", lambda: gpu_estimator.fit(train_df)
            )

            output_col = "embedding"
            transformed_df = gpu_model.setOutputCol(output_col).transform(train_df)
            _, transform_time = with_benchmark(
                "gpu transform", lambda: transformed_df.foreach(lambda _: None)
            )

            total_time = round(time.time() - func_start_time, 2)
            print(f"gpu total took: {total_time} sec")

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

        if num_cpus > 0:
            from pyspark.ml.feature import PCA as SparkPCA

            assert num_gpus <= 0

            if is_array_col:
                vector_df = train_df.select(
                    array_to_vector(train_df[first_col]).alias(first_col)
                )
            elif not is_vector_col:
                vector_assembler = VectorAssembler(outputCol=first_col).setInputCols(
                    input_cols
                )
                vector_df = vector_assembler.transform(train_df).drop(*input_cols)
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
            print(f"Passing {params} to SparkPCA")

            output_col = "pca_features"
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

            feature_col = first_col
            output_col = output_col + "_mean_removed"
            df_for_scoring = transformed_df.select(
                vector_to_array(col(output_col)).alias(output_col), feature_col
            )

        score = self.score(df_for_scoring, feature_col, output_col)
        print(f"trustworthiness score: {score}")

        report_dict = {
            "fit": fit_time,
            "transform": transform_time,
            "total_time": total_time,
            "trustworthiness": score,
            "num_gpus": num_gpus,
            "num_cpus": num_cpus,
            "no_cache": no_cache,
            "train_path": self.args.train_path,
        }

        if not no_cache:
            report_dict["prepare"] = prepare_time

        return report_dict
