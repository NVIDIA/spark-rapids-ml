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
import time
from typing import Any, Dict, List, Optional, Union

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, sum

from benchmark.base import BenchmarkBase
from benchmark.utils import with_benchmark


class BenchmarkUMAP(BenchmarkBase):
    def _add_extra_arguments(self) -> None:
        self._parser.add_argument(
            "--no_cache",
            action="store_true",
            default=False,
            help="whether to enable dataframe repartition, cache and cout outside fit function",
        )

    def run_once(
        self,
        spark: SparkSession,
        train_df: DataFrame,
        features_col: Union[str, List[str]],
        transform_df: Optional[DataFrame],
        label_name: Optional[str],
    ) -> Dict[str, Any]:
        """
        This function evaluates the runtimes of Spark Rapids ML UMAP and Spark PCA, but
        should not be used to compare the two. The purpose is to help understand GPU behavior
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

            params = self.class_params
            gpu_estimator = UMAP(
                num_workers=num_gpus, verbose=self.args.verbose, **params
            )

            if is_single_col:
                gpu_estimator = gpu_estimator.setFeaturesCol(first_col)
            else:
                gpu_estimator = gpu_estimator.setFeaturesCols(input_cols)

            gpu_model, fit_time = with_benchmark(
                "gpu fit", lambda: gpu_estimator.fit(train_df)
            )

            def transform(model: UMAPModel, df: DataFrame) -> DataFrame:
                transformed_df = model.transform(df)
                transformed_df.count()
                return transformed_df

            transformed_df, transform_time = with_benchmark(
                "gpu transform", lambda: transform(gpu_model, train_df)
            )
            total_time = round(time.time() - func_start_time, 2)
            print(f"gpu total took: {total_time} sec")

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

        report_dict = {
            "fit": fit_time,
            "transform": transform_time,
            "total_time": total_time,
            "num_gpus": num_gpus,
            "num_cpus": num_cpus,
            "no_cache": no_cache,
            "train_path": self.args.train_path,
        }

        if not no_cache:
            report_dict["prepare"] = prepare_time

        return report_dict
