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
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import array_to_vector
from pyspark.sql import DataFrame, SparkSession

from benchmark.base import BenchmarkBase
from benchmark.utils import with_benchmark
from spark_rapids_ml.core import (
    EvalMetricInfo,
    _ConstructFunc,
    _EvaluateFunc,
    _TransformFunc,
    alias,
)
from spark_rapids_ml.knn import ApproximateNearestNeighborsModel


class CPUNearestNeighborsModel(ApproximateNearestNeighborsModel):
    def __init__(self, item_df: DataFrame):
        super().__init__(item_df)

    def kneighbors(
        self, query_df: DataFrame, sort_knn_df_by_query_id: bool = True
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        self._item_df_withid = self._ensureIdCol(self._item_df_withid)
        return super().kneighbors(
            query_df, sort_knn_df_by_query_id=sort_knn_df_by_query_id
        )

    def _get_cuml_transform_func(
        self, dataset: DataFrame, eval_metric_info: Optional[EvalMetricInfo] = None
    ) -> Tuple[
        _ConstructFunc,
        _TransformFunc,
        Optional[_EvaluateFunc],
    ]:
        self._cuml_params["algorithm"] = "brute"
        _, _transform_internal, _ = super()._get_cuml_transform_func(
            dataset, eval_metric_info
        )

        from sklearn.neighbors import NearestNeighbors as SKNN

        n_neighbors = self.getK()

        def _construct_sknn() -> SKNN:
            nn_object = SKNN(algorithm="brute", n_neighbors=n_neighbors)
            return nn_object

        return _construct_sknn, _transform_internal, None

    def _concate_pdf_batches(self) -> bool:
        return False


class BenchmarkNearestNeighbors(BenchmarkBase):
    def _supported_class_params(self) -> Dict[str, Any]:
        params = {"n_neighbors": 200}
        return params

    def _add_extra_arguments(self) -> None:
        self._parser.add_argument(
            "--no_cache",
            action="store_true",
            default=False,
            help="whether to enable dataframe repartition, cache and cout outside fit function",
        )

        self._parser.add_argument(
            "--fraction_sampled_queries",
            type=float,
            required=True,
            help="the number of vectors sampled from the dataset as query vectors",
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
        This function evaluates the runtimes of Spark Rapids ML NearestNeighbors and Spark LSH, but
        should not be used to compare the two. The purpose is to help understand GPU behavior
        and performance.
        """
        num_gpus = self.args.num_gpus
        num_cpus = self.args.num_cpus
        no_cache = self.args.no_cache
        n_neighbors = self.args.n_neighbors
        fraction_sampled_queries = self.args.fraction_sampled_queries
        seed = 0

        func_start_time = time.time()

        first_col = train_df.dtypes[0][0]
        first_col_type = train_df.dtypes[0][1]
        is_array_col = True if "array" in first_col_type else False
        is_vector_col = True if "vector" in first_col_type else False
        is_single_col = is_array_col or is_vector_col
        if not is_single_col:
            input_cols = [c for c in train_df.schema.names]

        query_df = train_df.sample(
            withReplacement=False, fraction=fraction_sampled_queries, seed=seed
        )

        def cache_df(dfA: DataFrame, dfB: DataFrame) -> Tuple[DataFrame, DataFrame]:
            dfA = dfA.cache()
            dfB = dfB.cache()

            def func_dummy(pdf_iter):  # type: ignore
                import pandas as pd

                yield pd.DataFrame({"dummy": [1]})

            dfA.mapInPandas(func_dummy, schema="dummy int").count()
            dfB.mapInPandas(func_dummy, schema="dummy int").count()
            return (dfA, dfB)

        params = self.class_params
        if num_gpus > 0:
            from spark_rapids_ml.knn import NearestNeighbors, NearestNeighborsModel

            assert num_cpus <= 0
            if not no_cache:
                (train_df, query_df), prepare_time = with_benchmark(
                    "prepare dataset", lambda: cache_df(train_df, query_df)
                )

            gpu_estimator = NearestNeighbors(
                num_workers=num_gpus, verbose=self.args.verbose, **params
            )

            if is_single_col:
                gpu_estimator = gpu_estimator.setInputCol(first_col)
            else:
                gpu_estimator = gpu_estimator.setInputCols(input_cols)

            gpu_model, fit_time = with_benchmark(
                "gpu fit", lambda: gpu_estimator.fit(train_df)
            )

            def transform(model: NearestNeighborsModel, df: DataFrame) -> DataFrame:
                (item_df_withid, query_df_withid, knn_df) = model.kneighbors(df)
                knn_df.count()
                return knn_df

            knn_df, transform_time = with_benchmark(
                "gpu transform", lambda: transform(gpu_model, query_df)
            )
            total_time = round(time.time() - func_start_time, 2)
            print(f"gpu total took: {total_time} sec")

        if num_cpus > 0:
            assert num_gpus <= 0
            if not no_cache:
                (train_df, query_df), prepare_time = with_benchmark(
                    "prepare dataset", lambda: cache_df(train_df, query_df)
                )

            def get_cpu_model() -> CPUNearestNeighborsModel:
                cpu_estimator = CPUNearestNeighborsModel(train_df).setK(
                    params["n_neighbors"]
                )

                return cpu_estimator

            cpu_model, fit_time = with_benchmark(
                "cpu fit time", lambda: get_cpu_model()
            )

            if is_single_col:
                cpu_model = cpu_model.setInputCol(first_col)
            else:
                cpu_model = cpu_model.setInputCols(input_cols)

            def cpu_transform(
                model: CPUNearestNeighborsModel, df: DataFrame
            ) -> DataFrame:
                (item_df_withid, query_df_withid, knn_df) = model.kneighbors(df)
                knn_df.count()
                return knn_df

            knn_df, transform_time = with_benchmark(
                "cpu transform",
                lambda: cpu_transform(cpu_model, query_df),
            )

            total_time = round(time.time() - func_start_time, 2)
            print(f"cpu total took: {total_time} sec")

        report_dict = {
            "fit": fit_time,
            "transform": transform_time,
            "total_time": total_time,
            "n_neighbors": n_neighbors,
            "fraction_sampled_queries": fraction_sampled_queries,
            "num_gpus": num_gpus,
            "num_cpus": num_cpus,
            "no_cache": no_cache,
            "train_path": self.args.train_path,
        }

        if not no_cache:
            report_dict["prepare"] = prepare_time

        return report_dict
