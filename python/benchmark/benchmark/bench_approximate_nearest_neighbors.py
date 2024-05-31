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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import array_to_vector
from pyspark.sql import DataFrame, SparkSession

from benchmark.base import BenchmarkBase
from benchmark.utils import with_benchmark


class BenchmarkApproximateNearestNeighbors(BenchmarkBase):
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

        def dict_str_parse(cmd_value: str) -> Dict[str, Any]:
            res: Dict[str, Any] = {}
            for pair in cmd_value.split(","):
                key, value = pair.split("=")
                assert key in {
                    "algorithm",
                    "nlist",
                    "nprobe",
                    "numHashTables",
                    "bucketLength",
                }
                if key == "algorithm":
                    res[key] = value
                elif key == "bucketLength":
                    res[key] = float(value)
                else:
                    res[key] = int(value)

            return res

        self._parser.add_argument(
            "--cpu_algo_params",
            type=dict_str_parse,
            help="algorithm parameters to use in CPU Classes",
        )

        self._parser.add_argument(
            "--gpu_algo_params",
            type=dict_str_parse,
            help="algorithm parameters to use in GPU Classes",
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
        cpu_algo_params = self.args.cpu_algo_params
        gpu_algo_params = self.args.gpu_algo_params

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

        from pyspark.sql.functions import monotonically_increasing_id

        train_df = train_df.withColumn("id", monotonically_increasing_id())

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

        if num_gpus > 0:
            from spark_rapids_ml.knn import (
                ApproximateNearestNeighbors,
                ApproximateNearestNeighborsModel,
            )

            assert num_cpus <= 0
            if not no_cache:
                (train_df, query_df), prepare_time = with_benchmark(
                    "prepare dataset", lambda: cache_df(train_df, query_df)
                )

            rt_algo = gpu_algo_params["algorithm"]
            rt_algo_params = {
                "nlist": gpu_algo_params["nlist"],
                "nprobe": gpu_algo_params["nprobe"],
            }
            params = self.class_params
            gpu_estimator = ApproximateNearestNeighbors(
                verbose=self.args.verbose,
                algorithm=rt_algo,
                algoParams=rt_algo_params,
                **params,
            ).setIdCol("id")

            if is_single_col:
                gpu_estimator = gpu_estimator.setInputCol(first_col)
            else:
                gpu_estimator = gpu_estimator.setInputCols(input_cols)

            gpu_model, fit_time = with_benchmark(
                "gpu fit", lambda: gpu_estimator.fit(train_df)
            )

            def transform(
                model: ApproximateNearestNeighborsModel, df: DataFrame
            ) -> DataFrame:
                (item_df_withid, query_df_withid, knn_df) = model.kneighbors(df)
                knn_df = knn_df.cache()
                knn_df.count()
                return knn_df

            knn_df, transform_time = with_benchmark(
                "gpu transform", lambda: transform(gpu_model, query_df)
            )
            total_time = round(time.time() - func_start_time, 2)
            print(f"gpu total took: {total_time} sec")

        if num_cpus > 0:
            assert num_gpus <= 0
            if is_array_col:
                vector_df = train_df.select(
                    "id", array_to_vector(train_df[first_col]).alias(first_col)
                )

                vector_query_df = query_df.select(
                    "id", array_to_vector(train_df[first_col]).alias(first_col)
                )

            elif not is_vector_col:
                vector_assembler = VectorAssembler(outputCol="features").setInputCols(
                    input_cols
                )
                vector_df = vector_assembler.transform(train_df).drop(*input_cols)
                vector_query_df = vector_assembler.transform(query_df).drop(*input_cols)
                first_col = "features"
            else:
                vector_df = train_df
                vector_query_df = query_df

            if not no_cache:

                (vector_df, vector_query_df), prepare_time = with_benchmark(
                    "prepare dataset", lambda: cache_df(vector_df, vector_query_df)
                )

            from pyspark.ml.feature import (
                BucketedRandomProjectionLSH,
                BucketedRandomProjectionLSHModel,
            )

            cpu_estimator = BucketedRandomProjectionLSH(
                inputCol=first_col,
                outputCol="hashes",
                **cpu_algo_params,
            )

            cpu_model, fit_time = with_benchmark(
                "cpu fit time", lambda: cpu_estimator.fit(vector_df)
            )

            def cpu_transform(
                model: BucketedRandomProjectionLSHModel,
                df: DataFrame,
                df_query: DataFrame,
                n_neighbors: int,
            ) -> None:
                queries = df_query.collect()
                for row in queries:
                    query = row[first_col]
                    knn_df = model.approxNearestNeighbors(
                        dataset=df, key=query, numNearestNeighbors=n_neighbors
                    )
                    knn_df.count()

            _, transform_time = with_benchmark(
                "cpu transform",
                lambda: cpu_transform(
                    cpu_model, vector_df, vector_query_df, n_neighbors
                ),
            )

            total_time = round(time.time() - func_start_time, 2)
            print(f"cpu total took: {total_time} sec")

        if num_gpus > 0:
            eval_start_time = time.time()
            input_col_actual: Union[str, List[str]] = (
                first_col if is_single_col else input_cols
            )
            avg_recall = self.evaluate_avg_recall(
                train_df, query_df, knn_df, n_neighbors, input_col_actual
            )
            print(f"evaluation took: {round(time.time() - eval_start_time, 2)} sec")
        else:
            print(f"benchmark script does not evaluate LSH")
            avg_recall = None

        report_dict = {
            "fit": fit_time,
            "transform": transform_time,
            "total_time": total_time,
            "avg_recall": avg_recall,
            "n_neighbors": n_neighbors,
            "fraction_sampled_queries": fraction_sampled_queries,
            "algo_params": gpu_algo_params if num_gpus > 0 else cpu_algo_params,
            "num_gpus": num_gpus,
            "num_cpus": num_cpus,
            "no_cache": no_cache,
            "train_path": self.args.train_path,
        }

        if not no_cache:
            report_dict["prepare"] = prepare_time

        return report_dict

    def evaluate_avg_recall(
        self,
        train_df: DataFrame,
        query_df: DataFrame,
        knn_df: DataFrame,
        n_neighbors: int,
        input_col: Union[str, List[str]],
        limit: int = 1000,
    ) -> float:

        knn_selected = knn_df.limit(limit).sort("query_id").collect()
        qid_eval_set = set([row["query_id"] for row in knn_selected])

        query_df_eval_set = query_df.filter(query_df["id"].isin(qid_eval_set))

        from spark_rapids_ml.knn import NearestNeighbors

        gpu_nn = (
            NearestNeighbors(n_neighbors=n_neighbors).setK(n_neighbors).setIdCol("id")
        )

        if isinstance(input_col, str):
            gpu_nn = gpu_nn.setInputCol(input_col)
        else:
            gpu_nn = gpu_nn.setInputCols(input_col)

        gpu_model = gpu_nn.fit(train_df)
        _, _, knn_df_exact = gpu_model.kneighbors(query_df_eval_set)

        knn_exact_selected = (
            knn_df_exact.filter(knn_df_exact["query_id"].isin(qid_eval_set))
            .sort("query_id")
            .collect()
        )

        indices_ann = np.array([row["indices"] for row in knn_selected])
        indices_exact = np.array([row["indices"] for row in knn_exact_selected])
        return self.cal_avg_recall(indices_ann, indices_exact)

    def cal_avg_recall(
        self, indices_ann: np.ndarray, indices_exact: np.ndarray
    ) -> float:
        assert (
            indices_ann.shape == indices_exact.shape
        ), f"indices_ann.shape {indices_ann.shape} does not match indices_exact.shape {indices_exact.shape}"
        n_neighbors = indices_ann.shape[1]

        retrievals = [np.intersect1d(a, b) for a, b in zip(indices_ann, indices_exact)]
        recalls = np.array([len(nns) / n_neighbors for nns in retrievals])
        return recalls.mean()
