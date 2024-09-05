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
        params = {"k": 200}
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
                    "algorithm",  # gpu
                    "nlist",  # gpu ivfflat, ivfpq
                    "nprobe",  # gpu ivfflat, ivfpq
                    "numHashTables",  # cpu lsh
                    "bucketLength",  # cpu lsh
                    "M",  # gpu ivfpq, ivfpq
                    "n_bits",  # gpu, ivfpq
                    "build_algo",  # gpu cagra build
                    "intermediate_graph_degree",  # gpu cagra build
                    "graph_degree",  # gpu cagra build
                    "itopk_size",  # gpu cagra search
                    "max_iterations",  # gpu cagra search
                    "min_iterations",  # gpu cagra search
                    "search_width",  # gpu cagra search
                    "num_random_samplings",  # gpu cagra search
                }
                if key in {"algorithm", "build_algo"}:
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

            rt_algo_params = gpu_algo_params.copy()
            rt_algo_params.pop("algorithm")

            rt_algo_params = rt_algo_params if rt_algo_params else None

            params = self.class_params
            gpu_estimator = ApproximateNearestNeighbors(
                verbose=self.args.verbose,
                algorithm=rt_algo,
                algoParams=rt_algo_params,
                metric="sqeuclidean" if rt_algo == "cagra" else "euclidean",
                **params,
            ).setIdCol("id")

            print(f"Running algorithm '{rt_algo}' with algoParams {rt_algo_params}")

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

            fit_time = "Not-supported"
            transform_time = "Not-supported"
            total_time = "Not-supported"  # type: ignore
            print(
                "Currently does not support CPU benchmarking for approximate nearest neighbors"
            )

        if num_gpus > 0:
            eval_start_time = time.time()
            input_col_actual: Union[str, List[str]] = (
                first_col if is_single_col else input_cols
            )
            avg_recall = self.evaluate_avg_recall(
                train_df, query_df, knn_df, self.args.k, input_col_actual
            )
            print(f"evaluation took: {round(time.time() - eval_start_time, 2)} sec")
        else:
            print(f"benchmark script does not evaluate LSH")
            avg_recall = "Not-supported"  # type: ignore

        report_dict = {
            "fit": fit_time,
            "transform": transform_time,
            "total_time": total_time,
            "avg_recall": avg_recall,
            "k": self.args.k,
            "fraction_sampled_queries": fraction_sampled_queries,
            "algorithm": gpu_estimator.getAlgorithm() if num_gpus > 0 else "LSH",
            "algoParams": (
                gpu_estimator.getAlgoParams() if num_gpus > 0 else cpu_algo_params
            ),
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

        fraction = min(limit / knn_df.count(), 1.0)

        knn_selected = knn_df.sample(fraction).sort("query_id").collect()
        qid_eval_set = set([row["query_id"] for row in knn_selected])

        query_df_eval_set = query_df.filter(query_df["id"].isin(qid_eval_set))

        from spark_rapids_ml.knn import NearestNeighbors

        gpu_nn = NearestNeighbors().setK(n_neighbors).setIdCol("id")

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
