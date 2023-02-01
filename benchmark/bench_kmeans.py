#
# Copyright (c) 2022, NVIDIA CORPORATION.
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

import argparse
import datetime
import time
from typing import List, Union

import numpy as np
import pandas as pd
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import array_to_vector

from benchmark.utils import WithSparkSession

from typing import Dict, Tuple, Any

def bench_alg(
    n_clusters: int,
    max_iter: int,
    tol: float,
    num_gpus: int,
    num_cpus: int,
    no_cache: bool,
    parquet_path: str,
) -> Tuple[float, float, float]:

    fit_time = None
    transform_time = None
    total_time = None

    func_start_time = time.time()

    df = spark.read.parquet(parquet_path)
    first_col = df.dtypes[0][0]
    first_col_type = df.dtypes[0][1]
    is_array_col = True if 'array' in first_col_type else False
    is_vector_col = True if 'vector' in first_col_type else False
    is_single_col = is_array_col or is_vector_col
    if not is_single_col:
        input_cols = [c for c in df.schema.names]
    output_col = "cluster_idx"

    if num_gpus > 0:
        from spark_rapids_ml.clustering import KMeans as SparkCumlKMeans
        assert num_cpus <= 0
        start_time = time.time()
        if not no_cache:
            df = df.repartition(num_gpus).cache()
            df.count()
            print(f"prepare session and dataset took: {time.time() - start_time} sec")

        start_time = time.time()
        gpu_estimator = (
            SparkCumlKMeans(num_workers=num_gpus, n_clusters=n_clusters, max_iter=max_iter, tol=tol, init='random', verbose=6)
            .setPredictionCol(output_col)
        )

        if is_single_col:
            gpu_estimator = gpu_estimator.setFeaturesCol(first_col)
        else:
            gpu_estimator = gpu_estimator.setFeaturesCol(input_cols)

        gpu_model = gpu_estimator.fit(df)
        fit_time = time.time() - start_time
        print(f"gpu fit took: {fit_time} sec")

        start_time = time.time()
        gpu_model.transform(df).count()
        transform_time = time.time() - start_time
        print(f"gpu transform took: {transform_time} sec")

        total_time = time.time() - func_start_time
        print(f"gpu total took: {total_time} sec")

    if num_cpus > 0:
        assert num_gpus <= 0
        start_time = time.time()
        if is_array_col:
            vector_df = df.select(array_to_vector(df[first_col]).alias(first_col))
        elif not is_vector_col:
            vector_assembler = VectorAssembler(outputCol="features").setInputCols(input_cols)
            vector_df = vector_assembler.transform(df).drop(*input_cols)
            first_col = "features"
        else:
            vector_df = df

        if not no_cache:
            vector_df = vector_df.cache()
            vector_df.count()
            print(f"prepare session and dataset: {time.time() - start_time} sec")

        start_time = time.time()
        cpu_estimator = KMeans(initMode='random').setFeaturesCol(first_col).setPredictionCol(output_col).setK(n_clusters).setMaxIter(max_iter).setTol(tol)
        cpu_model = cpu_estimator.fit(vector_df)
        fit_time = time.time() - start_time
        print(f"cpu fit took: {fit_time} sec")

        start_time = time.time()
        cpu_model.transform(vector_df).count()
        transform_time = time.time() - start_time
        print(f"cpu transform took: {transform_time} sec")

        total_time = time.time() - func_start_time
        print(f"cpu total took: {total_time} sec")

    return (fit_time, transform_time, total_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, default=200)
    parser.add_argument("--max_iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=-1, help='sparkcuml requires tol to be negative in order to finish max_iters')
    parser.add_argument("--num_gpus", type=int, default=1, help='number of available GPUs. If num_gpus > 0, sparkcuml will run with the number of dataset partitions equal to num_gpus.')
    parser.add_argument("--num_cpus", type=int, default=6, help='number of available CPUs. If num_cpus > 0, spark will run and with the number of dataset partitions to num_cpus.')
    parser.add_argument("--no_cache", action='store_true', default=False, help='whether to enable dataframe repartition, cache and cout outside sparkcuml fit')
    parser.add_argument("--num_runs", type=int, default=2, help='set the number of repetitions for cold/warm runs')
    parser.add_argument("--report_path", type=str, default="")
    parser.add_argument("--parquet_path", type=str, default="")
    parser.add_argument("--spark_confs", action="append", default=[])
    parser.add_argument("--no_shutdown", action='store_true', default=False, help="do not stop spark session when finished")
    args = parser.parse_args()

    print(f"invoked time: {datetime.datetime.now()}")

    report_pd = pd.DataFrame()

    with WithSparkSession(args.spark_confs, shutdown=(not args.no_shutdown)) as spark:
        for run_id in range(args.num_runs):
            (fit_time, transform_time, total_time) = bench_alg(
                args.n_clusters,
                args.max_iter,
                args.tol,
                args.num_gpus,
                args.num_cpus,
                args.no_cache,
                args.parquet_path,
            )

            report_dict = {
                "run_id": run_id,
                "fit": fit_time,
                "transform": transform_time,
                "total": total_time,
                "n_clusters": args.n_clusters,
                "max_iter": args.max_iter,
                "tol": args.tol,
                "num_gpus": args.num_gpus,
                "num_cpus": args.num_cpus,
                "no_cache": args.no_cache,
                "parquet_path": args.parquet_path,
            }

            for sconf in args.spark_confs:
                key, value = sconf.split("=")
                report_dict[key] = value

            alg_name = 'sparkcuml_kmeans' if args.num_gpus > 0 else 'spark_kmeans'
            pdf = pd.DataFrame(
                data = {k : [v] for k, v in report_dict.items()},
                index = [alg_name]
            )
            print(pdf)
            report_pd = pd.concat([report_pd, pdf])

    print(f"\nsummary of the total {args.num_runs} runs:\n")
    print(report_pd)
    if args.report_path != "":
        report_pd.to_csv(args.report_path, mode="a")
        report_pd.to_csv(args.report_path, mode="a")
