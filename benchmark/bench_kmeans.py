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
import time
from typing import List, Union

import numpy as np
import pandas as pd
from pyspark.ml.clustering import KMeans
from pyspark.ml.functions import array_to_vector

from benchmark.utils import WithSparkSession
from sparkcuml.cluster import SparkCumlKMeans

from typing import Dict, Tuple, Any

def bench_alg(
    run_id: int, 
    n_clusters: int,
    max_iter: int,
    tol: float,
    num_gpus: int,
    num_cpus: int,
    no_cache: bool,
    dtype: Union[np.float64, np.float32],
    parquet_path: str,
    spark_confs: List[str],
) -> Tuple[float, float, float]:

    fit_time = None
    transform_time = None
    total_time = None

    func_start_time = time.time()

    df = spark.read.parquet(parquet_path)
    input_col = df.dtypes[0][0]
    output_col = "cluster_idx"

    if num_gpus > 0:
        assert num_cpus <= 0
        start_time = time.time()
        if not no_cache:
            df = df.repartition(num_gpus).cache()
            df.count()
            print(f"prepare session and dataset took: {time.time() - start_time} sec")

        start_time = time.time()
        gpu_estimator = (
            SparkCumlKMeans(num_workers=num_gpus, n_clusters=n_clusters, max_iter=max_iter, tol=tol, init='random', verbose=6)
            .setFeaturesCol(input_col)
            .setPredictionCol(output_col)
        )
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
        vector_df = df.select(array_to_vector(df[input_col]).alias(input_col))
        if not no_cache:
            vector_df = vector_df.cache()
            vector_df.count()
            print(f"prepare session and dataset: {time.time() - start_time} sec")

        start_time = time.time()
        cpu_estimator = KMeans(initMode='random').setFeaturesCol(input_col).setPredictionCol(output_col).setK(n_clusters).setMaxIter(max_iter).setTol(tol)
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
    parser.add_argument("--no_cache", type=bool, default=False, help='whether to enable dataframe repartition, cache and cout outside sparkcuml fit')
    parser.add_argument("--dtype", type=str, choices=["float64"], default="float64")
    parser.add_argument("--num_runs", type=int, default=2, help='set the number of repetitions for cold/warm runs')
    parser.add_argument("--report_path", type=str, default="")
    parser.add_argument("--parquet_path", type=str, default="")
    parser.add_argument("--spark_confs", action="append", default=[])
    args = parser.parse_args()

    report_pd = pd.DataFrame()

    with WithSparkSession(args.spark_confs) as spark:
        for run_id in range(args.num_runs):
            (fit_time, transform_time, total_time) = bench_alg(
                run_id, 
                args.n_clusters,
                args.max_iter,
                args.tol,
                args.num_gpus,
                args.num_cpus,
                args.no_cache,
                args.dtype,
                args.parquet_path,
                args.spark_confs,
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
                "dtype": args.dtype,
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
