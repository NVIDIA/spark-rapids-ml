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

from typing import Dict, Any

def bench_alg(
    run_id: int, 
    n_clusters: int,
    max_iter: int,
    tol: float,
    num_gpus: int,
    num_cpus: int,
    dtype: Union[np.float64, np.float32],
    parquet_path: str,
    spark_confs: List[str],
) -> pd.DataFrame:

    func_start_time = time.time()

    report_row = {
        "run_id": run_id, 
        "fit": None,
        "transform": None,
        "total": None,
        "n_clusters": n_clusters,
        "max_iter": max_iter,
        "tol": tol,
        "num_gpus": num_gpus,
        "num_cpus": num_cpus,
        "dtype": dtype,
        "parquet_path": parquet_path,
    }

    for sconf in spark_confs:
        key, value = sconf.split("=")
        report_row[key] = value

    report_pd = pd.DataFrame(columns = report_row.keys())

    with WithSparkSession(spark_confs) as spark:
        df = spark.read.parquet(parquet_path)
        input_col = df.dtypes[0][0]
        output_col = "cluster_idx"

        if num_gpus > 0:
            assert num_cpus <= 0
            start_time = time.time()
            df = df.repartition(num_gpus).cache()
            df.count()
            print(f"prepare session and dataset took: {time.time() - start_time} sec")

            start_time = time.time()
            gpu_estimator = (
                SparkCumlKMeans(num_workers=num_gpus, n_clusters=n_clusters, max_iter=max_iter, tol=tol)
                .setFeaturesCol(input_col)
                .setPredictionCol(output_col)
            )
            gpu_model = gpu_estimator.fit(df)
            report_row["fit"] = time.time() - start_time
            print(f"gpu fit took: {report_row['fit']} sec")

            start_time = time.time()
            gpu_model.transform(df).count()
            report_row["transform"] = time.time() - start_time
            print(f"gpu transform took: {report_row['transform']} sec")

            report_row['total'] = time.time() - func_start_time
            print(f"gpu total took: {report_row['total']} sec")
            report_pd.loc["sparkcuml_kmeans"] = report_row

        if num_cpus > 0:
            assert num_gpus <= 0
            start_time = time.time()
            df = df.repartition(num_cpus)
            vector_df = df.select(array_to_vector(df[input_col]).alias(input_col)).cache()
            vector_df.count()
            print(f"prepare session and dataset: {time.time() - start_time} sec")

            start_time = time.time()
            cpu_estimator = KMeans().setFeaturesCol(input_col).setPredictionCol(output_col).setK(n_clusters).setMaxIter(max_iter).setTol(tol)
            cpu_model = cpu_estimator.fit(vector_df)
            report_row["fit"] = time.time() - start_time
            print(f"cpu fit took: {report_row['fit']} sec")

            start_time = time.time()
            cpu_model.transform(vector_df).count()
            report_row["transform"] = time.time() - start_time
            print(f"cpu transform took: {report_row['transform']} sec")

            report_row['total'] = time.time() - func_start_time
            print(f"cpu total took: {report_row['total']} sec")
            report_pd.loc["spark_kmeans"] = report_row

        return report_pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, default=200)
    parser.add_argument("--max_iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=0)
    parser.add_argument("--num_gpus", type=int, default=1, help='number of available GPUs. If num_gpus > 0, sparkcuml will run with the number of dataset partitions equal to num_gpus.')
    parser.add_argument("--num_cpus", type=int, default=6, help='number of available CPUs. If num_cpus > 0, spark will run and with the number of dataset partitions to num_cpus.')
    parser.add_argument("--dtype", type=str, choices=["float64"], default="float64")
    parser.add_argument("--num_runs", type=int, default=2, help='set the number of repetitions for cold/warm runs')
    parser.add_argument("--report_path", type=str, default="")
    parser.add_argument("--parquet_path", type=str, default="")
    parser.add_argument("--spark_confs", action="append", default=[])
    args = parser.parse_args()

    report_pd = pd.DataFrame()
    for run_id in range(args.num_runs):
        rpd = bench_alg(
            run_id, 
            args.n_clusters,
            args.max_iter,
            args.tol,
            args.num_gpus,
            args.num_cpus,
            args.dtype,
            args.parquet_path,
            args.spark_confs,
        )
        print(rpd)
        report_pd = pd.concat([report_pd, rpd])

    print(f"\nsummary of the total {args.num_runs} runs:\n")
    print(report_pd)
    if args.report_path != "":
        report_pd.to_csv(args.report_path, mode="a")
        report_pd.to_csv(args.report_path, mode="a")