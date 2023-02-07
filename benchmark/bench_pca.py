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
from typing import Tuple

import numpy as np
import pandas as pd
from pyspark.ml.feature import PCA
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, sum
from pyspark.sql.types import StructType, StructField, DoubleType

from typing import Iterator, Tuple

from benchmark.utils import WithSparkSession

def score(
    pc_vectors: np.ndarray,
    transformed_df: DataFrame,
    transformed_col: str
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
            transformed_vecs = np.array(list(pdf[transformed_col]), dtype=np.float64)
            partition_score += np.sum(transformed_vecs**2)
        yield pd.DataFrame({'partition_score': [partition_score]})

    total_score = ( 
        transformed_df.mapInPandas(partition_score_udf, 
                                   StructType([StructField('partition_score',DoubleType(),True)]))
                      .agg(sum('partition_score').alias('total_score'))
                      .toPandas()
    )
    total_score = total_score['total_score'][0]
    return orthonormality_score, total_score

def test_pca_bench(
    spark: SparkSession,
    n_components: int,
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

    if num_gpus > 0:
        from spark_rapids_ml.feature import PCA as SparkCumlPCA
        assert num_cpus <= 0
        start_time = time.time()
        if not no_cache:
            df = df.repartition(num_gpus).cache()
            df.count()
            print(f"prepare session and dataset took: {time.time() - start_time} sec")

        start_time = time.time()
        gpu_pca = (
            SparkCumlPCA(num_workers=num_gpus)
            .setK(n_components)
        )

        if is_single_col:
            output_col = "pca_features"
            gpu_pca = gpu_pca.setInputCol(first_col).setOutputCol(output_col)
        else:
            output_cols = ["o" + str(i) for i in range(n_components)]
            gpu_pca = gpu_pca.setInputCols(input_cols).setOutputCols(output_cols)

        gpu_model = gpu_pca.fit(df)
        fit_time = time.time() - start_time
        print(f"gpu fit took: {fit_time} sec")

        start_time = time.time()
        transformed_df = gpu_model.transform(df)
        if is_single_col:
            transformed_df.select((col(output_col)[0]).alias("zero")).agg(sum("zero")).collect()
        else:
            transformed_df.agg(sum(output_cols[0])).collect()
        transform_time = time.time() - start_time
        print(f"gpu transform took: {transform_time} sec")

        total_time = time.time() - func_start_time
        print(f"gpu total took: {total_time} sec")

        df_for_scoring = transformed_df.select(output_col)
        feature_col = output_col
        if not is_single_col:
            feature_col = 'features_array'
            df_for_scoring = transformed_df.select(array(*output_cols).alias(feature_col))
        # restore and change when output is set to vector udt if input is vector udt
        # elif is_vector_col: 
        #    df_for_scoring = transformed_df.select(vector_to_array(feature_col), output_col)

        pc_for_scoring = gpu_model.pc.toArray()

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
        output_col = "pca_features"
        cpu_pca = PCA().setK(n_components)
        cpu_pca = cpu_pca.setInputCol(first_col).setOutputCol(output_col)

        cpu_model = cpu_pca.fit(vector_df)
        fit_time = time.time() - start_time
        print(f"cpu fit took: {fit_time} sec")

        start_time = time.time()
        transformed_df = cpu_model.transform(vector_df)
        transformed_df.select((vector_to_array(output_col)[0]).alias("zero")).agg(sum("zero")).collect()
        transform_time = time.time() - start_time
        print(f"cpu transform took: {transform_time} sec")

        total_time = time.time() - func_start_time
        print(f"cpu total took: {total_time} sec")

        # spark ml does not remove the mean in the transformed features, so do that here
        # needed for scoring
        standard_scaler = StandardScaler().setWithStd(False).setWithMean(True).setInputCol(output_col).setOutputCol(output_col+"_mean_removed")

        scaler_model = standard_scaler.fit(transformed_df)
        transformed_df = scaler_model.transform(transformed_df).drop(output_col)
        
        feature_col = output_col+"_mean_removed"
        pc_for_scoring = cpu_model.pc.toArray()
        df_for_scoring = transformed_df.select(vector_to_array(feature_col).alias(feature_col))


    orthonormality, variance = score(pc_for_scoring, df_for_scoring, feature_col)
    print(f"orthonormality score: {orthonormality}, variance score {variance}")

    return (fit_time, transform_time, total_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_components", type=int, default=3)
    parser.add_argument("--num_gpus", type=int, default=1, help='number of available GPUs. If num_gpus > 0, sparkcuml will run with the number of dataset partitions equal to num_gpus.')
    parser.add_argument("--num_cpus", type=int, default=6, help='number of available CPUs. If num_cpus > 0, spark will run and with the number of dataset partitions to num_cpus.')
    parser.add_argument("--no_cache", action='store_true', default=False, help='whether to enable dataframe repartition, cache and cout outside sparkcuml fit')
    parser.add_argument("--num_runs", type=int, default=2, help='set the number of repetitions for cold/warm runs')
    parser.add_argument("--report_path", type=str, default="")
    parser.add_argument("--parquet_path", type=str, default="")
    parser.add_argument("--spark_confs", action="append", default=[])
    parser.add_argument("--no_shutdown", action='store_true', help="do not stop spark session when finished")
    args = parser.parse_args()

    print(f"invoked time: {datetime.datetime.now()}")

    report_pd = pd.DataFrame()

    with WithSparkSession(args.spark_confs, shutdown=(not args.no_shutdown)) as spark:
        for run_id in range(args.num_runs):
            (fit_time, transform_time, total_time) = test_pca_bench(
                spark,
                args.n_components,
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
                "n_components": args.n_components,
                "num_gpus": args.num_gpus,
                "num_cpus": args.num_cpus,
                "no_cache": args.no_cache,
                "parquet_path": args.parquet_path,
            }

            for sconf in args.spark_confs:
                key, value = sconf.split("=")
                report_dict[key] = value

            alg_name = 'sparkcuml_pca' if args.num_gpus > 0 else 'spark_pca'
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
