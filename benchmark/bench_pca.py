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
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import time
from pyspark.ml.feature import PCA as SparkPCA
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import array, col, sum
from pyspark.sql.types import DoubleType, StructField, StructType
from spark_rapids_ml.feature import PCA

from benchmark.base import BenchmarkBase
from benchmark.utils import with_benchmark


class BenchmarkPCA(BenchmarkBase):
    test_cls = PCA
    unsupported_params = test_cls._param_excludes() + [
        "featuresCol",
        "labelCol",
        "predictionCol",
        "probabilityCol",
        "rawPredictionCol",
        "weightCol",
        "leafCol",
    ]

    def add_arguments(self):
        self._parser.add_argument("--no_cache", action='store_true', default=False, help='whether to enable dataframe repartition, cache and cout outside sparkcuml fit')

    def score(self,
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

    def run_once(
        self,
        spark: SparkSession,
        df: DataFrame,
        features_col: Union[str, List[str]],
        label_name: Optional[str],
    ) -> None:
        n_components = self._args.k
        num_gpus = self._args.num_gpus
        num_cpus = self._args.num_cpus
        no_cache = self._args.no_cache

        func_start_time = time.time()

        first_col = df.dtypes[0][0]
        first_col_type = df.dtypes[0][1]
        is_array_col = True if 'array' in first_col_type else False
        is_vector_col = True if 'vector' in first_col_type else False
        is_single_col = is_array_col or is_vector_col

        if not is_single_col:
            input_cols = [c for c in df.schema.names]

        if num_gpus > 0:
            assert num_cpus <= 0
            if not no_cache:
                def cache_df(df: DataFrame):
                    df = df.repartition(num_gpus).cache()
                    df.count()
                    return df
                df, prepare_time = with_benchmark("prepare session and dataset", lambda: cache_df(df))

            params = self.spark_cuml_params
            print(f"Passing {params} to PCA")

            gpu_pca = PCA(num_workers=num_gpus, **params)

            if is_single_col:
                output_col = "pca_features"
                gpu_pca = gpu_pca.setInputCol(first_col).setOutputCol(output_col)
            else:
                output_cols = ["o" + str(i) for i in range(n_components)]
                gpu_pca = gpu_pca.setInputCols(input_cols).setOutputCols(output_cols)

            gpu_model, fit_time = with_benchmark(
                "gpu fit", lambda: gpu_pca.fit(df))

            def transform(df: DataFrame) -> DataFrame:
                transformed_df = gpu_model.transform(df)
                if is_single_col:
                    transformed_df.select((col(output_col)[0]).alias("zero")).agg(sum("zero")).collect()
                else:
                    transformed_df.agg(sum(output_cols[0])).collect()
                return transformed_df

            transformed_df, transform_time = with_benchmark(
                "gpu transform",
                lambda: transform(df)
            )

            total_time = round(time.time() - func_start_time, 2)
            print(f"gpu total took: {total_time} sec")

            # spark ml does not remove the mean in the transformed features, so do that here
            # needed for scoring
            feature_col = output_col
            df_for_scoring = transformed_df.select(array_to_vector(output_col).alias(output_col+"_vec"))
            standard_scaler = (
                StandardScaler()
                .setWithStd(False)
                .setWithMean(True)
                .setInputCol(output_col+"_vec")
                .setOutputCol(output_col+"_mean_removed")
            )
            scaler_model = standard_scaler.fit(df_for_scoring)
            df_for_scoring = (
                scaler_model
                .transform(df_for_scoring)
                .drop(output_col+"_vec")
                .select(vector_to_array(output_col+"_mean_removed").alias(feature_col))
            )
            if not is_single_col:
                feature_col = 'features_array'
                df_for_scoring = transformed_df.select(array(*output_cols).alias(feature_col))
            # restore and change when output is set to vector udt if input is vector udt
            # elif is_vector_col:
            #    df_for_scoring = transformed_df.select(vector_to_array(feature_col), output_col)

            pc_for_scoring = gpu_model.pc.toArray()

        if num_cpus > 0:
            assert num_gpus <= 0
            if is_array_col:
                vector_df = df.select(array_to_vector(df[first_col]).alias(first_col))
            elif not is_vector_col:
                vector_assembler = VectorAssembler(outputCol="features").setInputCols(input_cols)
                vector_df = vector_assembler.transform(df).drop(*input_cols)
                first_col = "features"
            else:
                vector_df = df

            if not no_cache:
                vector_df, prepare_time = with_benchmark(
                    "prepare dataset",
                    vector_df.cache().count()
                )

            output_col = "pca_features"

            params = self.spark_params
            print(f"Passing {params} to SparkPCA")

            cpu_pca = (
                SparkPCA(**params)
                .setInputCol(first_col)
                .setOutputCol(output_col)
            )

            cpu_model, fit_time = with_benchmark("cpu fit",
                                                 lambda: cpu_pca.fit(vector_df))

            def transform():
                transformed_df = cpu_model.transform(vector_df)
                transformed_df.select((vector_to_array(output_col)[0]).alias("zero")).agg(sum("zero")).collect()

            _, transform_time = with_benchmark("cpu transform", transform)

            total_time = round(time.time() - func_start_time, 2)
            print(f"cpu total took: {total_time} sec")

            # spark ml does not remove the mean in the transformed features, so do that here
            # needed for scoring
            standard_scaler = StandardScaler().setWithStd(False).setWithMean(True).setInputCol(output_col).setOutputCol(output_col+"_mean_removed")

            scaler_model = standard_scaler.fit(transformed_df)
            transformed_df = scaler_model.transform(transformed_df).drop(output_col)

            feature_col = output_col+"_mean_removed"
            pc_for_scoring = cpu_model.pc.toArray()
            df_for_scoring = transformed_df.select(vector_to_array(feature_col).alias(feature_col))

        orthonormality, variance = self.score(pc_for_scoring, df_for_scoring, feature_col)
        print(f"orthonormality score: {orthonormality}, variance score {variance}")

        report_dict = {
            "prepare": prepare_time,
            "fit": fit_time,
            "transform": transform_time,
            "total": total_time,
            "orthonormality": orthonormality,
            "variance": variance,
            "k": self._args.k,
            "num_gpus": self._args.num_gpus,
            "num_cpus": self._args.num_cpus,
            "no_cache": self._args.no_cache,
            "train_path": self._args.train_path,
        }

        return report_dict