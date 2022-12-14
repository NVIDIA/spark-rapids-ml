import argparse
from abc import abstractmethod
from typing import Iterator, Union, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import pyspark.sql
from pyspark.sql import SparkSession
from pyspark.sql.functions import array
from sklearn.datasets import make_regression

from benchmark.utils import WithSparkSession
from pyspark.mllib.random import RandomRDDs

from cuml.datasets import make_blobs


def dtype_to_pyspark_type(dtype: Union[np.dtype, str]) -> str:
    """Convert np.dtype to the corresponding pyspark type"""
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        return "float"
    elif dtype == np.float64:
        return "double"
    else:
        raise RuntimeError("Unsupported dtype, found ", dtype)


class DataGen(object):
    """DataGen interface"""

    @abstractmethod
    def gen_dataframe(self, spark: SparkSession) -> Tuple[pyspark.sql.DataFrame, List[str]]:
        raise NotImplementedError()


class DataGenBase(DataGen):
    """Base class datagen"""

    def __init__(self,
                 num_rows: int = 100,
                 num_cols: int = 30,
                 dtype: np.dtype = np.dtype(np.float32),
                 random_state: int = 10) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.dtype = np.dtype(dtype)
        self.random_state = random_state

        self.pyspark_type = dtype_to_pyspark_type(dtype)
        self.feature_cols: Union[str, List[str]] = [f"c{i}" for i in range(num_cols)]
        self.schema = [f"{c} {self.pyspark_type}" for c in self.feature_cols]


class DefaultDataGen(DataGenBase):
    """Generate default dataset only containing features"""

    def gen_dataframe(self, spark: SparkSession) -> Tuple[pyspark.sql.DataFrame, List[str]]:
        rdd = (RandomRDDs
               .uniformVectorRDD(spark, self.num_rows, self.num_cols)
               .map(lambda nparray: nparray.tolist()))

        return spark.createDataFrame(rdd, schema=",".join(self.schema)), self.feature_cols


class BlobsDataGen(DataGenBase):
    """Generate random dataset using cuml.datasets.make_blobs, 
       which creates blobs for bechmarking unsupervised clustering algorithms (e.g. KMeans)"""

    def __init__(self, n_clusters: int = 20, **kargs: Dict[str, Any]) -> None:
        super().__init__(**kargs)
        self.n_clusters = n_clusters

    def gen_dataframe(self, spark: SparkSession) -> Tuple[pyspark.sql.DataFrame, List[str]]:
        "More information about the implementation can be found in RegressionDataGen."

        def make_blobs_udf(iter: Iterator[pd.Series]) -> pd.DataFrame:
            data, _ = make_blobs(self.num_rows, self.num_cols, self.n_clusters, random_state=self.random_state,
                                 dtype=self.dtype)
            data = data.tolist()
            yield pd.DataFrame(data=data)

        return (spark
                .range(0, self.num_rows, 1, 1)
                .mapInPandas(make_blobs_udf, schema=",".join(self.schema))
                ), self.feature_cols


class RegressionDataGen(DataGenBase):
    """Generate regression dataset including features and label."""

    def gen_dataframe(self, spark: SparkSession) -> Tuple[pyspark.sql.DataFrame, List[str]]:
        num_cols = self.num_cols
        random_state = self.random_state
        dtype = self.dtype

        def make_regression_udf(iter: Iterator[pd.Series]) -> pd.DataFrame:
            """Pandas udf to call make_regression of sklearn to generate regression dataset
            """
            total_rows = 0
            for pdf in iter:
                total_rows += pdf.shape[0]
            # here we iterator all batches of a single partition to get total rows.
            X, y = make_regression(n_samples=total_rows, n_features=num_cols, noise=10, random_state=random_state)
            data = np.concatenate((X.astype(dtype), y.reshape(total_rows, 1).astype(dtype)), axis=1)
            del X
            del y
            yield pd.DataFrame(data=data)

        label_col = "label"
        self.schema.append(f"{label_col} {self.pyspark_type}")

        # Each make_regression calling will return regression dataset with different coef.
        # So force to only 1 task to generate the regression dataset, which may cause OOM
        # and perf issue easily. I tested this script can generate 100, 000, 000 * 30
        # matrix without issues with 60g executor memory, which, I think, is really enough
        # to do the perf test.
        return (spark
                .range(0, self.num_rows, 1, 1)
                .mapInPandas(make_regression_udf, schema=",".join(self.schema))
                ), self.feature_cols


class DataGenProxy(DataGen):

    def __init__(self, args):
        if args.category == "default":
            print("DefaultDataGen!")
            self.data_gen = DefaultDataGen(args.num_rows, args.num_cols, args.dtype)
        elif args.category == "regression":
            print("RegressionDataGen!")
            self.data_gen = RegressionDataGen(args.num_rows, args.num_cols, args.dtype, args.random_state)
        elif args.category == "blobs":
            print("BlobsDataGen!")
            self.data_gen = BlobsDataGen(
                n_clusters=args.n_clusters,
                num_rows=args.num_rows,
                num_cols=args.num_cols,
                dtype=args.dtype,
                random_state=args.random_state)

    def gen_dataframe(self, spark: SparkSession) -> Tuple[pyspark.sql.DataFrame, List[str]]:
        return self.data_gen.gen_dataframe(spark)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rows", type=int, default=100)
    parser.add_argument("--num_cols", type=int, default=30)
    parser.add_argument("--dtype", type=str, choices=["float64", "float32"], default="float32")
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--feature_type", type=str, choices=["array", "multi_cols"], default="multi_cols")
    parser.add_argument("--n_clusters", type=int, default=20,
                        help="reauired for using BlobsDataGen and dummy otherwise")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_num_files", type=int)
    parser.add_argument("--overwrite", type=bool, default=False)
    parser.add_argument("--category", type=str, choices=["regression", "default", "blobs"], default="default")
    parser.add_argument("--spark_confs", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    """
    python gen_data.py \
    --num_rows 5000 \
    --num_cols 3000 \
    --dtype "float64" \
    --output_dir "./5k_2k_float64.parquet" \
    --spark_confs "spark.master=local[*]" \
    --spark_confs "spark.driver.memory=128g" 
    """
    args = parse_arguments()

    with WithSparkSession(args.spark_confs) as spark:
        df, feature_cols = DataGenProxy(args).gen_dataframe(spark)

        if args.feature_type == "array":
            df = df.withColumn("feature_array", array(*feature_cols)).drop(*feature_cols)

        if args.output_num_files is not None:
            df = df.repartition(args.output_num_files)

        df.printSchema()

        writer = df.write
        if args.overwrite:
            writer = df.write.mode("overwrite")

        writer.parquet(args.output_dir)

        print("gen_data finished")
