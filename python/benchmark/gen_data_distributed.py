# Copyright (c) 2007-2024 The scikit-learn developers. All rights reserved.
# Modifications copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

import logging
import random
from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import pyspark
import scipy as sp
from gen_data import DataGenBase, DefaultDataGen, main
from packaging import version
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.mllib.random import RandomRDDs
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from sklearn.datasets import (
    make_blobs,
    make_classification,
    make_low_rank_matrix,
    make_regression,
)
from sklearn.datasets._samples_generator import _generate_hypercube
from sklearn.utils import shuffle as util_shuffle

from benchmark.utils import inspect_default_params_from_func

if TYPE_CHECKING:
    import cupy


class DataGenBaseMeta(DataGenBase):
    """Base class datagen with meta info support"""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def gen_dataframe_and_meta(
        self, spark: SparkSession
    ) -> Tuple[DataFrame, List[str], np.ndarray]:
        raise NotImplementedError()

    def gen_dataframe(self, spark: SparkSession) -> Tuple[DataFrame, List[str]]:
        df, feature_cols, _ = self.gen_dataframe_and_meta(spark)
        return df, feature_cols


class BlobsDataGen(DataGenBaseMeta):
    """Generate random dataset using distributed calls to sklearn.datasets.make_blobs,
    which creates blobs for benchmarking unsupervised clustering algorithms (e.g. KMeans)
    """

    def __init__(self, argv: List[Any]) -> None:
        super().__init__()
        self._parse_arguments(argv)

    def _supported_extra_params(self) -> Dict[str, Any]:
        params = inspect_default_params_from_func(
            make_blobs, ["n_samples", "n_features", "return_centers"]
        )
        # must replace the None to the correct type
        params["centers"] = int
        params["random_state"] = int
        params["include_labels"] = bool
        return params

    def gen_dataframe_and_meta(
        self, spark: SparkSession
    ) -> Tuple[DataFrame, List[str], np.ndarray]:
        dtype = self.dtype
        params = self.extra_params

        if "random_state" not in params:
            # for reproducible dataset.
            params["random_state"] = 1

        print(f"Passing {params} to make_blobs")

        include_labels = params.pop("include_labels", False)

        rows = self.num_rows
        cols = self.num_cols
        assert self.args is not None
        num_partitions = self.args.output_num_files

        # Set num_partitions to Spark's default if output_num_files is not provided.
        if num_partitions is None:
            num_partitions = spark.sparkContext.defaultParallelism

        # Produce partition seeds for reproducibility.
        random.seed(params["random_state"])
        seed_maxval = 100 * num_partitions
        partition_seeds = random.sample(range(1, seed_maxval), num_partitions)

        partition_sizes = [rows // num_partitions] * num_partitions
        partition_sizes[-1] += rows % num_partitions

        # Generate centers upfront.
        _, _, centers = make_blobs(
            n_samples=1, n_features=cols, **params, return_centers=True
        )

        # Update params for partition-specific calls.
        params["centers"] = centers
        del params["random_state"]

        maxRecordsPerBatch = int(
            spark.sparkContext.getConf().get(
                "spark.sql.execution.arrow.maxRecordsPerBatch", "10000"
            )
        )

        # UDF to distribute make_blobs() calls across partitions. Each partition
        # produces an equal fraction of the total samples around the predefined centers.
        def make_blobs_udf(iter: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
            for pdf in iter:
                partition_index = pdf.iloc[0][0]
                n_partition_samples = partition_sizes[partition_index]
                X, y = make_blobs(
                    n_samples=n_partition_samples,
                    n_features=cols,
                    **params,
                    random_state=partition_seeds[partition_index],
                )
                if include_labels:
                    data = np.concatenate(
                        (
                            X.astype(dtype),
                            y.reshape(n_partition_samples, 1).astype(dtype),
                        ),
                        axis=1,
                    )
                else:
                    data = X.astype(dtype)
                del X
                del y
                for i in range(0, n_partition_samples, maxRecordsPerBatch):
                    end_idx = min(i + maxRecordsPerBatch, n_partition_samples)
                    yield pd.DataFrame(data=data[i:end_idx])

        if include_labels:
            label_col = "label"
            self.schema.append(f"{label_col} {self.pyspark_type}")

        return (
            (
                spark.range(
                    0, num_partitions, numPartitions=num_partitions
                ).mapInPandas(make_blobs_udf, schema=",".join(self.schema))
            ),
            self.feature_cols,
            centers,
        )


class LowRankMatrixDataGen(DataGenBase):
    """Generate random dataset using a distributed version of sklearn.datasets.make_low_rank_matrix,
    which creates large low rank matrices for benchmarking dimensionality reduction algos like pca
    """

    def __init__(self, argv: List[Any]) -> None:
        super().__init__()
        self._parse_arguments(argv)

    def _supported_extra_params(self) -> Dict[str, Any]:
        params = inspect_default_params_from_func(
            make_low_rank_matrix, ["n_samples", "n_features"]
        )
        # must replace the None to the correct type
        params["random_state"] = int
        params["use_gpu"] = bool
        params["logistic_regression"] = bool
        return params

    def gen_dataframe(self, spark: SparkSession) -> Tuple[DataFrame, List[str]]:
        dtype = self.dtype
        params = self.extra_params

        if "random_state" not in params:
            # for reproducible dataset.
            params["random_state"] = 1

        print(f"Passing {params} to make_low_rank_matrix")

        rows = self.num_rows
        cols = self.num_cols
        assert self.args is not None
        num_partitions = self.args.output_num_files

        # Set num_partitions to Spark's default if output_num_files is not provided.
        if num_partitions is None:
            num_partitions = spark.sparkContext.defaultParallelism

        n = min(rows, cols)
        np.random.seed(params["random_state"])
        # If params not provided, set to defaults.
        effective_rank = params.get("effective_rank", 10)
        tail_strength = params.get("tail_strength", 0.5)
        use_gpu = params.get("use_gpu", False)

        partition_sizes = [rows // num_partitions] * num_partitions
        partition_sizes[-1] += rows % num_partitions
        # Check sizes to ensure QR decomp produces a matrix of the correct dimension.
        assert partition_sizes[0] >= cols, (
            f"Num samples per partition ({partition_sizes[0]}) must be >= num_features ({cols});"
            f" decrease num_partitions from {num_partitions} to <= {rows // cols}"
        )

        # Generate U, S, V, the SVD decomposition of the output matrix.
        # Code adapted from sklearn.datasets.make_low_rank_matrix().
        singular_ind = np.arange(n, dtype=dtype)
        low_rank = (1 - tail_strength) * np.exp(
            -1.0 * (singular_ind / effective_rank) ** 2
        )
        tail = tail_strength * np.exp(-0.1 * singular_ind / effective_rank)
        # S and V are generated upfront, U is generated across partitions.
        s = np.identity(n) * (low_rank + tail)
        v, _ = np.linalg.qr(
            np.random.standard_normal(size=(cols, n)),
            mode="reduced",
        )

        # Precompute the S*V.T multiplicland with partition-wise normalization.
        sv_normed = np.dot(s, v.T) * np.sqrt(1 / num_partitions)
        del s
        del v

        maxRecordsPerBatch = int(
            spark.sparkContext.getConf().get(
                "spark.sql.execution.arrow.maxRecordsPerBatch", "10000"
            )
        )

        # UDF for distributed generation of U and the resultant product U*S*V.T
        def make_matrix_udf(iter: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
            for pdf in iter:
                use_cupy = use_gpu
                if use_cupy:
                    try:
                        import cupy as cp
                    except ImportError:
                        use_cupy = False
                        logging.warning("cupy import failed; falling back to numpy.")

                partition_index = pdf.iloc[0][0]
                n_partition_rows = partition_sizes[partition_index]
                # Additional batch-wise normalization.
                if use_cupy:
                    batch_norm = cp.sqrt(-(-n_partition_rows // maxRecordsPerBatch))
                    sv_batch_normed = cp.asarray(sv_normed) * batch_norm
                else:
                    batch_norm = np.sqrt(-(-n_partition_rows // maxRecordsPerBatch))
                    sv_batch_normed = sv_normed * batch_norm
                del batch_norm
                for i in range(0, n_partition_rows, maxRecordsPerBatch):
                    end_idx = min(i + maxRecordsPerBatch, n_partition_rows)
                    if use_cupy:
                        u, _ = cp.linalg.qr(
                            cp.random.standard_normal(size=(end_idx - i, n)),
                            mode="reduced",
                        )
                        data = cp.dot(u, sv_batch_normed).get()
                    else:
                        u, _ = np.linalg.qr(
                            np.random.standard_normal(size=(end_idx - i, n)),
                            mode="reduced",
                        )
                        data = np.dot(u, sv_batch_normed)
                    del u
                    yield pd.DataFrame(data=data)

        return (
            (
                spark.range(
                    0, num_partitions, numPartitions=num_partitions
                ).mapInPandas(make_matrix_udf, schema=",".join(self.schema))
            ),
            self.feature_cols,
        )


class RegressionDataGen(DataGenBaseMeta):
    """Generate regression dataset using a distributed version of sklearn.datasets.regression,
    including features and labels.
    """

    def __init__(self, argv: List[Any]) -> None:
        super().__init__()
        self._parse_arguments(argv)

    def _supported_extra_params(self) -> Dict[str, Any]:
        params = inspect_default_params_from_func(
            make_regression, ["n_samples", "n_features", "coef"]
        )
        # must replace the None to the correct type
        params["effective_rank"] = int
        params["random_state"] = int
        params["n_classes"] = int
        params["use_gpu"] = bool
        params["logistic_regression"] = bool
        return params

    def gen_dataframe_and_meta(
        self, spark: SparkSession
    ) -> Tuple[DataFrame, List[str], np.ndarray]:
        dtype = self.dtype
        params = self.extra_params

        if "random_state" not in params:
            # for reproducible dataset.
            params["random_state"] = 1

        print(f"Passing {params} to make_regression")

        rows = self.num_rows
        cols = self.num_cols
        assert self.args is not None
        num_partitions = self.args.output_num_files

        # Set num_partitions to Spark's default if output_num_files is not provided.
        if num_partitions is None:
            num_partitions = spark.sparkContext.defaultParallelism

        # Retrieve input params or set to defaults.
        seed = params["random_state"]
        generator = np.random.RandomState(seed)
        bias = params.get("bias", 0.0)
        noise = params.get("noise", 0.0)
        shuffle = params.get("shuffle", True)
        effective_rank = params.get("effective_rank", None)
        n_informative = params.get("n_informative", 10)
        n_targets = params.get("n_targets", 1)
        n_classes = params.get("n_classes", 2)
        use_gpu = params.get("use_gpu", False)
        logistic_regression = params.get("logistic_regression", False)

        # Check for multinomial logistic regression
        if logistic_regression and n_classes < 2:
            logging.warning(
                "Can not have logistic regression with 2 classes, default to 2"
            )
            n_classes = 2

        multinomial_log = logistic_regression and (n_classes > 2)

        # Check for bias for each class
        if multinomial_log:
            if isinstance(bias, list):
                if len(bias) < n_classes:
                    logging.warning(
                        "Insufficient bias number, setting all the the first entry"
                    )
                    bias = [bias[0] for i in range(n_classes)]
            else:
                bias = [bias for i in range(n_classes)]

        # Description (from sklearn):
        #
        # Input set is either well conditioned (default) or has a low rank fat tail singular profile (see LowRankMatrixDataGen).
        # Output is generated by applying a (potentially biased) random linear regression model to the input, with n_informative
        # nonzero regressors and some gaussian centered noise with adjustable scale.
        #
        # Code adapted from sklearn.datasets.make_regression().
        if effective_rank is not None:
            tail_strength = params.get("tail_strength", 0.5)
            lrm_input_args = [
                "--num_rows",
                str(rows),
                "--num_cols",
                str(cols),
                "--dtype",
                str(dtype),
                "--output_dir",
                "temp",
                "--output_num_files",
                str(num_partitions),
                "--effective_rank",
                str(effective_rank),
                "--tail_strength",
                str(tail_strength),
                "--random_state",
                str(seed),
                "--use_gpu",
                str(use_gpu),
            ]
            # Generate a low-rank, fat tail input set.
            X, _ = LowRankMatrixDataGen(lrm_input_args).gen_dataframe(spark)
            assert X.rdd.getNumPartitions() == num_partitions, (
                f"Unexpected num partitions received from LowRankMatrix;"
                f"expected {num_partitions}, got {X.rdd.getNumPartitions()}"
            )
        else:
            # Randomly generate a well-conditioned input set.
            X = spark.createDataFrame(
                RandomRDDs.normalVectorRDD(
                    spark.sparkContext,
                    rows,
                    cols,
                    numPartitions=num_partitions,
                    seed=seed,
                ).map(
                    lambda nparray: nparray.tolist()  # type: ignore
                ),
                schema=",".join(self.schema),
            )

            assert X.rdd.getNumPartitions() == num_partitions, (
                f"Unexpected num partitions received from RandomRDDs;"
                f"expected {num_partitions}, got {X.rdd.getNumPartitions()}"
            )

        # Generate ground truth upfront.
        if multinomial_log:
            ground_truth = np.zeros((cols, n_classes))
            ground_truth[:n_informative, :] = 100 * generator.uniform(
                size=(n_informative, n_classes)
            )
        else:
            ground_truth = np.zeros((cols, 1))
            ground_truth[:n_informative, :] = 100 * generator.uniform(
                size=(n_informative, 1)
            )

        if shuffle:
            # Shuffle feature indices upfront.
            col_indices = np.arange(cols)
            generator.shuffle(col_indices)
            if multinomial_log:
                ground_truth = ground_truth[col_indices]
            else:
                ground_truth = ground_truth[col_indices]

        # Create different partition seeds for sample generation.
        global_random_state = params["random_state"]

        # UDF for distributed generation of X and y.
        def make_regression_udf(iter: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
            use_cupy = use_gpu
            if use_cupy:
                try:
                    import cupy as cp
                except ImportError:
                    use_cupy = False
                    logging.warning("cupy import failed; falling back to numpy.")

            partition_index = pyspark.TaskContext().partitionId()
            my_seed = global_random_state + 100 * partition_index
            if use_cupy:
                generator_p = cp.random.RandomState(my_seed)
                ground_truth_cp = cp.asarray(ground_truth)
                col_indices_cp = cp.asarray(col_indices)
                bias_p = cp.asarray(bias)
            else:
                generator_p = np.random.RandomState(my_seed)
                bias_p = np.array(bias)

            for pdf in iter:
                if use_cupy:
                    X_p = cp.asarray(pdf.to_numpy())
                else:
                    X_p = pdf.to_numpy()

                if shuffle:
                    # Column-wise shuffle (global)
                    if use_cupy:
                        X_p[:, :] = X_p[:, col_indices_cp]
                    else:
                        X_p[:, :] = X_p[:, col_indices]

                # Label Calculation
                if use_cupy:
                    y = cp.dot(X_p, ground_truth_cp) + bias_p
                else:
                    y = np.dot(X_p, ground_truth) + bias_p

                if noise > 0.0:
                    y += generator_p.normal(scale=noise, size=y.shape)

                # Logistric Regression sigmoid and sample
                if logistic_regression:
                    if multinomial_log:
                        if use_cupy:
                            y = y.get()

                        probs = sp.special.softmax(y, axis=1)

                        if use_cupy:
                            probs_p = cp.asarray(probs)
                            cdf = cp.cumsum(probs_p, axis=1)
                            anchors = cp.random.random(size=(cdf.shape[0], 1))
                            y = cp.sum(anchors > cdf, axis=1)
                        else:
                            probs_p = probs
                            cdf = np.cumsum(probs_p, axis=1)
                            anchors = np.random.random(size=(cdf.shape[0], 1))
                            y = np.sum(anchors > cdf, axis=1)
                    else:
                        if use_cupy:
                            prob = 1 / (1 + cp.exp(-y))
                            y = cp.random.binomial(1, prob)
                        else:
                            prob = 1 / (1 + np.exp(-y))
                            y = np.random.binomial(1, prob)

                n_partition_rows = X_p.shape[0]
                if shuffle:
                    # Row-wise shuffle (partition)
                    if use_cupy:
                        row_indices = cp.random.permutation(n_partition_rows)
                        X_p = X_p[row_indices]
                        y = y[row_indices]
                    else:
                        X_p, y = util_shuffle(X_p, y, random_state=generator_p)

                if use_cupy:
                    y = cp.squeeze(y)
                    data = cp.concatenate(
                        (
                            X_p.astype(dtype),
                            y.reshape(n_partition_rows, 1).astype(dtype),
                        ),
                        axis=1,
                    ).get()
                else:
                    y = np.squeeze(y)
                    data = np.concatenate(
                        (
                            X_p.astype(dtype),
                            y.reshape(n_partition_rows, 1).astype(dtype),
                        ),
                        axis=1,
                    )
                del X_p
                del y
                yield pd.DataFrame(data=data)

        label_col = "label"
        self.schema.append(f"{label_col} {self.pyspark_type}")

        return (
            (X.mapInPandas(make_regression_udf, schema=",".join(self.schema))),
            self.feature_cols,
            np.squeeze(ground_truth),
        )


class SparseRegressionDataGen(DataGenBaseMeta):
    """Generate sparse regression dataset using a distributed version of sklearn.datasets.regression,
    including features and labels.
    """

    def __init__(self, argv: List[Any]) -> None:
        super().__init__()
        self._parse_arguments(argv)

    def _supported_extra_params(self) -> Dict[str, Any]:
        params = inspect_default_params_from_func(
            make_regression, ["n_samples", "n_features", "coef"]
        )
        # must replace the None to the correct type
        params["redundant_cols"] = int
        params["random_state"] = int
        params["density"] = float
        params["use_gpu"] = bool
        params["logistic_regression"] = bool
        params["density_curve"] = str
        params["n_classes"] = int
        params["n_chunk"] = int
        return params

    def gen_dataframe_and_meta(
        self, spark: SparkSession
    ) -> Tuple[DataFrame, List[str], np.ndarray]:
        dtype = self.dtype
        if dtype != np.float64:
            logging.warning(
                "Sparse VectorUDT in Spark Dataframe only support float64, would be auto-transformed"
            )

        params = self.extra_params

        if "random_state" not in params:
            # for reproducible dataset.
            params["random_state"] = 1

        print(f"Passing {params} to make_sparse_regression")

        rows = self.num_rows
        cols = self.num_cols
        orig_cols = self.num_cols
        assert self.args is not None
        num_partitions = self.args.output_num_files

        # Set num_partitions to Spark's default if output_num_files is not provided.
        if num_partitions is None:
            num_partitions = spark.sparkContext.defaultParallelism

        # Retrieve input params or set to defaults.
        seed = params["random_state"]
        generator = np.random.RandomState(seed)
        bias = params.get("bias", 0.0)
        noise = params.get("noise", 0.0)
        shuffle = params.get("shuffle", True)
        n_informative = params.get("n_informative", 10)
        n_targets = params.get("n_targets", 1)
        n_classes = params.get("n_classes", 2)
        n_chunks = params.get("n_chunk", 10)
        use_gpu = params.get("use_gpu", False)
        logistic_regression = params.get("logistic_regression", False)
        density = params.get("density", 0.1)
        redundant_cols = params.get("redundant_cols", 0)
        density_curve = params.get("density_curve", "None")

        # Check for multinomial logistic regression
        if logistic_regression and n_classes < 2:
            logging.warning(
                "Can not have logistic regression with 2 classes, default to 2"
            )
            n_classes = 2

        multinomial_log = logistic_regression and (n_classes > 2)

        # Check for bias for each class
        if multinomial_log:
            if isinstance(bias, list):
                if len(bias) < n_classes:
                    logging.warning(
                        "Insufficient bias number, setting all the the first entry"
                    )
                    bias = [bias[0] for i in range(n_classes)]
            else:
                bias = [bias for i in range(n_classes)]

        # Number of non_redundant columns
        orig_cols = cols - redundant_cols

        # Compute Density Curve
        density_values = np.array([])
        if density_curve != "None":
            if isinstance(density, List):
                logging.warning(
                    "Only the first density value will be used as density curve final target density"
                )
                density = density[0]

            if n_chunks > orig_cols:
                logging.warning(
                    "Not enough columns for the density chunks, setting chunk number to non-redundant column number"
                )
                n_chunks = orig_cols

            if density_curve == "Linear":
                density_values = np.linspace(num_partitions / rows, density, n_chunks)
                density_values *= n_chunks * density / sum(density_values)
            elif density_curve == "Exponential":
                density_values = np.logspace(
                    np.log10(num_partitions / rows), np.log10(density), n_chunks
                )
                density_values *= n_chunks * density / sum(density_values)
            else:
                logging.warning(
                    "Unsupported density curve, canceling density curve option",
                    density_curve,
                )
                density_curve = "None"
        else:
            if isinstance(density, List):
                density_values = np.array(density)
            else:
                density_values = np.array([density])

            n_chunks = len(density_values)

            avg_density = sum(density_values) // len(density_values)
            if (
                redundant_cols > 0
                and redundant_cols / cols > avg_density
                and density_curve == "None"
            ):
                logging.warning(
                    "Redundant columns would break density property, setting to zero instead"
                )
                redundant_cols = 0
                orig_cols = cols

        # Crop too large value
        for i in range(len(density_values)):
            if density_values[i] > 1:
                density_values[i] = 1

        # Generate ground truth upfront.
        if multinomial_log:
            ground_truth = np.zeros((cols, n_classes))
            ground_truth[:n_informative, :] = 100 * generator.uniform(
                size=(n_informative, n_classes)
            )
        else:
            ground_truth = np.zeros((cols, 1))
            ground_truth[:n_informative, :] = 100 * generator.uniform(
                size=(n_informative, 1)
            )

        if shuffle:
            # Shuffle feature indices upfront.
            col_indices = np.arange(cols)
            generator.shuffle(col_indices)
            ground_truth = ground_truth[col_indices]

        # Create different partition seeds for sample generation.
        global_random_seed = params["random_state"]

        # UDF for distributed generation of X and y.
        def make_sparse_regression_udf(
            iter: Iterable[pd.DataFrame],
        ) -> Iterable[pd.DataFrame]:
            use_cupy = use_gpu
            if use_cupy:
                try:
                    import cupy as cp
                except ImportError:
                    use_cupy = False
                    logging.warning("cupy import failed; falling back to numpy.")

            partition_index = pyspark.TaskContext().partitionId()

            # Get #rows in charge
            num_rows_per_partition = 0
            start = -1

            dfs = []
            for pdf in iter:
                num_rows_per_partition += pdf.shape[0]
                dfs.append(pdf)
                if start == -1:
                    start = pdf["id"][0]

            # Generate column by column if there is a density curve
            sparse_matrix = sp.sparse.csr_matrix((0, 0))

            col_per_chunk = np.full(n_chunks, orig_cols // n_chunks)
            col_per_chunk[: (orig_cols % n_chunks)] += 1

            for i in range(n_chunks):
                d = density_values[i]
                chunk_cols = col_per_chunk[i]

                # Generate a chunk
                sparse_col = sp.sparse.random(
                    num_rows_per_partition,
                    chunk_cols,
                    density=d,
                    random_state=generator,
                    format="csr",
                    dtype=dtype,
                    data_rvs=np.random.randn,
                )

                if sparse_matrix.shape[0] == 0:
                    sparse_matrix = sparse_col
                else:
                    sparse_matrix = sp.sparse.hstack(
                        [sparse_matrix, sparse_col]
                    ).tocsr()

            # Add in redundant cols of linear combinations of generated random sparse cols
            if redundant_cols > 0:
                # Generate random sparse matrix
                if density_curve == "None":
                    sparse_matrix = sp.sparse.random(
                        num_rows_per_partition,
                        orig_cols,
                        density=(density - redundant_cols / cols)
                        / (1 - redundant_cols / cols),
                        random_state=generator,
                        format="csr",
                        dtype=dtype,
                        data_rvs=np.random.randn,
                    )

                # Separate informative and non-informative columns
                if shuffle:
                    informative_shuffle_indices = np.arange(orig_cols)
                    sparse_matrix = sparse_matrix[:, informative_shuffle_indices]
                informative = sparse_matrix[:, :n_informative]

                if use_cupy:
                    redundant_mul = np.random.rand(n_informative, redundant_cols)
                else:
                    redundant_mul = np.random.rand(n_informative, redundant_cols)

                redundants = informative.dot(redundant_mul)
                sparse_matrix = sp.sparse.hstack([sparse_matrix, redundants]).tocsr()

            # Shuffle the matrix in scipy matrix with support to indexing
            if shuffle:
                sparse_matrix = sparse_matrix[:, col_indices]

            # Sort the csr matrix representation for better indices retrieval
            sparse_matrix.sum_duplicates()

            # Support parameters and library adaptation
            my_seed = global_random_seed + 100 * partition_index
            if use_cupy:
                generator_p = cp.random.RandomState(my_seed)
            else:
                generator_p = np.random.RandomState(my_seed)

            # Label Calculation
            y = sparse_matrix.dot(ground_truth) + bias

            # Type conversion
            if use_cupy:
                y_p = cp.asarray(y)
            else:
                y_p = np.asarray(y)

            # Random Noise
            if noise > 0.0:
                y_p += generator_p.normal(scale=noise, size=y.shape)

            # Logistric Regression sigmoid/softmax and sample
            if logistic_regression:
                if multinomial_log:
                    if use_cupy:
                        y = y_p.get()
                        del y_p

                    probs = sp.special.softmax(y, axis=1)

                    if use_cupy:
                        probs_p = cp.asarray(probs)
                        cdf = cp.cumsum(probs_p, axis=1)
                        anchors = cp.random.random(size=(cdf.shape[0], 1))
                        y = cp.sum(anchors > cdf, axis=1)
                    else:
                        probs_p = probs
                        cdf = np.cumsum(probs_p, axis=1)
                        anchors = np.random.random(size=(cdf.shape[0], 1))
                        y = np.sum(anchors > cdf, axis=1)
                else:
                    if use_cupy:
                        prob = 1 / (1 + cp.exp(-y_p))
                        del y_p
                        y = cp.random.binomial(1, prob)
                    else:
                        prob = 1 / (1 + np.exp(-y_p))
                        del y_p
                        y = np.random.binomial(1, prob)
            else:
                y = y_p

            if use_cupy:
                y = cp.squeeze(y).get()
            else:
                y = np.squeeze(y)

            for idx in dfs:
                truncated_idx = idx["id"].to_numpy() - start
                y_partial = y[truncated_idx]
                sparse_matrix_partial = sparse_matrix[truncated_idx]

                # Get indices and data for each row in csr format
                X_indices = np.split(
                    sparse_matrix_partial.indices, sparse_matrix_partial.indptr
                )[1:-1]
                X_data = np.split(
                    sparse_matrix_partial.data, sparse_matrix_partial.indptr
                )[1:-1]

                res = pd.DataFrame(
                    {"indices": X_indices, "values": X_data, "label": y_partial}
                )

                yield res

        label_col = "label"
        label_type = IntegerType() if logistic_regression else FloatType()

        vec_schema = StructType(
            [
                StructField("features", VectorUDT(), nullable=False),
                StructField(label_col, label_type, nullable=False),
            ]
        )

        schema = StructType(
            [
                StructField("indices", ArrayType(IntegerType()), nullable=False),
                StructField("values", ArrayType(FloatType()), nullable=False),
                StructField(label_col, label_type, nullable=False),
            ]
        )

        # Initial DataFrame with only row numbers
        init = spark.range(rows, numPartitions=num_partitions)

        res = init.mapInPandas(make_sparse_regression_udf, schema)

        # Map the indices and values back to a sparse vector
        vec_rdd = res.rdd.map(
            lambda row: (Vectors.sparse(cols, row[0], row[1]), row[2])
        )
        vec_res = vec_rdd.toDF(vec_schema)

        return (
            vec_res,
            self.feature_cols,
            np.squeeze(ground_truth),
        )


class ClassificationDataGen(DataGenBase):
    """Generate classification dataset using a distributed version of sklearn.datasets.classification,
    including features and labels."""

    def __init__(self, argv: List[Any]) -> None:
        super().__init__()
        self._parse_arguments(argv)

    def _supported_extra_params(self) -> Dict[str, Any]:
        params = inspect_default_params_from_func(
            make_classification, ["n_samples", "n_features", "weights"]
        )
        # must replace the None to the correct type
        params["random_state"] = int
        return params

    def gen_dataframe(self, spark: SparkSession) -> Tuple[DataFrame, List[str]]:
        dtype = self.dtype
        params = self.extra_params

        if "random_state" not in params:
            # for reproducible dataset.
            params["random_state"] = 1

        print(f"Passing {params} to make_classification")

        n_samples = self.num_rows
        n_features = self.num_cols
        assert self.args is not None
        num_partitions = self.args.output_num_files

        # Set num_partitions to Spark's default if output_num_files is not provided.
        if num_partitions is None:
            num_partitions = spark.sparkContext.defaultParallelism

        # For detailed parameter descriptions, see below:
        # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html

        # Retrieve input params or set to defaults.
        n_informative = params.get("n_informative", 2)
        n_redundant = params.get("n_redundant", 2)
        n_repeated = params.get("n_repeated", 0)
        n_classes = params.get("n_classes", 2)
        n_clusters_per_class = params.get("n_clusters_per_class", 2)
        flip_y = params.get("flip_y", 0.01)
        class_sep = params.get("class_sep", 1.0)
        hypercube = params.get("hypercube", True)
        shift = params.get("shift", 0.0)
        scale = params.get("scale", 1.0)
        shuffle = params.get("shuffle", True)
        generator = np.random.RandomState(params["random_state"])

        # Generate a random n-class classification problem.
        # Code adapted from sklearn.datasets.make_classification.

        # Check feature and cluster counts.
        if n_informative + n_redundant + n_repeated > n_features:
            raise ValueError(
                "Number of informative, redundant and repeated "
                "features must sum to less than the number of total"
                " features"
            )
        if n_informative < np.log2(
            n_classes * n_clusters_per_class
        ):  # log2 to avoid overflow errors
            msg = "n_classes({}) * n_clusters_per_class({}) must be"
            msg += " smaller or equal 2**n_informative({})={}"
            raise ValueError(
                msg.format(
                    n_classes, n_clusters_per_class, n_informative, 2**n_informative
                )
            )

        n_useless = n_features - n_informative - n_redundant - n_repeated
        n_clusters = n_classes * n_clusters_per_class

        # Distribute samples among clusters.
        n_samples_per_cluster = [n_samples // n_clusters] * n_clusters
        for i in range(n_samples - sum(n_samples_per_cluster)):
            n_samples_per_cluster[i % n_clusters] += 1

        # Distribute cluster samples among partitions.
        def distribute_samples(
            samples_per_cluster: List[int], num_partitions: int
        ) -> List[List[int]]:
            # Generates a list of num_partitions lists, each containing the samples to generate per cluster for that partition.
            num_clusters = len(samples_per_cluster)
            samples_per_partition = [[0] * num_clusters for _ in range(num_partitions)]
            for i, samples in enumerate(samples_per_cluster):
                quotient, remainder = divmod(samples, num_partitions)
                for j in range(num_partitions):
                    samples_per_partition[j][i] += quotient
                for j in range(remainder):
                    samples_per_partition[j][i] += 1
            return samples_per_partition

        n_samples_per_cluster_partition = distribute_samples(
            n_samples_per_cluster, num_partitions
        )

        # Build the polytope whose vertices become cluster centroids
        centroids = _generate_hypercube(n_clusters, n_informative, generator).astype(
            float, copy=False
        )
        centroids *= 2 * class_sep
        centroids -= class_sep
        if not hypercube:
            centroids *= generator.uniform(size=(n_clusters, 1))
            centroids *= generator.uniform(size=(1, n_informative))

        # Precompute covariance coefficients / noise parameters
        A = [
            2 * generator.uniform(size=(n_informative, n_informative)) - 1
            for _ in range(n_clusters)
        ]
        if n_redundant > 0:
            B = 2 * generator.uniform(size=(n_informative, n_redundant)) - 1
        if n_repeated > 0:
            n = n_informative + n_redundant
            repeat_indices = (
                (n - 1) * generator.uniform(size=n_repeated) + 0.5
            ).astype(np.intp)
        if shift is None:
            shift = (2 * generator.uniform(size=n_features) - 1) * class_sep
        if scale is None:
            scale = 1 + 100 * generator.uniform(size=n_features)
        if shuffle:
            shuffle_indices = np.arange(n_features)
            generator.shuffle(shuffle_indices)

        # Create different partition seeds for sample generation
        random.seed(params["random_state"])
        seed_maxval = 100 * num_partitions
        partition_seeds = random.sample(range(1, seed_maxval), num_partitions)

        maxRecordsPerBatch = int(
            spark.sparkContext.getConf().get(
                "spark.sql.execution.arrow.maxRecordsPerBatch", "10000"
            )
        )

        def make_classification_udf(
            iter: Iterable[pd.DataFrame],
        ) -> Iterable[pd.DataFrame]:
            for pdf in iter:
                partition_index = pdf.iloc[0][0]
                n_cluster_samples = n_samples_per_cluster_partition[partition_index]
                n_partition_samples = sum(n_cluster_samples)
                X_p = np.zeros((n_partition_samples, n_features))
                y = np.zeros(n_partition_samples, dtype=int)
                generator = np.random.RandomState(partition_seeds[partition_index])

                # Create informative features
                X_p[:, :n_informative] = generator.standard_normal(
                    size=(n_partition_samples, n_informative)
                )

                # Generate the samples per cluster for which this partition is responsible
                stop = 0
                for k, centroid in enumerate(centroids):
                    start, stop = stop, stop + n_cluster_samples[k]
                    y[start:stop] = k % n_classes  # assign labels
                    X_k = X_p[start:stop, :n_informative]  # slice a view of the cluster
                    X_k[...] = np.dot(X_k, A[k])  # introduce random covariance
                    X_k += centroid  # shift the cluster to vertex

                # Create redundant features
                if n_redundant > 0:
                    X_p[:, n_informative : n_informative + n_redundant] = np.dot(
                        X_p[:, :n_informative], B
                    )
                # Repeat some features
                if n_repeated > 0:
                    X_p[:, n : n + n_repeated] = X_p[:, repeat_indices]
                # Fill useless features
                if n_useless > 0:
                    X_p[:, -n_useless:] = generator.standard_normal(
                        size=(n_partition_samples, n_useless)
                    )
                # Randomly replace labels
                if flip_y >= 0.0:
                    flip_mask = generator.uniform(size=n_partition_samples) < flip_y
                    y[flip_mask] = generator.randint(n_classes, size=flip_mask.sum())
                # Randomly shift and scale
                X_p += shift
                X_p *= scale
                if shuffle:
                    X_p, y = util_shuffle(
                        X_p, y, random_state=generator
                    )  # Randomly permute samples
                    X_p[:, :] = X_p[:, shuffle_indices]  # Randomly permute features

                data = np.concatenate(
                    (
                        X_p.astype(dtype),
                        y.reshape(n_partition_samples, 1).astype(dtype),
                    ),
                    axis=1,
                )

                del X_p
                del y
                for i in range(0, n_partition_samples, maxRecordsPerBatch):
                    end_idx = min(i + maxRecordsPerBatch, n_partition_samples)
                    yield pd.DataFrame(data=data[i:end_idx])

        label_col = "label"
        self.schema.append(f"{label_col} {self.pyspark_type}")

        return (
            spark.range(0, num_partitions, numPartitions=num_partitions).mapInPandas(
                make_classification_udf, schema=",".join(self.schema)
            )
        ), self.feature_cols


if __name__ == "__main__":
    """
    See gen_data.main for more info.
    """

    registered_data_gens = {
        "blobs": BlobsDataGen,
        "default": DefaultDataGen,
        "low_rank_matrix": LowRankMatrixDataGen,
        "regression": RegressionDataGen,
        "classification": ClassificationDataGen,
        "sparse_regression": SparseRegressionDataGen,
    }

    main(registered_data_gens=registered_data_gens, repartition=False)
