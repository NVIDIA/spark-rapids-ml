#
# Copyright (c) 2022-2026, NVIDIA CORPORATION.
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

import cudf
import numpy as np
from cuml.cluster import KMeans as CumlKMeans
from gen_data_distributed import BlobsDataGen

from spark_rapids_ml.clustering import KMeans

from .conftest import _spark


def test_kmeans_large_model_exceeds_int32_max() -> None:
    """
    Test KMeans with 100,000 centers and 4,000 dimensions per center.
    The resulting model size = 100,000 * 4,000 * 8 bytes (DoubleType) = 3.2GB,
    which exceeds Spark's default driver maxResultSize (2GB), as well as the BufferHolder
    size limit of 2,147,483,632 bytes. To collect this large model to the driver,
    spark.driver.maxResultSize must be set to 0 (unlimited), and Spark payload buffer sizes
    must be sufficient to handle the large model.
    """
    gpu_number = 1
    n_centers = 100_000
    n_dimensions = 4_000
    n_samples = 120_000  # need more samples than centers for k-means
    output_num_files = 12  # partitions for distributed data gen

    data_gen_args = [
        "--num_rows",
        str(n_samples),
        "--num_cols",
        str(n_dimensions),
        "--centers",
        "10",
        "--output_num_files",
        str(output_num_files),
        "--dtype",
        "float32",
        "--output_dir",
        "./temp",
        "--random_state",
        "42",
    ]
    data_gen = BlobsDataGen(data_gen_args)
    df, feature_cols, _ = data_gen.gen_dataframe_and_meta(_spark)
    df = df.cache()

    kmeans = KMeans(
        num_workers=gpu_number,
        k=n_centers,
        max_iter=2,
        initMode="random",
        seed=42,
    ).setFeaturesCols(feature_cols)

    kmeans_model = kmeans.fit(df)

    # Verify model shape
    assert len(kmeans_model.cluster_centers_) == n_centers
    assert len(kmeans_model.cluster_centers_[0]) == n_dimensions
    assert kmeans_model.n_cols == n_dimensions

    # Transform
    pred_col = kmeans_model.getPredictionCol()
    predictions = kmeans_model.transform(df)
    assert predictions.count() == n_samples
    pred_max = predictions.agg({pred_col: "max"}).head()[0]
    assert pred_max < n_centers
