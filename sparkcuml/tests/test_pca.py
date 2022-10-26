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

import pytest
from pyspark.sql import SparkSession

from sparkcuml.decomposition import SparkCumlPCA


def test_fit(spark: SparkSession, gpu_number: int) -> None:
    data = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]

    topk = 1

    rdd = spark.sparkContext.parallelize(data).map(lambda row: (row,))

    df = rdd.toDF(["features"])

    gpu_pca = SparkCumlPCA(num_workers=gpu_number).setInputCol("features").setK(topk)

    gpu_model = gpu_pca.fit(df)

    assert gpu_model.getInputCol() == "features"

    assert len(gpu_model.mean) == 2
    assert gpu_model.mean[0] == pytest.approx(2.0, 0.001)
    assert gpu_model.mean[1] == pytest.approx(2.0, 0.001)

    assert len(gpu_model.pc) == 1
    assert len(gpu_model.pc[0]) == 2
    assert gpu_model.pc[0][0] == pytest.approx(0.707, 0.001)
    assert gpu_model.pc[0][1] == pytest.approx(0.707, 0.001)

    assert len(gpu_model.explained_variance) == 1
    assert gpu_model.explained_variance[0] == pytest.approx(2.0, 0.001)
