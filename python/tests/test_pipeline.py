#
# Copyright (c) 2025, NVIDIA CORPORATION.
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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

if TYPE_CHECKING:
    from pyspark.ml._typing import PipelineStage

import numpy as np
import pyspark.ml.functions as MLF
import pyspark.sql.functions as SQLF
import pytest
from pyspark.ml import Pipeline as SparkPipeline
from pyspark.ml import PipelineModel as SparkPipelineModel
from pyspark.ml.base import Estimator, Transformer
from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression
from pyspark.ml.classification import (
    LogisticRegressionModel as SparkLogisticRegressionModel,
)
from pyspark.ml.feature import HashingTF, Tokenizer, VectorAssembler
from pyspark.ml.tuning import CrossValidator as SparkCrossValidator
from pyspark.sql import DataFrame, SparkSession

from spark_rapids_ml.classification import LogisticRegression as GPULogisticRegression
from spark_rapids_ml.classification import (
    LogisticRegressionModel as GPULogisticRegressionModel,
)
from spark_rapids_ml.core import _CumlEstimator
from spark_rapids_ml.pipeline import NoOpTransformer, Pipeline
from spark_rapids_ml.tuning import CrossValidator
from spark_rapids_ml.utils import getInputOrFeaturesCols, setInputOrFeaturesCol

from .utils import array_equal, create_pyspark_dataframe, make_classification_dataset


def create_toy_dataframe(
    spark: SparkSession, all_scalar_columns: bool = True
) -> Tuple[DataFrame, List[str], str]:
    import numpy as np

    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [2.0, 1.0],
            [3.0, 1.0],
        ]
    )

    y = np.array(
        [
            1.0,
            1.0,
            0.0,
            0.0,
        ]
    )

    (df, feature_cols, label_col) = create_pyspark_dataframe(
        spark, "multi_cols", np.dtype("float32"), X, y
    )

    if all_scalar_columns is False:
        col_name = feature_cols[0]

        df = df.withColumn("c0_vec", MLF.array_to_vector(SQLF.array([col_name])))
        df = df.drop(col_name)
        df = df.withColumnRenamed("c0_vec", col_name)

    return (df, feature_cols, label_col)  # type: ignore


def check_two_stage_pipeline(
    pipeline: SparkPipeline,
    pipeline_model: SparkPipelineModel,
    assembler: VectorAssembler,
    Est: Type,
    Model: Type,
) -> None:
    assert len(pipeline.getStages()) == 2
    assert len(pipeline_model.stages) == 2

    def check_va_stage(t_stage: "PipelineStage") -> None:
        assert isinstance(t_stage, VectorAssembler)
        assert t_stage == assembler
        assert t_stage.getInputCols() == assembler.getInputCols()
        assert t_stage.getOutputCol() == assembler.getOutputCol()

    check_va_stage(pipeline.getStages()[0])
    check_va_stage(pipeline_model.stages[0])

    assert isinstance(pipeline.getStages()[1], Est)
    assert getInputOrFeaturesCols(pipeline.getStages()[1]) == assembler.getOutputCol()

    assert isinstance(pipeline_model.stages[1], Model)
    assert getInputOrFeaturesCols(pipeline_model.stages[1]) == assembler.getOutputCol()


@pytest.mark.parametrize(
    "PipelineEst,Est,Model",
    [
        (Pipeline, GPULogisticRegression, GPULogisticRegressionModel),
        (Pipeline, SparkLogisticRegression, SparkLogisticRegressionModel),
        (SparkPipeline, GPULogisticRegression, GPULogisticRegressionModel),
        (SparkPipeline, SparkLogisticRegression, SparkLogisticRegressionModel),
    ],
)
def test_example(
    PipelineEst: Type,
    Est: Type,
    Model: Type,
    caplog: pytest.LogCaptureFixture,
    all_scalar_colum: bool = True,
) -> None:

    from .conftest import _spark

    (df, input_cols, label_col) = create_toy_dataframe(_spark, all_scalar_colum)
    assembler = VectorAssembler(inputCols=input_cols, outputCol="features")

    algo_params = {
        "featuresCol": "features",
        "labelCol": label_col,
        "maxIter": 10,
        "regParam": 0.001,
    }
    est = Est(**algo_params)

    pipeline = PipelineEst(stages=[assembler, est])
    pipeline_model = pipeline.fit(df)

    info_msg = "Spark Rapids ML pipeline bypasses VectorAssembler for GPU-based estimators to achieve optimal performance."
    if PipelineEst is Pipeline and all_scalar_colum and isinstance(est, _CumlEstimator):
        assert info_msg in caplog.text
    else:
        assert info_msg not in caplog.text

    check_two_stage_pipeline(pipeline, pipeline_model, assembler, Est, Model)

    # ensure model correctness
    spark_lr = SparkLogisticRegression(**algo_params)  # type: ignore
    df_udt = assembler.transform(df)
    spark_model = spark_lr.fit(df_udt)

    from .test_logistic_regression import compare_model

    compare_model(pipeline_model.stages[1], spark_model, df_udt, y_true_col=label_col)


def test_mixed_columns(caplog: pytest.LogCaptureFixture) -> None:
    test_example(
        Pipeline,
        GPULogisticRegression,
        GPULogisticRegressionModel,
        caplog=caplog,
        all_scalar_colum=False,
    )


@pytest.mark.parametrize(
    "PipelineEst,Est,Model,CVEst",
    [
        (Pipeline, GPULogisticRegression, GPULogisticRegressionModel, CrossValidator),
        (
            SparkPipeline,
            SparkLogisticRegression,
            SparkLogisticRegressionModel,
            SparkCrossValidator,
        ),
    ],
)
def test_compat_cv(
    PipelineEst: Type,
    Est: Type,
    Model: Type,
    CVEst: Type,
    caplog: pytest.LogCaptureFixture,
) -> None:

    from .conftest import _spark

    X_train, X_test, y_train, y_test = make_classification_dataset(
        datatype=np.dtype(np.float32),
        nrows=20000,
        ncols=20,
        n_classes=2,
        n_informative=8,
        n_redundant=0,
        n_repeated=0,
    )

    (df, feature_cols, label_col) = create_pyspark_dataframe(
        _spark, "multi_cols", np.dtype("float32"), X_train, y_train
    )

    assert isinstance(feature_cols, List)
    assert all(isinstance(col, str) for col in feature_cols)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    algo_params = {
        "featuresCol": "features",
        "labelCol": label_col,
        "maxIter": 10,
    }
    cv_regParam_list = [0.01, 0.05]

    lr = Est(**algo_params)

    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml.tuning import ParamGridBuilder

    grid = ParamGridBuilder().addGrid(lr.regParam, cv_regParam_list).build()

    evaluator = MulticlassClassificationEvaluator(
        metricName="logLoss", labelCol=label_col  # type: ignore
    )
    cv = CVEst(
        estimator=lr,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        parallelism=1,
        seed=1,
        numFolds=2,
    )

    pipeline = PipelineEst(stages=[assembler, cv])
    pipeline_model = pipeline.fit(df)

    cv_model = pipeline_model.stages[1]
    print(cv_model.avgMetrics)

    assert array_equal(
        cv_model.avgMetrics,
        [0.2373587566355036, 0.27749137873167856],
    )

    info_msg = "Spark Rapids ML pipeline bypasses VectorAssembler for GPU-based estimators to achieve optimal performance."
    if PipelineEst == Pipeline and isinstance(lr, _CumlEstimator):
        assert info_msg in caplog.text
    else:
        assert info_msg not in caplog.text


from pyspark.ml.clustering import KMeans as SparkKmeans
from pyspark.ml.clustering import KMeansModel as SparkKMeansModel

from spark_rapids_ml.clustering import KMeans, KMeansModel


@pytest.mark.parametrize(
    "PipelineEst,Est,Model,algo_params",
    [
        (Pipeline, KMeans, KMeansModel, {"maxIter": 10}),
        (SparkPipeline, SparkKmeans, SparkKMeansModel, {"maxIter": 10}),
    ],
)
def test_compat_est(
    PipelineEst: Type,
    Est: Type,
    Model: Type,
    algo_params: Dict[str, Any],
    caplog: pytest.LogCaptureFixture,
    all_scalar_colum: bool = True,
) -> None:

    from .conftest import _spark

    (df, input_cols, label_col) = create_toy_dataframe(_spark, all_scalar_colum)
    output_col = "features"
    assembler = VectorAssembler(inputCols=input_cols, outputCol=output_col)

    est = Est(**algo_params)

    setInputOrFeaturesCol(est, "features", label_col)

    pipeline = PipelineEst(stages=[assembler, est])
    pipeline_model = pipeline.fit(df)

    info_msg = "Spark Rapids ML pipeline bypasses VectorAssembler for GPU-based estimators to achieve optimal performance."
    if PipelineEst is Pipeline and all_scalar_colum and isinstance(est, _CumlEstimator):
        assert info_msg in caplog.text

    else:
        assert info_msg not in caplog.text

    check_two_stage_pipeline(pipeline, pipeline_model, assembler, Est, Model)


from pyspark.ml.feature import PCA as SparkPCA
from pyspark.ml.feature import PCAModel as SparkPCAModel

from spark_rapids_ml.feature import PCA, PCAModel


@pytest.mark.parametrize(
    "PipelineEst,Est,Model,algo_params",
    [
        (Pipeline, PCA, PCAModel, {"k": 1}),
        (SparkPipeline, SparkPCA, SparkPCAModel, {"k": 1}),
    ],
)
def test_compat_pca(
    PipelineEst: Type,
    Est: Type,
    Model: Type,
    algo_params: Dict[str, Any],
    caplog: pytest.LogCaptureFixture,
    all_scalar_colum: bool = True,
) -> None:
    test_compat_est(PipelineEst, Est, Model, algo_params, caplog, all_scalar_colum)


from pyspark.ml.regression import LinearRegression as SparkLinearRegression
from pyspark.ml.regression import LinearRegressionModel as SparkLinearRegressionModel

from spark_rapids_ml.regression import LinearRegression, LinearRegressionModel


@pytest.mark.parametrize(
    "PipelineEst,Est,Model,algo_params",
    [
        (Pipeline, LinearRegression, LinearRegressionModel, {"regParam": 0.01}),
        (
            SparkPipeline,
            SparkLinearRegression,
            SparkLinearRegressionModel,
            {"regParam": 0.01},
        ),
    ],
)
def test_compat_linear_regression(
    PipelineEst: Type,
    Est: Type,
    Model: Type,
    algo_params: Dict[str, Any],
    caplog: pytest.LogCaptureFixture,
    all_scalar_colum: bool = True,
) -> None:
    test_compat_est(PipelineEst, Est, Model, algo_params, caplog, all_scalar_colum)


from pyspark.ml.classification import (
    RandomForestClassificationModel as SparkRandomForestClassificationModel,
)
from pyspark.ml.classification import (
    RandomForestClassifier as SparkRandomForestClassifier,
)
from pyspark.ml.regression import (
    RandomForestRegressionModel as SparkRandomForestRegressionModel,
)
from pyspark.ml.regression import RandomForestRegressor as SparkRandomForestRegressor

from spark_rapids_ml.classification import (
    RandomForestClassificationModel,
    RandomForestClassifier,
)
from spark_rapids_ml.regression import (
    RandomForestRegressionModel,
    RandomForestRegressor,
)


@pytest.mark.parametrize(
    "PipelineEst,Est,Model,algo_params",
    [
        (
            Pipeline,
            RandomForestClassifier,
            RandomForestClassificationModel,
            {"maxDepth": 3},
        ),
        (
            SparkPipeline,
            SparkRandomForestClassifier,
            SparkRandomForestClassificationModel,
            {"maxDepth": 3},
        ),
        (Pipeline, RandomForestRegressor, RandomForestRegressionModel, {"maxDepth": 3}),
        (
            SparkPipeline,
            SparkRandomForestRegressor,
            SparkRandomForestRegressionModel,
            {"maxDepth": 3},
        ),
    ],
)
def test_compat_random_forest(
    PipelineEst: Type,
    Est: Type,
    Model: Type,
    algo_params: Dict[str, Any],
    caplog: pytest.LogCaptureFixture,
    all_scalar_colum: bool = True,
) -> None:
    test_compat_est(PipelineEst, Est, Model, algo_params, caplog, all_scalar_colum)


from spark_rapids_ml.clustering import DBSCAN, DBSCANModel
from spark_rapids_ml.umap import UMAP, UMAPModel


@pytest.mark.parametrize(
    "PipelineEst,Est,Model,algo_params",
    [
        (
            Pipeline,
            UMAP,
            UMAPModel,
            {"n_components": 1},
        ),
        (
            Pipeline,
            DBSCAN,
            DBSCANModel,
            {"eps": 0.01, "min_samples": 1},
        ),
    ],
)
def test_non_spark_algo(
    PipelineEst: Type,
    Est: Type,
    Model: Type,
    algo_params: Dict[str, Any],
    caplog: pytest.LogCaptureFixture,
    all_scalar_colum: bool = True,
) -> None:
    test_compat_est(PipelineEst, Est, Model, algo_params, caplog, all_scalar_colum)


@pytest.mark.parametrize("handleInvalid", ["skip", "keep"])
def test_handleinvalid(handleInvalid: str, caplog: pytest.LogCaptureFixture) -> None:
    from .conftest import _spark

    (df, input_cols, label_col) = create_toy_dataframe(_spark, all_scalar_columns=True)
    assembler = VectorAssembler(
        inputCols=input_cols, outputCol="features", handleInvalid=handleInvalid
    )

    algo_params: Dict[str, Any] = {
        "featuresCol": "features",
        "labelCol": label_col,
        "maxIter": 10,
        "regParam": 0.001,
    }

    est = GPULogisticRegression(**algo_params)

    pipeline = Pipeline(stages=[assembler, est])
    pipeline_model = pipeline.fit(df)

    info_msg = "Spark Rapids ML pipeline bypasses VectorAssembler for GPU-based estimators to achieve optimal performance."
    assert info_msg not in caplog.text


def test_compact_linear_regression_with_unsupported_gpu_param() -> None:
    from .sparksession import CleanSparkSession

    conf = {"spark.rapids.ml.cpu.fallback.enabled": True}

    with CleanSparkSession(conf) as spark:
        (df, input_cols, label_col) = create_toy_dataframe(
            spark, all_scalar_columns=True
        )
        features_col = "features"

        assembler = VectorAssembler(outputCol=features_col, inputCols=input_cols)
        algo_params: Dict[str, Any] = {
            "labelCol": label_col,
            "featuresCol": features_col,
            "solver": "l-bfgs",
        }
        est = LinearRegression(**algo_params)
        pipeline = Pipeline(stages=[assembler, est])
        model = pipeline.fit(df)
        assert isinstance(model.stages[1], SparkLinearRegressionModel)
