from typing import Any, Dict, Tuple, Type, TypeVar

import numpy as np
import pytest
from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression
from pyspark.ml.classification import (
    LogisticRegressionModel as SparkLogisticRegressionModel,
)
from pyspark.ml.functions import array_to_vector
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
from pyspark.sql.functions import array, col

from spark_rapids_ml.classification import LogisticRegression, LogisticRegressionModel
from .sparksession import CleanSparkSession
from .utils import (
    create_pyspark_dataframe,
    idfn,
    make_classification_dataset,
    array_equal
)


def test_toy_example(gpu_number: int) -> None:
    # reduce the number of GPUs for toy dataset to avoid empty partition
    gpu_number = min(gpu_number, 2)
    data = [
        ([1.0, 2.0], 1.0),
        ([1.0, 3.0], 1.0),
        ([2.0, 1.0], 0.0),
        ([3.0, 1.0], 0.0),
    ]

    with CleanSparkSession() as spark:
        features_col = "features"
        label_col = "label"
        schema = features_col + " array<float>, " + label_col + " float"
        df = spark.createDataFrame(data, schema=schema)

        lr_estimator = LogisticRegression(num_workers=gpu_number)
        lr_estimator.setFeaturesCol(features_col)
        lr_estimator.setLabelCol(label_col)
        lr_model = lr_estimator.fit(df)

        assert lr_model.n_cols == 2
        assert lr_model.dtype == "float32"

        assert len(lr_model.coef_) == 1
        assert lr_model.coef_[0] == pytest.approx([-0.71483153, 0.7148315], abs=1e-6)
        assert lr_model.intercept_ == pytest.approx([-2.2614916e-08], abs=1e-6)

        assert lr_model.coefficients.toArray() == pytest.approx(
            [-0.71483153, 0.7148315], abs=1e-6
        )
        assert lr_model.intercept == pytest.approx(-2.2614916e-08, abs=1e-6)

        preds_df = lr_model.transform(df)
        preds = [ row["prediction"] for row in preds_df.collect()]
        assert preds == [1., 1., 0., 0.]


# TODO support float64
# 'vector' will be converted to float64 so It depends on float64 support  
@pytest.mark.parametrize("feature_type", ["array", "multi_cols"])  
@pytest.mark.parametrize("data_shape", [(2000, 8)], ids=idfn)
@pytest.mark.parametrize("data_type", [np.float32])  
@pytest.mark.parametrize("max_record_batch", [100, 10000]) 
@pytest.mark.parametrize("n_classes", [2])
@pytest.mark.slow
def test_classifier(
    feature_type: str,
    data_shape: Tuple[int, int],
    data_type: np.dtype,
    max_record_batch: int,
    n_classes: int,
    gpu_number: int,
) -> None:
    tolerance = 0.001

    X_train, X_test, y_train, y_test = make_classification_dataset(
        datatype=data_type,
        nrows=data_shape[0],
        ncols=data_shape[1],
        n_classes=n_classes,
        n_informative=8,
        n_redundant=0,
        n_repeated=0,
    )

    from cuml import LogisticRegression as cuLR

    cu_lr = cuLR()
    cu_lr.fit(X_train, y_train)

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_record_batch)}
    with CleanSparkSession(conf) as spark:
        train_df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X_train, y_train
        )

        assert label_col is not None
        spark_lr = LogisticRegression(
            num_workers=gpu_number,
        )
        spark_lr.setFeaturesCol(features_col)
        spark_lr.setLabelCol(label_col)
        spark_lr_model: LogisticRegressionModel = spark_lr.fit(train_df)

        assert spark_lr_model.n_cols == cu_lr.n_cols
        assert spark_lr_model.dtype == cu_lr.dtype
        assert len(spark_lr_model.coef_) == len(cu_lr.coef_)
        for i in range(len(spark_lr_model.coef_)):
            assert spark_lr_model.coef_[i] == pytest.approx(cu_lr.coef_[i], tolerance)
        assert spark_lr_model.intercept_ == pytest.approx(cu_lr.intercept_, tolerance)

        # test coefficients and intercepts
        assert spark_lr_model.n_cols == cu_lr.n_cols
        assert spark_lr_model.dtype == cu_lr.dtype

        assert len(spark_lr_model.coef_) == 1
        assert len(cu_lr.coef_) == 1
        assert array_equal(spark_lr_model.coef_[0], cu_lr.coef_[0], tolerance)
        assert array_equal(spark_lr_model.intercept_, cu_lr.intercept_, tolerance)

        assert array_equal(
            spark_lr_model.coefficients.toArray(), cu_lr.coef_[0], tolerance
        )
        assert spark_lr_model.intercept == pytest.approx(cu_lr.intercept_[0], tolerance)

        # test transform
        test_df, _, _ = create_pyspark_dataframe(spark, feature_type, data_type, X_test)

        result = spark_lr_model.transform(test_df).collect()
        spark_preds = [row["prediction"] for row in result]
        cu_preds = cu_lr.predict(X_test)
        assert array_equal(cu_preds, spark_preds, 1e-3)


LogisticRegressionType = TypeVar(
    "LogisticRegressionType", Type[LogisticRegression], Type[SparkLogisticRegression]
)
LogisticRegressionModelType = TypeVar(
    "LogisticRegressionModelType",
    Type[LogisticRegressionModel],
    Type[SparkLogisticRegressionModel],
)


@pytest.mark.compat
@pytest.mark.parametrize(
    "lr_types",
    [
        (SparkLogisticRegression, SparkLogisticRegressionModel),
        (LogisticRegression, LogisticRegressionModel),
    ],
)
def test_compat(
    lr_types: Tuple[LogisticRegressionType, LogisticRegressionModelType],
    tmp_path: str,
) -> None:
    _LogisticRegression, _LogisticRegressionModel = lr_types

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
    num_rows = len(X)

    weight = np.ones([num_rows])
    feature_cols = ["c0", "c1"]
    schema = ["c0 float, c1 float, weight float, label float"]

    with CleanSparkSession() as spark:
        np_array = np.concatenate(
            (X, weight.reshape(num_rows, 1), y.reshape(num_rows, 1)), axis=1
        )

        df = spark.createDataFrame(
            np_array.tolist(),
            ",".join(schema),
        )

        df = df.withColumn("features", array_to_vector(array(*feature_cols))).drop(
            *feature_cols
        )

        lr = _LogisticRegression()
        # lr = _LogisticRegression(regParam=0.1, standardization=False)
        # assert lr.getRegParam() == 0.1

        lr.setFeaturesCol("features")
        # lr.setMaxIter(30)
        # lr.setRegParam(0.01)
        lr.setLabelCol("label")
        if isinstance(lr, SparkLogisticRegression):
            lr.setWeightCol("weight")

        assert lr.getFeaturesCol() == "features"
        # assert lr.getMaxIter() == 30
        # assert lr.getRegParam() == 0.01
        assert lr.getLabelCol() == "label"

        model = lr.fit(df)
        coefficients = model.coefficients.toArray()
        intercept = model.intercept

        if isinstance(lr, SparkLogisticRegression):
            assert array_equal(coefficients, [-17.65543489, 17.65543489])
            assert intercept == pytest.approx(0, abs=1e-6)
        else:
            assert array_equal(coefficients, [-0.71483159, 0.71483147])
            assert intercept == pytest.approx(0, abs=1e-6)

        # example = df.head()
        # if example:
        #     model.predict(example.features)

        # model.setPredictionCol("prediction")
        # output = model.transform(df).head()
        ## Row(weight=1.0, label=2.0374512672424316, features=DenseVector([-0.2052, 1.4941]), prediction=2.037452415464224)
        # assert np.isclose(output.prediction, 2.037452415464224)

        lr_path = tmp_path + "/log_reg"
        lr.save(lr_path)

        lr2 = _LogisticRegression.load(lr_path)
        # assert lr2.getMaxIter() == 5

        model_path = tmp_path + "/log_reg_model"
        model.save(model_path)

        model2 = _LogisticRegressionModel.load(model_path)
        assert array_equal(model.coefficients.toArray(), model2.coefficients.toArray())
        assert model.intercept == model2.intercept
        # assert model.transform(df).take(1) == model2.transform(df).take(1)
        # assert model.numFeatures == 2

