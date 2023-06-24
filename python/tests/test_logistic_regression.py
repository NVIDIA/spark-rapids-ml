from .sparksession import CleanSparkSession
from spark_rapids_ml.classification import LogisticRegression
import pytest

def test_toy_example(gpu_number: int) -> None:
    # reduce the number of GPUs for toy dataset to avoid empty partition
    gpu_number = min(gpu_number, 2)
    data = [
        ([1., 2.], 1.),
        ([1., 3.], 1.),
        ([2., 1.], 0.),
        ([3., 1.], 0.),
    ]

    with CleanSparkSession() as spark:
        features_col = "features"
        label_col = "label"
        schema = features_col + " array<float>, " + label_col + " float" 
        df = spark.createDataFrame(data, schema=schema)
        df.show()
        lr_estimator = LogisticRegression(num_workers=gpu_number)
        lr_estimator.setFeaturesCol(features_col)
        lr_estimator.setLabelCol(label_col)
        lr_model = lr_estimator.fit(df)

        assert len(lr_model.coef_) == 1 
        assert lr_model.coef_[0] == pytest.approx([-0.71483153, 0.7148315], abs=1e-6)
        assert lr_model.intercept_ == pytest.approx([-2.2614916e-08], abs=1e-6)
        assert lr_model.n_cols == 2
        assert lr_model.dtype == "float32"
        
    #from cuml import LogisticRegression as CuLogisticRegression
    #cuml_lr = CuLogisticRegression()
    #cuml_lr.fit(X, y)


@pytest.mark.parametrize("feature_type", pyspark_supported_feature_types)
@pytest.mark.parametrize("data_shape", [(2000, 8)], ids=idfn)
@pytest.mark.parametrize("data_type", cuml_supported_data_types)
@pytest.mark.parametrize("max_record_batch", [100, 10000])
@pytest.mark.parametrize("n_classes", [2, 4])
@pytest.mark.parametrize("num_workers", num_workers)
@pytest.mark.slow
def test_random_forest_classifier(
    feature_type: str,
    data_shape: Tuple[int, int],
    data_type: np.dtype,
    max_record_batch: int,
    n_classes: int,
    num_workers: int,
) -> None:
    X_train, X_test, y_train, y_test = make_classification_dataset(
        datatype=data_type,
        nrows=data_shape[0],
        ncols=data_shape[1],
        n_classes=n_classes,
        n_informative=8,
        n_redundant=0,
        n_repeated=0,
    )

    rf_params: Dict[str, Any] = {
        "n_estimators": 100,
        "n_bins": 128,
        "max_depth": 16,
        "bootstrap": False,
        "max_features": 1.0,
    }

    from cuml import RandomForestClassifier as cuRf

    cu_rf = cuRf(n_streams=1, **rf_params)
    cu_rf.fit(X_train, y_train)
    cu_preds = cu_rf.predict(X_test)
    cu_preds_proba = cu_rf.predict_proba(X_test)

    cu_acc = accuracy_score(y_test, cu_preds)

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_record_batch)}
    with CleanSparkSession(conf) as spark:
        train_df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X_train, y_train
        )

        assert label_col is not None
        spark_rf = RandomForestClassifier(
            num_workers=num_workers,
            **rf_params,
        )
        spark_rf.setFeaturesCol(features_col)
        spark_rf.setLabelCol(label_col)
        spark_rf_model: RandomForestClassificationModel = spark_rf.fit(train_df)

        test_df, _, _ = create_pyspark_dataframe(
            spark, feature_type, data_type, X_test, y_test
        )

        result = spark_rf_model.transform(test_df).collect()
        pred_result = [row.prediction for row in result]

        if feature_type == feature_types.vector:
            # no need to compare all feature type.
            spark_cpu_result = spark_rf_model.cpu().transform(test_df).collect()
            spark_cpu_pred_result = [row.prediction for row in spark_cpu_result]
            assert array_equal(spark_cpu_pred_result, pred_result)

        spark_acc = accuracy_score(y_test, np.array(pred_result))

        # Since vector type will force to convert to array<double>
        # which may cause precision issue for random forest.
        if num_workers == 1 and not (
            data_type == np.float32 and feature_type == feature_types.vector
        ):
            assert cu_acc == spark_acc

            pred_proba_result = [row.probability for row in result]
            np.testing.assert_allclose(pred_proba_result, cu_preds_proba, rtol=1e-3)
        else:
            assert cu_acc - spark_acc < 0.07

        # for multi-class classification evaluation
        if n_classes > 2:
            from pyspark.ml.evaluation import MulticlassClassificationEvaluator

            evaluator = MulticlassClassificationEvaluator(
                predictionCol=spark_rf_model.getPredictionCol(),
                labelCol=spark_rf_model.getLabelCol(),
            )

            spark_cuml_f1_score = spark_rf_model._transformEvaluate(test_df, evaluator)

            transformed_df = spark_rf_model.transform(test_df)
            pyspark_f1_score = evaluator.evaluate(transformed_df)

            assert math.fabs(pyspark_f1_score - spark_cuml_f1_score[0]) < 1e-6