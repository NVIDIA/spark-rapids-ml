from typing import Union

import numpy as np
from pyspark.ml.classification import (
    LogisticRegressionModel as SparkLogisticRegressionModel,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler
from pyspark.sql import DataFrame

from spark_rapids_ml.classification import LogisticRegressionModel


def logistic_regression_objective(
    df: DataFrame,
    lr_model: Union["LogisticRegressionModel", "SparkLogisticRegressionModel"],
) -> float:
    """can be used in testing and examples to calculate the full objective of a logistic regression model using Spark MLlib

    Args:
        df: DataFrame
        lr_model: Union[LogisticRegressionModel, SparkLogisticRegressionModel]

    Returns:
        Full objective of the logistic regression model:
        log_loss + reg_param * (0.5 * (1 - elasticnet_param) * ||coefs||_2^2 + elasticnet_param * |coefs|_1)
        where:
        log_loss = (1/n) * sum_i(-log(prob(y_i))) for labels y_1, y_2, ..., y_n
    """
    if isinstance(lr_model, LogisticRegressionModel):
        lr_model = lr_model.cpu()

    df_with_preds = lr_model.transform(df)

    prediction_col = lr_model.getPredictionCol()
    probability_col = lr_model.getProbabilityCol()
    label_name = lr_model.getLabelCol()
    features_col = lr_model.getFeaturesCol()

    evaluator = (
        MulticlassClassificationEvaluator()
        .setMetricName("logLoss")  # type:ignore
        .setPredictionCol(prediction_col)
        .setProbabilityCol(probability_col)
        .setLabelCol(label_name)
    )

    log_loss = evaluator.evaluate(df_with_preds)
    coefficients = (
        np.array(lr_model.coefficients)
        if lr_model.numClasses == 2
        else lr_model.coefficientMatrix.toArray()
    )

    # account for effects of standardization on the coefficients for regularization penalty
    if lr_model.getStandardization() is True:
        column_names = df.columns
        outputCol = "_objective_tmp"
        while outputCol in column_names:
            outputCol = "_" + outputCol

        scaler = StandardScaler(
            inputCol=features_col,
            outputCol=outputCol,
        )
        scaler_model = scaler.fit(df)
        stdev = np.array(scaler_model.std)
        coefficients = coefficients * stdev

    coefs_l1 = np.sum(np.abs(coefficients))
    coefs_l2 = np.sum(coefficients**2)

    elasticnet_param = lr_model.getElasticNetParam()
    full_objective = log_loss + lr_model.getRegParam() * (
        0.5 * (1 - elasticnet_param) * coefs_l2 + elasticnet_param * coefs_l1
    )

    return full_objective
