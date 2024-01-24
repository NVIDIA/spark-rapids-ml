#
# Copyright (c) 2024, NVIDIA CORPORATION.
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
from typing import Tuple, Union

import numpy as np
import pytest
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidatorModel, ParamGridBuilder

from spark_rapids_ml.regression import RandomForestRegressor
from spark_rapids_ml.tuning import CrossValidator

from .sparksession import CleanSparkSession
from .utils import (
    create_pyspark_dataframe,
    feature_types,
    idfn,
    make_regression_dataset,
)


@pytest.mark.parametrize("feature_type", [feature_types.vector])
@pytest.mark.parametrize("data_type", [np.float32])
@pytest.mark.parametrize("data_shape", [(100, 8)], ids=idfn)
def test_crossvalidator(
    tmp_path: str,
    feature_type: str,
    data_type: np.dtype,
    data_shape: Tuple[int, int],
) -> None:
    X, _, y, _ = make_regression_dataset(
        datatype=data_type,
        nrows=data_shape[0],
        ncols=data_shape[1],
    )

    with CleanSparkSession() as spark:
        df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X, y
        )
        assert label_col is not None

        rfc = RandomForestRegressor()
        rfc.setFeaturesCol(features_col)
        rfc.setLabelCol(label_col)

        evaluator = RegressionEvaluator()
        evaluator.setLabelCol(label_col)

        grid = ParamGridBuilder().addGrid(rfc.maxBins, [3, 5]).build()

        cv = CrossValidator(
            estimator=rfc,
            estimatorParamMaps=grid,
            evaluator=evaluator,
            numFolds=2,
            seed=101,
        )

        def check_cv(cv_est: Union[CrossValidator, CrossValidatorModel]) -> None:
            assert isinstance(cv_est, (CrossValidator, CrossValidatorModel))
            assert isinstance(cv_est.getEstimator(), RandomForestRegressor)
            assert isinstance(cv_est.getEvaluator(), RegressionEvaluator)
            assert cv_est.getNumFolds() == 2
            assert cv_est.getSeed() == 101
            assert cv_est.getEstimatorParamMaps() == grid

        check_cv(cv)

        path = tmp_path + "/cv"
        cv_path = f"{path}/cv"

        cv.write().overwrite().save(cv_path)
        cv_loaded = CrossValidator.load(cv_path)

        check_cv(cv_loaded)

        cv_model = cv.fit(df)
        check_cv(cv_model)

        cv_model_path = f"{path}/cv-model"
        cv_model.write().overwrite().save(cv_model_path)
        cv_model_loaded = CrossValidatorModel.load(cv_model_path)

        check_cv(cv_model_loaded)
        assert evaluator.evaluate(cv_model.transform(df)) == evaluator.evaluate(
            cv_model_loaded.transform(df)
        )
