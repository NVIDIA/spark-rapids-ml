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

import math
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import pytest
from gen_data_distributed import SparseRegressionDataGen
from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression
from pyspark.sql import functions as SparkF

from spark_rapids_ml.classification import LogisticRegression
from tests.test_logistic_regression import compare_model

from .conftest import _spark

from pyspark.sql.types import (
    DoubleType,
    FloatType,
    IntegerType,
    StructField,
    StructType,
)


def _compare_with_cpu_estimator(gpu_model, est_params, df, fraction_sampled_for_test, tolerance):
    cpu_est = SparkLogisticRegression(**est_params)
    cpu_model = cpu_est.fit(df)
    cpu_objective = cpu_model.summary.objectiveHistory[-1]

    for i, loss in enumerate(cpu_model.summary.objectiveHistory):
        print(f"Iteration {i}: loss = {loss:.6f}")

    df_test = df.sample(fraction=fraction_sampled_for_test, seed=0)

    print(f"CPU coef_: {cpu_model.coefficientMatrix.toArray()}")
    res = cpu_model.transform(df_test).select("rawPrediction", "probability")
    print(f"CPU res: {res.first()}")

    assert (
        gpu_model.objective < cpu_objective
        or abs(gpu_model.objective - cpu_objective) < tolerance
    )

    compare_model(
        gpu_model,
        cpu_model,
        df_test,
        unit_tol=tolerance,
        total_tol=tolerance,
        accuracy_and_probability_only=True,
        y_true_col=cpu_est.getLabelCol(),
    )

def test_sparse_int64(
    multi_gpus: bool = False, standardization: bool = False, float32_inputs: bool = True
) -> None:
    """
    This test requires minimum 128G CPU memory, 32 GB GPU memory
    TODO: move generated dataset to a unified place

    if standardization is True, the test case reduces more GPU memory since standardization copies the value array
    """
    gpu_number = 2 if multi_gpus else 1
    output_num_files = 100  # large value smaller CPU memory for each spark task
    data_shape = (int(1e7), 2200)

    fraction_sampled_for_test = (
        1.0 if data_shape[0] <= 100000 else 100000 / data_shape[0]
    )
    n_classes = 8
    tolerance = 0.001
    est_params: Dict[str, Any] = {
        "regParam": 0.02,
        "maxIter": 10,
        "standardization": standardization,
    }
    density = 0.1

    data_gen_args = [
        "--n_informative",
        f"{math.ceil(data_shape[1] / 3)}",
        "--num_rows",
        str(data_shape[0]),
        "--num_cols",
        str(data_shape[1]),
        "--output_num_files",
        str(output_num_files),
        "--dtype",
        "float32",
        "--feature_type",
        "vector",
        "--output_dir",
        "./temp",
        "--n_classes",
        str(n_classes),
        "--random_state",
        "0",
        "--logistic_regression",
        "True",
        "--density",
        str(density),
        "--use_gpu",
        "True",
    ]

    data_gen = SparseRegressionDataGen(data_gen_args)
    df, _, _ = data_gen.gen_dataframe_and_meta(_spark)

    df = df.cache()
    df_gpu = df

    if gpu_number > 1:
        main_pid = 0
        pid_col = "pid"
        delta_ratio = 0.1

        delta_df = df.sample(fraction=delta_ratio, seed=0)

        df = df.withColumn(pid_col, SparkF.lit(main_pid))
        delta_df = delta_df.withColumn(
            pid_col, SparkF.monotonically_increasing_id() % (gpu_number * 4)
        )

        df = df.union(delta_df)
        df_gpu = df.repartition(gpu_number, pid_col)

    def get_nnz_func(pdf_iter: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
        for pdf in pdf_iter:
            pd_res = pdf["features"].apply(lambda sparse_vec: len(sparse_vec["values"]))
            yield pd_res.rename("nnz").to_frame()

    nnz_df = df.mapInPandas(get_nnz_func, schema="nnz long")

    total_nnz = nnz_df.select(SparkF.sum("nnz").alias("res")).first()["res"]  # type: ignore
    assert total_nnz > np.iinfo(np.int32).max

    # compare gpu with spark cpu
    gpu_est = LogisticRegression(
        num_workers=gpu_number,
        verbose=True,
        float32_inputs=float32_inputs,
        **est_params,
    )
    gpu_model = gpu_est.fit(df_gpu)

    _compare_with_cpu_estimator(gpu_model, est_params, df, fraction_sampled_for_test, tolerance)
    """
    cpu_est = SparkLogisticRegression(**est_params)
    cpu_model = cpu_est.fit(df)
    cpu_objective = cpu_model.summary.objectiveHistory[-1]
    assert (
        gpu_model.objective < cpu_objective
        or abs(gpu_model.objective - cpu_objective) < tolerance
    )

    df_test = df.sample(fraction=fraction_sampled_for_test, seed=0)
    compare_model(
        gpu_model,
        cpu_model,
        df_test,
        unit_tol=tolerance,
        total_tol=tolerance,
        accuracy_and_probability_only=True,
    )
    """

    if gpu_number == 1:
        assert gpu_est._index_dtype == "int64"


def test_sparse_int64_mg() -> None:
    test_sparse_int64(multi_gpus=True)


@pytest.mark.parametrize("float32_inputs", [True, False])
def test_sparse_int64_standardization(float32_inputs: bool) -> None:
    test_sparse_int64(
        multi_gpus=False, standardization=True, float32_inputs=float32_inputs
    )


def _standardize(X_csr) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler
    X_dense = X_csr.toarray()
    
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_dense)

    return X_std
    

def _cal_objective(model, X, y) -> float:
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    y_prob = model.predict_proba(X)[:, 1]

    X_likelihood = y * np.log(y_prob)  + (1-y) * np.log(1 - y_prob)
    X_loss = -(np.sum(X_likelihood) / len(y))
    return X_loss


def _train_cuml_sg(df, vec_input_col, float32_inputs, label_col, standardization, **algo_params) -> None:
    assert standardization is True

    from pyspark.sql.functions import col
    from spark_rapids_ml.core import _get_unwrapped_vec_cols, _read_csr_matrix_from_unwrapped_spark_vec

    select_cols = []
    select_cols += _get_unwrapped_vec_cols(
        col(vec_input_col), float32_inputs
    )

    df_unwrap = df.select(*select_cols)

    pdf_unwrap = df_unwrap.toPandas()

    X_csr = _read_csr_matrix_from_unwrapped_spark_vec(pdf_unwrap)
    y = [r[label_col] for r in df.select(label_col).collect()]

    X_std = _standardize(X_csr)

    from cuml import LogisticRegression
    sg = LogisticRegression(verbose=6, **algo_params)
    # sg.solver_model.penalty_normalized = False
    # sg.solver_model.lbfgs_memory = 10
    # sg.solver_model.linesearch_max_iter = 20

    print(f"sg start: ")
    sg.fit(X_std, y)

    print(f"sg finished with objective: {sg.objective}")
    print(f"sg calculated objective: {_cal_objective(sg, X_std, y)}")

    from sklearn.linear_model import LogisticRegression as SKLR
    sk = SKLR(verbose=6, **algo_params)

    print(f"sk start: ")
    sk.fit(X_std, y)
    print(f"sk calculated objective: {_cal_objective(sk, X_std, y)}")


def test_extreme_value() -> None:
    from pyspark.ml.linalg import Vectors
    data = [
        (Vectors.dense([2.0, 296000.0]), 1.),
        (Vectors.dense([2.0, 273000.0]), 1.),
        (Vectors.dense([1.0, 132000.0]), 0.),
        (Vectors.dense([1.0, 135000.0]), 0.),
    ]
    df = _spark.createDataFrame(data)

    est_params = {
        "standardization": True,
        "maxIter": 3,
        "regParam": 0.,
        "elasticNetParam": 0.,
        "fitIntercept": False,
        "featuresCol": "_1",
        "labelCol": "_2",
    }

    lr_gpu = LogisticRegression(verbose=6, float32_inputs=False, **est_params)
    lr_gpu_model = lr_gpu.fit(df)
    _compare_with_cpu_estimator(lr_gpu_model, est_params, df, fraction_sampled_for_test=1., tolerance=0.01)


def test_sparse_mortgage() -> None:
    float32_inputs = False 
    features_col = "features"
    label_col = "delinquency_12"
    tolerance = 0.001
    fraction_sampled_for_test = 0.01
    max_iter = 3 
    reg_param = 0.
    elasticNet_param = 0.
    fit_intercept=False
    standardization=True

    est_params = {
        "standardization": standardization,
        "maxIter": max_iter,
        "regParam": reg_param,
        "elasticNetParam": elasticNet_param,
        "fitIntercept": fit_intercept,
        "featuresCol": features_col,
        "labelCol": label_col,
    }

    penalty, C, l1_ratio = LogisticRegression._reg_params_value_mapping(reg_param, elasticNet_param)
    cuml_sg_params = {
        "max_iter": max_iter,
        "penalty": penalty,
        "C": C,
        "l1_ratio": l1_ratio,
        "fit_intercept": fit_intercept,
    }

    #input_path = "/data/eordentlich/mortgage/output/data/part-00000*"
    #input_path = "/data/eordentlich/mortgage/output/data/part-0000{0,1,2,3}*"
    input_path = "/data/eordentlich/mortgage/output/data/part-000{0,1}*"
    etlDf = _spark.read.parquet(input_path)

    num_rows = etlDf.count()
    print(f"debug num_rows: {num_rows}")

    etlDf = etlDf.withColumn("loc",(etlDf.msa*1000+etlDf.zip).cast("int")).drop("zip" ,"msa")

    schema = StructType(
        [
            StructField("orig_channel", FloatType()), # 0-th
            StructField("first_home_buyer", FloatType()),
            StructField("loan_purpose", FloatType()),
            StructField("property_type", FloatType()),
            StructField("occupancy_status", FloatType()),
            StructField("property_state", FloatType()),
            StructField("product_type", FloatType()),
            StructField("relocation_mortgage_indicator", FloatType()),
            StructField("seller_name", FloatType()),
            StructField("mod_flag", FloatType()),
            StructField("orig_interest_rate", FloatType()),
            StructField("orig_upb", DoubleType()), # 11-th
            StructField("orig_loan_term", IntegerType()),
            StructField("orig_ltv", FloatType()),
            StructField("orig_cltv", FloatType()),
            StructField("num_borrowers", FloatType()),
            StructField("dti", FloatType()),
            StructField("borrower_credit_score", FloatType()),
            StructField("num_units", IntegerType()),
            #StructField("zip", IntegerType()),
            StructField("loc", IntegerType()), # to be deleted intermediate
            StructField("mortgage_insurance_percent", FloatType()),
            StructField("current_loan_delinquency_status", IntegerType()),
            StructField("current_actual_upb", FloatType()), # 21-th
            StructField("interest_rate", FloatType()),
            StructField("loan_age", FloatType()),
            # StructField("msa", FloatType()),
            StructField("non_interest_bearing_upb", FloatType()),
            StructField(label_col, IntegerType()),
        ]
    )
    features = [x.name for x in schema if x.name != label_col]

    from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
    str_index = StringIndexer().setInputCol("loc").setOutputCol("str_index") # (hno support, fallback
    one_hot= OneHotEncoder().setInputCol("str_index").setOutputCol("zip_onehot") # no support, fallback

    features_copy = features.copy()
    features_copy.remove("loc")
    features_copy.append("zip_onehot")

    va = (
        VectorAssembler().setInputCols(features_copy).setOutputCol(features_col)
    )  # no support, fallback

    logistic = (
        LogisticRegression(verbose=6, float32_inputs=float32_inputs, **est_params)
    )


    from pyspark.ml.pipeline import Pipeline, PipelineModel
    pipeline = Pipeline(stages = [str_index, one_hot, va, logistic])
    pipeline_model = pipeline.fit(etlDf)

    etlDf_transformed = PipelineModel(pipeline_model.stages[:3]).transform(etlDf)
    gpu_model = pipeline_model.stages[3]

    first = etlDf_transformed.first()
    print(f"first: {first}")
    if num_rows < 100:
        res = etlDf_transformed.select(features_col).collect()
        for r in res:
            print(r)

    #fraction_sampled_for_test = (
    #    1.0 if num_rows <= 100000 else 100000 / num_rows
    #)
    #_compare_with_cpu_estimator(gpu_model, est_params, etlDf_transformed, fraction_sampled_for_test, tolerance)

    ### compare with cuml sg
    #_train_cuml_sg(etlDf_transformed, features_col, float32_inputs, label_col, standardization, **cuml_sg_params)
    #print("finished")

