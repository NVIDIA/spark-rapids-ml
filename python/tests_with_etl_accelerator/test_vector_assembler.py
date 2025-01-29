from typing import Union, Dict, Any
from pyspark.ml.feature import VectorAssembler as CPUVectorAssembler
from spark_rapids_ml.feature import VectorAssembler as GPUVectorAssembler
import pytest


@pytest.mark.parametrize(
    "Assembler",
    [
        GPUVectorAssembler,
        #CPUVectorAssembler,
    ],
)
def test_compat(
    Assembler: Union[CPUVectorAssembler, GPUVectorAssembler],
) -> None:

    from .conftest import _spark
    df = _spark.read.parquet("/tmp/tmp.parquet")

    assembler = Assembler(
        inputCols = ["first_home_buyer", "borrower_credit_score", "zip"],
        outputCol="features")

    df_res = assembler.transform(df)
    df_res.explain()

