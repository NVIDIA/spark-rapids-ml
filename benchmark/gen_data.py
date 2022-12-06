import argparse
from benchmark.utils import prepare_spark_session 
from pyspark.mllib.random import RandomRDDs
from pyspark.sql.types import (
    ArrayType, 
    DoubleType, 
    FloatType,
    StructType,
    StructField
)

if __name__ == "__main__":

    """
    python gen_data.py \
    --num_vecs 5000 \
    --dim 3000 \
    --dtype "float64" \
    --parquet_path "./5k_2k_float64.parquet" \
    --spark_conf "spark.master=local[*]" \
    --spark_confs "spark.driver.memory=128g" 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_vecs", type=int, default=5000)
    parser.add_argument("--dim", type=int, default=3000)
    parser.add_argument("--dtype", type=str, choices=["float64"], default="float64")
    parser.add_argument("--parquet_path", type=str, default="./5k_3k_float64.parquet")
    parser.add_argument("--spark_confs", action="append", default=[])
    args = parser.parse_args()

    spark = prepare_spark_session(args.spark_confs)
    rdd = RandomRDDs.uniformVectorRDD(spark, args.num_vecs, args.dim).map(lambda nparray: (nparray.tolist(),)) 
    col_type = DoubleType() if args.dtype == "float64" else FloatType()
    schema = StructType(
        [StructField("", ArrayType(col_type, False), False)]
    )
    df = spark.createDataFrame(rdd, schema)

    assert df.count() == args.num_vecs
    row = df.first()
    assert len(row[0]) == args.dim

    df.write.mode("overwrite").parquet(args.parquet_path)
    print("gen_data finished")
