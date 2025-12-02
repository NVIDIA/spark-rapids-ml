#
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import faulthandler
import json
import os
import sys
from typing import IO

import py4j
from py4j.java_gateway import GatewayParameters, java_import
from pyspark import SparkConf, SparkContext
from pyspark.accumulators import _accumulatorRegistry
from pyspark.serializers import (
    SpecialLengths,
    UTF8Deserializer,
    read_int,
    write_int,
    write_with_length,
)
from pyspark.sql import DataFrame, SparkSession
from pyspark.util import (  # type: ignore[attr-defined]
    handle_worker_exception,
    local_connect_and_auth,
)
from pyspark.worker_util import send_accumulator_updates  # type: ignore[attr-defined]
from pyspark.worker_util import setup_broadcasts  # type: ignore[attr-defined]
from pyspark.worker_util import setup_memory_limits  # type: ignore[attr-defined]
from pyspark.worker_util import setup_spark_files  # type: ignore[attr-defined]
from pyspark.worker_util import (
    check_python_version,
)

utf8_deserializer = UTF8Deserializer()


def _java_import(gateway) -> None:  # type: ignore[no-untyped-def]
    java_import(gateway.jvm, "org.apache.spark.SparkConf")
    java_import(gateway.jvm, "org.apache.spark.api.java.*")
    java_import(gateway.jvm, "org.apache.spark.api.python.*")
    java_import(gateway.jvm, "org.apache.spark.ml.python.*")
    java_import(gateway.jvm, "org.apache.spark.mllib.api.python.*")
    java_import(gateway.jvm, "org.apache.spark.resource.*")
    java_import(gateway.jvm, "org.apache.spark.sql.Encoders")
    java_import(gateway.jvm, "org.apache.spark.sql.OnSuccessCall")
    java_import(gateway.jvm, "org.apache.spark.sql.functions")
    java_import(gateway.jvm, "org.apache.spark.sql.classic.*")
    java_import(gateway.jvm, "org.apache.spark.sql.api.python.*")
    java_import(gateway.jvm, "org.apache.spark.sql.hive.*")
    java_import(gateway.jvm, "scala.Tuple2")


def main(infile: IO, outfile: IO) -> None:
    """
    Main method for running spark-rapids-ml.
    """
    faulthandler_log_path = os.environ.get("PYTHON_FAULTHANDLER_DIR", None)
    try:
        if faulthandler_log_path:
            faulthandler_log_path = os.path.join(
                faulthandler_log_path, str(os.getpid())
            )
            faulthandler_log_file = open(faulthandler_log_path, "w")
            faulthandler.enable(file=faulthandler_log_file)

        check_python_version(infile)
        memory_limit_mb = int(os.environ.get("PYSPARK_PLANNER_MEMORY_MB", "-1"))
        setup_memory_limits(memory_limit_mb)
        setup_spark_files(infile)
        setup_broadcasts(infile)
        _accumulatorRegistry.clear()

        # Receive variables from JVM
        auth_token = utf8_deserializer.loads(infile)
        operator_name = utf8_deserializer.loads(infile)
        params = utf8_deserializer.loads(infile)
        java_sc_key = utf8_deserializer.loads(infile)
        dataset_key = utf8_deserializer.loads(infile)

        # Create a Java Gateway
        gateway = py4j.java_gateway.JavaGateway(
            gateway_parameters=GatewayParameters(
                auth_token=auth_token, auto_convert=True
            )
        )
        _java_import(gateway)

        # Get the JavaObject of Dataset and JavaSparkContext
        jdf = py4j.java_gateway.JavaObject(dataset_key, gateway._gateway_client)
        jsc = py4j.java_gateway.JavaObject(java_sc_key, gateway._gateway_client)

        # Prepare to create SparkContext and SparkSession
        sc = SparkContext(
            conf=SparkConf(_jconf=jsc.sc().conf()), gateway=gateway, jsc=jsc
        )
        spark = SparkSession(sc, jdf.sparkSession())

        # Create DataFrame
        df = DataFrame(jdf, spark)

        print(f"Running {operator_name} with parameters: {params}")
        params = json.loads(params)

        def transform(MODEL_TYPE: type) -> DataFrame:
            attributes = utf8_deserializer.loads(infile)
            attributes = json.loads(attributes)  # type: ignore[arg-type]

            model = MODEL_TYPE(*attributes)  # type: ignore[arg-type]
            model._set_params(**params)
            return model.transform(df)

        if operator_name == "LogisticRegression":
            from .classification import LogisticRegression

            lr_model = LogisticRegression(**params).fit(df)
            attributes = [
                lr_model.coef_,
                lr_model.intercept_,
                lr_model.classes_,
                lr_model.n_cols,
                lr_model.dtype,
                lr_model.num_iters,
            ]
            write_with_length(json.dumps(attributes).encode("utf-8"), outfile)

        elif operator_name == "LogisticRegressionModel":
            from .classification import LogisticRegressionModel

            transformed_df = transform(LogisticRegressionModel)
            write_with_length(transformed_df._jdf._target_id.encode("utf-8"), outfile)

        elif operator_name == "RandomForestClassifier":
            from .classification import RandomForestClassifier

            rfc_model = RandomForestClassifier(**params).fit(df)
            # Model attributes
            attributes = [
                rfc_model.n_cols,
                rfc_model.dtype,
                rfc_model._treelite_model,
                rfc_model._model_json,
                rfc_model._num_classes,
            ]
            write_with_length(json.dumps(attributes).encode("utf-8"), outfile)

        elif operator_name == "RandomForestClassificationModel":
            from .classification import RandomForestClassificationModel

            transformed_df = transform(RandomForestClassificationModel)
            write_with_length(transformed_df._jdf._target_id.encode("utf-8"), outfile)

        elif operator_name == "RandomForestRegressor":
            from .regression import RandomForestRegressor

            rfc_model = RandomForestRegressor(**params).fit(df)
            # Model attributes
            attributes = [
                rfc_model.n_cols,
                rfc_model.dtype,
                rfc_model._treelite_model,
                rfc_model._model_json,
            ]
            write_with_length(json.dumps(attributes).encode("utf-8"), outfile)

        elif operator_name == "RandomForestRegressionModel":
            from .regression import RandomForestRegressionModel

            transformed_df = transform(RandomForestRegressionModel)
            write_with_length(transformed_df._jdf._target_id.encode("utf-8"), outfile)

        elif operator_name == "PCA":
            from .feature import PCA

            pca_model = PCA(**params).fit(df)
            # Model attributes
            attributes = [
                pca_model.mean_,
                pca_model.components_,
                pca_model.explained_variance_ratio_,
                pca_model.singular_values_,
                pca_model.n_cols,
                pca_model.dtype,
            ]
            write_with_length(json.dumps(attributes).encode("utf-8"), outfile)

        elif operator_name == "PCAModel":
            from .feature import PCAModel

            transformed_df = transform(PCAModel)
            write_with_length(transformed_df._jdf._target_id.encode("utf-8"), outfile)

        elif operator_name == "LinearRegression":
            from .regression import LinearRegression, LinearRegressionModel

            linear_model = LinearRegression(**params).fit(df)
            # Model attributes
            attributes = [
                linear_model.coef_,
                linear_model.intercept_,
                linear_model.n_cols,
                linear_model.dtype,
            ]
            write_with_length(json.dumps(attributes).encode("utf-8"), outfile)

        elif operator_name == "LinearRegressionModel":
            from .regression import LinearRegressionModel

            transformed_df = transform(LinearRegressionModel)
            write_with_length(transformed_df._jdf._target_id.encode("utf-8"), outfile)

        elif operator_name == "KMeans":
            from .clustering import KMeans, KMeansModel

            kmeans_model = KMeans(**params).fit(df)
            # Model attributes
            attributes = [
                kmeans_model.cluster_centers_,
                kmeans_model.n_cols,
                kmeans_model.dtype,
            ]
            write_with_length(json.dumps(attributes).encode("utf-8"), outfile)

        elif operator_name == "KMeansModel":
            from .clustering import KMeansModel

            transformed_df = transform(KMeansModel)
            write_with_length(transformed_df._jdf._target_id.encode("utf-8"), outfile)

        else:
            raise RuntimeError(f"Unsupported estimator: {operator_name}")

    except BaseException as e:
        print(f"spark-rapids-ml exception: {e}")
        handle_worker_exception(e, outfile)
        sys.exit(-1)
    finally:
        if faulthandler_log_path:
            faulthandler.disable()
            faulthandler_log_file.close()
            os.remove(faulthandler_log_path)

    send_accumulator_updates(outfile)

    def flush() -> None:
        outfile.flush()
        import time

        time.sleep(2)

    # check end of stream
    if read_int(infile) == SpecialLengths.END_OF_STREAM:
        write_int(SpecialLengths.END_OF_STREAM, outfile)
        flush()
    else:
        # write a different value to tell JVM to not reuse this worker
        write_int(SpecialLengths.END_OF_DATA_SECTION, outfile)
        flush()
        sys.exit(-1)


if __name__ == "__main__":
    # Read information about how to connect back to the JVM from the environment.
    java_port = int(os.environ["PYTHON_WORKER_FACTORY_PORT"])
    auth_secret = os.environ["PYTHON_WORKER_FACTORY_SECRET"]
    (sock_file, _) = local_connect_and_auth(java_port, auth_secret)
    write_int(os.getpid(), sock_file)
    sock_file.flush()
    main(sock_file, sock_file)
