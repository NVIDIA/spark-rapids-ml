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
import importlib
from importlib import import_module
from pkgutil import iter_modules
import os
import sys
from typing import IO

import py4j
from py4j.java_gateway import GatewayParameters, java_import

import pyspark
from pyspark import SparkConf, SQLContext, SparkContext
from pyspark.accumulators import _accumulatorRegistry
from pyspark.serializers import (
    read_int,
    write_int,
    write_with_length,
    SpecialLengths, UTF8Deserializer,
)
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.datasource import DataSource
from pyspark.util import handle_worker_exception, local_connect_and_auth
from pyspark.worker_util import (
    check_python_version,
    pickleSer,
    send_accumulator_updates,
    setup_broadcasts,
    setup_memory_limits,
    setup_spark_files,
)

utf8_deserializer = UTF8Deserializer()

def _java_import(gateway) -> None:
    java_import(gateway.jvm, "org.apache.spark.SparkConf")
    java_import(gateway.jvm, "org.apache.spark.api.java.*")
    java_import(gateway.jvm, "org.apache.spark.api.python.*")
    java_import(gateway.jvm, "org.apache.spark.ml.python.*")
    java_import(gateway.jvm, "org.apache.spark.mllib.api.python.*")
    java_import(gateway.jvm, "org.apache.spark.resource.*")
    # TODO(davies): move into sql
    java_import(gateway.jvm, "org.apache.spark.sql.Encoders")
    java_import(gateway.jvm, "org.apache.spark.sql.OnSuccessCall")
    java_import(gateway.jvm, "org.apache.spark.sql.functions")
    java_import(gateway.jvm, "org.apache.spark.sql.classic.*")
    java_import(gateway.jvm, "org.apache.spark.sql.api.python.*")
    java_import(gateway.jvm, "org.apache.spark.sql.hive.*")
    java_import(gateway.jvm, "scala.Tuple2")

def main(infile: IO, outfile: IO) -> None:
    """
    Main method for looking up the available Python Data Sources in Python path.

    This process is invoked from the `UserDefinedPythonDataSourceLookupRunner.runInPython`
    method in `UserDefinedPythonDataSource.lookupAllDataSourcesInPython` when the first
    call related to Python Data Source happens via `DataSourceManager`.

    This is responsible for searching the available Python Data Sources so they can be
    statically registered automatically.
    """
    faulthandler_log_path = os.environ.get("PYTHON_FAULTHANDLER_DIR", None)
    try:
        if faulthandler_log_path:
            faulthandler_log_path = os.path.join(faulthandler_log_path, str(os.getpid()))
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
        estimator_name = utf8_deserializer.loads(infile)
        java_sc_key = utf8_deserializer.loads(infile)
        dataset_key = utf8_deserializer.loads(infile)

        # Create a Java Gateway
        gateway = py4j.java_gateway.JavaGateway(
            gateway_parameters=GatewayParameters(auth_token=auth_token, auto_convert=True))
        _java_import(gateway)

        # Get the JavaObject of Dataset and JavaSparkContext
        jdf = py4j.java_gateway.JavaObject(dataset_key, gateway._gateway_client)
        jsc = py4j.java_gateway.JavaObject(java_sc_key, gateway._gateway_client)

        # Test
        jdf.show()

        # Prepare to create SparkContext and SparkSession
        sc = SparkContext(conf=SparkConf(_jconf=jsc.sc().conf()), gateway=gateway, jsc=jsc)
        spark = SparkSession(sc, jdf.sparkSession())

        # Create DataFrame
        df = DataFrame(jdf, spark)


        # Initialize the estimator of spark-rapids-ml
        module = importlib.import_module("spark_rapids_ml.classification")
        klass = getattr(module, "LogisticRegression")
        lr = klass()
        # lr.setFeaturesCols(["features"])
        lr.setMaxIter(26)
        print("-------------------------- 5")
        model = lr.fit(df)
        print("-------------------------- 7")
        print(f"the maxIter of model is {model.getMaxIter()}")

    except BaseException as e:
        print(f"-------------exception ------------- {e}")
        handle_worker_exception(e, outfile)
        sys.exit(-1)
    finally:
        if faulthandler_log_path:
            faulthandler.disable()
            faulthandler_log_file.close()
            os.remove(faulthandler_log_path)

    send_accumulator_updates(outfile)
    # check end of stream
    if read_int(infile) == SpecialLengths.END_OF_STREAM:
        write_int(SpecialLengths.END_OF_STREAM, outfile)
    else:
        # write a different value to tell JVM to not reuse this worker
        write_int(SpecialLengths.END_OF_DATA_SECTION, outfile)
        sys.exit(-1)


if __name__ == "__main__":
    # Read information about how to connect back to the JVM from the environment.
    java_port = int(os.environ["PYTHON_WORKER_FACTORY_PORT"])
    auth_secret = os.environ["PYTHON_WORKER_FACTORY_SECRET"]
    (sock_file, _) = local_connect_and_auth(java_port, auth_secret)
    write_int(os.getpid(), sock_file)
    sock_file.flush()
    main(sock_file, sock_file)
