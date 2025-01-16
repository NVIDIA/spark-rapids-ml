#! /bin/bash

which spark-submit

spark-submit \
 --class com.nvidia.rapids.ml.Main \
 target/com.nvidia.rapids.ml-1.0-SNAPSHOT.jar
