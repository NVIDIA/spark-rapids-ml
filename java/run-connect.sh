#! /bin/bash

which spark-submit


spark-submit --master "sc://localhost" \
 --conf spark.connect.ml.backend.classes=com.nvidia.rapids.ml.Plugin \
 rapids-plugin-demo.py


