#!/bin/bash

if ! command -v nvidia-smi &> /dev/null
then
    # default to the first GPU
    echo "{\"name\":\"gpu\",\"addresses\":[\"0\"]}"
    exit
else
    # https://github.com/apache/spark/blob/master/examples/src/main/scripts/getGpusResources.sh
    ADDRS=`nvidia-smi --query-gpu=index --format=csv,noheader | sed -e ':a' -e 'N' -e'$!ba' -e 's/\n/","/g'`
    echo {\"name\": \"gpu\", \"addresses\":[\"$ADDRS\"]}
fi