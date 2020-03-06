#!/bin/bash -e

cd $(dirname $0)

dataset=$1

if [ -z "$dataset" ]; then
    echo "usage: $0 dataset "
    exit 1
fi

exec python -m realadv find_edge_model output/$dataset/* \
    --log output/${dataset}.model.log
