#!/bin/bash -e

if [ -n "$ARGV1_NEED_PARSE" ]; then
    export -n ARGV1_NEED_PARSE
    exec bash -c "$0 $@"
fi

cd $(dirname $0)

dataset=$1
shift

if [ -z "$dataset" ]; then
    echo "usage: $0 dataset [other args ...]"
    echo "Note: see realadv.attack for more command line options"
    exit 1
fi

# multi-threading impl seems to occupy cpu cores without significant speedup
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

exec python -m realadv attack output/$dataset/*.model "$@" \
    --log-file output/$dataset.attack.log
