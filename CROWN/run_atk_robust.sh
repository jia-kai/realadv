#!/bin/bash -e

model=$1

if [ -z "$model" ]; then
    echo "usage: $0 <model file>"
    exit -1
fi

cd $(dirname $0)

echo "======= RUNNING CPU matmul ======="
./atk_robust.py $model --cpu --mm

echo "======= RUNNING CPU conv ======="
./atk_robust.py $model --cpu

echo "======= RUNNING GPU mm ======="
./atk_robust.py $model --mm

echo "======= RUNNING GPU conv ======="
./atk_robust.py $model

echo "======= RUNNING GPU Winograd ======="
./atk_robust.py $model --winograd
