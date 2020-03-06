#!/bin/bash -e

cd $(dirname $0)

dataset=$1
size=$2

if [ -z "$size" ]; then
    echo "usage: $0 dataset size"
    exit 1
fi

if [ "$dataset" = mnist ]; then
    eps=0.1
    linf=0.1
elif [ "$dataset" = cifar10 ]; then
    eps=0.0078431372549019607
    linf=2_255
else
    echo "unknown dataset"
    exit 2
fi

mkdir -pv output/$dataset

if [ ! -f data/$dataset.pth ]; then
    echo "converting the model to pytorch foramt ..."
    python -m realadv cvt_model data/$dataset.mat.xz data/$dataset.pth $dataset
fi

function get_seeded_random() {
    # see https://stackoverflow.com/questions/5914513/shuffling-lines-of-a-file-with-a-fixed-seed
    seed="$1"
    openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
        </dev/zero 2>/dev/null
}

CMD="python -m realadv find_edge_input  \
    -e $eps \
    data/$dataset.pth data/$dataset.mat.xz \
    data/$dataset-linf-$linf.csv"

nr_rob=$($CMD | grep 'robust examples' | grep -o '\[[0-9]*,' | tr -d '[,')
echo "robust test cases: $nr_rob"

for i in $(seq 0 $(($nr_rob-1)) | \
    shuf --random-source=<(get_seeded_random 92702102) | head -n $size); do
    $CMD --index $i -o output/$dataset/$i
done

