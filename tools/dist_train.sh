#!/usr/bin/env bash

CONFIG='/home/disk1/datasets/MGA_110_best/projects/configs/MGA/100query_class3_110epoch.py'
GPUS=$1
PORT=${PORT:-28510}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic
