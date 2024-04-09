#!/usr/bin/env bash

CONFIG='my_configs/coco/detr_r50_8xb2-150e_coco_original.py'
GPUS=4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29520}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
GPUS_ids="2,3,4,5"
WORK_DIR='local_results/od/coco/detr_r50_8xb2-150e_coco_original'

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$GPUS_ids python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    $CONFIG \
    --work-dir $WORK_DIR \
    --auto-scale-lr \
    --launcher pytorch ${@:3}


