#!/usr/bin/env bash

CONFIG1='my_configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712_suppressor_freeze_stage0.py'
CONFIG2='my_configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712_suppressor_freeze_stage2.py'
GPUS=4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
GPUS_ids="6,7,8,9"
WORK_DIR1='local_results/od/voc/fasterRCNN_r50_fpn_1x_suppressor_freeze_stage0'
WORK_DIR2='local_results/od/voc/fasterRCNN_r50_fpn_1x_suppressor_freeze_stage2'

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$GPUS_ids python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    $CONFIG1 \
    --work-dir $WORK_DIR1 \
    --auto-scale-lr \
    --launcher pytorch ${@:3}

# CUDA_VISIBLE_DEVICES=$GPUS_ids python -m torch.distributed.launch \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     tools/train.py \
#     $CONFIG2 \
#     --work-dir $WORK_DIR2 \
#     --auto-scale-lr \
#     --launcher pytorch ${@:3}


