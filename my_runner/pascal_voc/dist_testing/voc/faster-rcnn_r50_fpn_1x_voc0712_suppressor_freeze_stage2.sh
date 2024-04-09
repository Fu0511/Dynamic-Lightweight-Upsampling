#!/usr/bin/env bash

CONFIG='my_configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712_suppressor_freeze_stage2.py'
GPUS=4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29553}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
GPUS_ids="6,7,8,9"
CHECKPOINT='local_results/od/voc/fasterRCNN_r50_fpn_1x_suppressor_freeze_stage2/epoch_4.pth'
BASE_WORK_DIR='local_results/od/voc/fasterRCNN_r50_fpn_1x_suppressor_freeze_stage2/'
BASE_DATA_PREFIX='VOC2007-C'
RESULT_NAME='results.pkl'

CO_LIST=('defocus_blur' 'glass_blur' 'motion_blur' 'zoom_blur' 'snow' 
'frost' 'fog' 'brightness' 'contrast' 'elastic_transform' 'pixelate' 'jpeg_compression'
'gaussian_noise' 'shot_noise' 'impulse_noise')

CO_LEVEL=('1' '2' '3' '4' '5')

for item1 in "${CO_LIST[@]}"  
do  
    for item2 in "${CO_LEVEL[@]}"  
    do
        if [[ "${BASE_WORK_DIR: -1}" != "/" ]]; then  
            BASE_WORK_DIR="${BASE_WORK_DIR}/"

        fi

        WORK_DIR="${BASE_WORK_DIR}/${item1}/${item2}"
        RESULT_FILE="${BASE_WORK_DIR}/${item1}/${item2}/${RESULT_NAME}"
        SUB_DATA_ROOT="${BASE_DATA_PREFIX}/${item1}/${item2}"

        #PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
        CUDA_VISIBLE_DEVICES=$GPUS_ids python -m torch.distributed.launch \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --nproc_per_node=$GPUS \
            --master_port=$PORT \
            tools/test.py \
            $CONFIG \
            $CHECKPOINT \
            --work-dir $WORK_DIR \
            --out $RESULT_FILE \
            --cfg-options \
            test_dataloader.dataset.data_prefix.sub_data_root=$SUB_DATA_ROOT \
            --launcher pytorch ${@:3}
    done
done


