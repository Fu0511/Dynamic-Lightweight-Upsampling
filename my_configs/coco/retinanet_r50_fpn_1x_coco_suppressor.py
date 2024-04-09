_base_ = [
    '../../configs/_base_/models/retinanet_r50_fpn.py',
    '../../configs/_base_/datasets/coco_detection.py',
    '../../configs/_base_/schedules/schedule_1x.py', '../../configs/_base_/default_runtime.py',
    '../../configs/retinanet/retinanet_tta.py'
]

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
    
train_dataloader=dict(batch_size=4,num_workers=4)
model=dict(backbone=dict(init_cfg=dict(type='Pretrained',checkpoint='/data/young/codes/Robust_Vision/local_results/robust_exp/vr/finetune/partial_finetune/in1k/resnet50_imagenet_conv1_top30_e5_replace_diversify_2/model_best_for_voc.pth')))
