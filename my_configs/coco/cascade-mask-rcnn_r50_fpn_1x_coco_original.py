_base_ = [
    '../../configs/_base_/models/cascade-mask-rcnn_r50_fpn.py',
    '../../configs/_base_/datasets/coco_instance.py',
    '../../configs/_base_/schedules/schedule_1x.py', '../../configs/_base_/default_runtime.py'
]

train_dataloader=dict(batch_size=4,num_workers=4)