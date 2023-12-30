_base_ = [
    "/home/reusm/code/mmdet/configs/_base_/default_runtime.py",
    "/home/reusm/code/mmdet/configs/_base_/datasets/cuhk_sysu.py",
]

detector = dict(
    type="DeformableDETR",
    num_queries=100,
    num_feature_levels=1,
    with_box_refine=False,
    as_two_stage=False,
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1,
    ),
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(
            type="Pretrained",
            checkpoint="torchvision://resnet50",
        ),
    ),
    neck=dict(
        type="PSTRMapper",
        in_channels=[512, 1024, 2048],
        out_channels=256,
    ),
    encoder=dict(  # DeformableDetrTransformerEncoder
        num_layers=3,
        layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
            self_attn_cfg=dict(  # MultiScaleDeformableAttention
                num_levels=1,
                embed_dims=256,
            ),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=256,
                ffn_drop=0.1,
            ),
        ),
    ),
    decoder=dict(  # DeformableDetrTransformerDecoder
        num_layers=3,
        return_intermediate=True,
        layer_cfg=dict(  # DeformableDetrTransformerDecoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
            ),
            cross_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256,
                num_levels=1,
            ),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=256,
                ffn_drop=0.1,
            ),
        ),
        post_norm_cfg=None,
    ),
    positional_encoding=dict(  # SinePositionalEncoding
        num_feats=128, normalize=True, offset=-0.5),
    bbox_head=dict(
        type="DeformableDETRHead",
        num_classes=1,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),
    ))

reid = dict(  # PSTRHeadReID
    type="PSTRHeadReID",
    # from mmdet.models.layers.transformer.pstr_part_attentions_layers.py
    decoder=dict(
        num_layers=1,
        part_attn_cfg=dict(
            num_levels=1,
            num_points=4,
            embed_dims=256,
            # Deformable DETR forces this to True
            batch_first=True,
        ),
    ),
    num_person=5532,
    flag_tri=True,
    queue_size=5000)

model = dict(type="PSTR", detector=detector, reid=reid)

model["train_cfg"] = dict(
    assigner=dict(
        type="HungarianAssigner",
        match_costs=[
            dict(type="FocalLossCost", weight=2.0),
            dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
            dict(type="IoUCost", iou_mode="giou", weight=2.0),
        ]))
model["data_preprocessor"] = dict(
    type="DetDataPreprocessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=1)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        # There are duplicates because ReID must use first layer of
        # regression branch from detection head.
        # bypass_duplicate=True,
        custom_keys={
            "backbone": dict(lr_mult=0.2),
            "sampling_offsets": dict(lr_mult=0.1),
            "reference_points": dict(lr_mult=0.1),
        }, ),
)

max_epochs = 24
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs)
# TODO Change when training is ok
test_cfg = None

param_scheduler = [
    # Warmup
    dict(
        type="LinearLR",
        start_factor=1 / 3,
        end_factor=1,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    # Epoch update
    dict(type='MultiStepLR', milestones=[19, 23], gamma=0.1)
]
