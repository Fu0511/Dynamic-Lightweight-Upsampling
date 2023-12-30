dataset_type = "CUHK_SYSU"

# SPECIFY YOUR VARIABLES
data_root = "/home/reusm/data/sysu_pedes/mmlab"

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadReIDDetAnnotations"),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="RandomChoice",
        transforms=[
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (667, 400),
                        (1000, 600),
                        (1333, 800),
                        (1500, 900),
                        (1666, 1000),
                    ],
                    keep_ratio=True,
                )
            ],
        ],
    ),
    dict(
        type="Normalize",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="PackReIDDetInputs"),
]

train_dataloader = dict(
    shuffle=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        filter_cfg=dict(filter_empty_gt=False),
        ann_file="annotations/annotations_sysu_train.json",
        data_root=data_root,
        pipeline=train_pipeline,
    ),
)
