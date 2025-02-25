# model settings
model = dict(
    type='WhollyWood',
    backbone=dict(
        type='Resnet50',
        frozen_stages=1,
        norm_eval=True,
        return_stages=["layer1", "layer2", "layer3", "layer4"],
        pretrained=True),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_output",
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='H2RBoxV2PHead',
        num_classes=15,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        square_cls=[1, 9, 11],
        resize_cls=[1],
        loss_cls=dict(
            type='FocalLoss',
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_bce=True, loss_weight=1.0),
        loss_ss_symmetry=dict(
            type='SmoothL1Loss', loss_weight=0.2, beta=0.1),
        test_cfg=dict(
            centerness_factor=0.5,
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.3,
            nms=dict(type='obb_nms', iou_thr=0.1),
            max_per_img=2000)),
    p2rsubnet=dict(
        type='P2RSubnet',
        basic_pattern='data/basic_patterns/dota',
        warmup=10,
        dense_cls=[4, 5, 6, 9],
        square_cls=[1, 9, 11],
        use_setrc=False,
        use_setsk=True,
        use_synthesis=True,
        backbone=dict(
            type='Resnet50',
            frozen_stages=1,
            norm_eval=True,
            return_stages=["layer1", "layer2", "layer3", "layer4"],
            pretrained=True),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs="on_output",
            num_outs=5,
            relu_before_extra_convs=True),
        bbox_head=dict(
            type='P2RSubnetHead',
            num_classes=15,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            square_cls=[1, 9, 11],
            strides=[8],
            regress_ranges=[(-1, 1e8)],
            center_sampling=True,
            center_sample_radius=1.5,
            norm_on_bbox=True,
            centerness_on_reg=True,
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='IoULoss', loss_weight=1.0))))

# training and testing settings
dataset = dict(
    train=dict(
        type="WhollyWoodDOTADataset",
        dataset_dir='/home/yi/Data/processed_DOTA/trainval_1024_200_1.0',
        rhp=[0.0, 0.0, 1.0],
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict(type='RotatedRandomFlip', prob=0.5),
            dict(
                type="Pad",
                size_divisor=32),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False, )
        ],
        batch_size=2,
        num_workers=2,
        shuffle=True,
        filter_empty_gt=False
    ),
    val=dict(
        type="WhollyWoodDOTADataset",
        dataset_dir='/home/yi/Data/processed_DOTA/trainval_1024_200_1.0',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict(
                type="Pad",
                size_divisor=32),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
        ],
        batch_size=4,
        num_workers=4,
        shuffle=False
    ),
    test=dict(
        type="ImageDataset",
        images_dir='/home/yi/Data/processed_DOTA/test_1024_200_1.0/images',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict(
                type="Pad",
                size_divisor=32),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False, ),
        ],
        num_workers=4,
        batch_size=4,
    )
)

optimizer = dict(
    type='AdamW',
    lr=0.00005,
    betas=(0.9, 0.999),
    weight_decay=0.05
)

scheduler = dict(
    type='StepLR',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    milestones=[8, 11])

logger = dict(
    type="RunLogger")

# when we the trained model from cshuan, image is rgb
max_epoch = 12
eval_interval = 12
checkpoint_interval = 1
log_interval = 50
