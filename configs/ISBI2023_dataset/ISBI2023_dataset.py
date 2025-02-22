_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/ISBI2023.py'
    
    # './mmpose/configs/_base_/default_runtime.py',
    # './mmpose/configs/_base_/datasets/ISBI2015.py'
]
checkpoint_config = dict(interval=100)
evaluation = dict(interval=1, metric=['MRE_i2','MRE_std_i2','SDR_2_i2','SDR_2.5_i2','SDR_3_i2','SDR_4_i2'], save_best='MRE_i2')

optimizer = dict(
    type='AdamW',
    lr=4e-4,#4.5e-4
    betas=(0.9, 0.999),
    weight_decay=0.01,
    )
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=1e-5,
    warmup_by_epoch=True)

total_epochs = 200
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

channel_cfg = dict(
    num_output_channels=29,
    dataset_joints=29,
    dataset_channel=[
        list(range(29)),
    ],
    inference_channel=list(range(29)))
norm_cfg = dict(type='SyncBN', requires_grad=True)

# model settings
model = dict(
    type='TopDown_2Head_Neck',
    pretrained=None,
    backbone=dict(
        type='HTC',
        num_stages=5,
        num_layers=[1, 1, 1, 1, 1],
        patch_sizes=[4, 4, 4, 4, 4],
        strides=[2, 2, 2, 2, 2],
        paddings=[1, 1, 1, 1, 1],
        num_heads=[1, 2, 5, 8, 16],
        chratio=[16, 8, 4, 2, 1],
        out_indices=(0, 1, 2, 3, 4),
        mlp_ratios=[8, 8, 8, 4, 4  ],#[8, 8, 8, 4, 4 ]
        window_size=[16, 16, 16, 8, 8],  # 16, 16, 16, 8
        head_dim=32,
        nheads=[2, 4, 10, 16, 32],
        kv_downsample_mode='ada_avgpool',
        kv_per_wins=[4, 4, 4, 4, 4],
        kv_downsample_kernels=[4, 2, 1, 1, 1],#[4, 2, 1, 1]
        kv_downsample_ratios=[4, 2, 1, 1, 1], 
        topks=[1, 4, 8, 16, -2],#[1, 4, 16, 4, 4],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        param_routing=False,
        diff_routing=False,
        soft_routing=False,
        pre_norm=True,
        pe=None,
        ),
        neck=dict(
        type='Deconv_FPN_Neck',
        in_channels = [64,128,320,512,1024],
        out_channels = 256,#[64,128,320,512,1024]
        num_deconv_layers=3,
        num_deconv_filters=(512, 320, 128),
        num_deconv_kernels=(4, 4, 4),
    ),
    distance_head = dict(
        type='TopdownSimple_Distance_DeConvHead',
        in_channels=256,#256
        feat_channels=256,
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint_reg=dict(type='JointsMSELoss', use_target_weight=True,loss_weight=3000)
    ),
    keypoint_head=dict(
        type='TopdownSimple_Heatmap_ConvHead',
        in_channels=256,
        feat_channels=256,
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True,loss_weight=1000)),
    keypoint_head_global=dict(
        type='TopdownSimple_Heatmap_ConvHead_global',
        in_channels=317,
        feat_channels=317,
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True,loss_weight=3000)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11,
        use_udp=False))


data_cfg = dict(
    image_size=[1024,1216],
    heatmap_size=[256,304],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file=''
)
albu_train_transforms = [
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.5],
        contrast_limit=[0.1, 0.5],
        p=0.2),
    dict(type='ChannelShuffle', p=0.7),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=0.8),
            dict(type='GaussianBlur', blur_limit=3, p=1.0)
        ],
        p=0.3),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.3),#0.4
    dict(type='PhotometricDistortion'),
    dict(type='TopDownGetRandomScaleRotation'),
    dict(type='TopDownRandomTranslation'),
    dict(type='TopDownAffine'),
    dict(
        type='NormalizeTensor', 
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        to_rgb=True),
    dict(type='ToTensor'),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(type='Topdown_Heatmap_Target_Size',sigma=2,size=2),
    dict(type='Topdown_Heatmap_Target_Size_down',sigma=4,size=4),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight','target_reg','target_weight_reg','target_reg_down','target_weight_reg_down'],
        meta_keys=[
            'image_file','center', 'scale', 'joints_3d', 'joints_3d_visible',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='ToTensor'),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file','center', 'scale', 'rotation', 'bbox_score','joints_3d',
            'flip_pairs'
        ]),
]

samples_per_gpu_new=1 #Sample 
workers_per_gpu_new=1 #worker  
test_pipeline = val_pipeline
datatype='TopDownISBI2023Dataset'
data_root = './ISBI2023_dataset/'
data = dict(
    samples_per_gpu=samples_per_gpu_new,
    workers_per_gpu=workers_per_gpu_new,
    val_dataloader=dict(samples_per_gpu=samples_per_gpu_new),
    test_dataloader=dict(samples_per_gpu=samples_per_gpu_new),
    train=dict(
        type=datatype,
        ann_file=f'{data_root}/JSON/train.json',
        img_prefix=f'{data_root}/2D_img/train',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type=datatype,
        ann_file=f'{data_root}/JSON/test.json',
        img_prefix=f'{data_root}/2D_img/test',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type=datatype,
        ann_file=f'{data_root}/JSON/test.json',
        img_prefix=f'{data_root}/2D_img/test',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
work_dir = ''
