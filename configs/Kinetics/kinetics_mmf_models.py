# rgb model settings
rgb_model = dict(
    type='Recognizer3D',
    backbone=dict(type='ResNet2Plus1d',
                  depth=50,
                  pretrained=None,
                  pretrained2d=False,
                  norm_eval=False,
                  conv_cfg=dict(type='Conv2plus1d'),
                  norm_cfg=dict(type='SyncBN', requires_grad=True, eps=1e-3),
                  conv1_kernel=(3, 7, 7),
                  conv1_stride_t=1,
                  pool1_stride_t=1,
                  inflate=(1, 1, 1, 1),
                  spatial_strides=(1, 2, 2, 2),
                  temporal_strides=(1, 2, 2, 2),
                  zero_init_residual=False),
    cls_head=dict(type='I3DHead',
                  num_classes=400,
                  in_channels=512,
                  spatial_type='avg',
                  dropout_ratio=0.5,
                  init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# optical flow model settings
flow_model = dict(
    type='Recognizer3D',
    backbone=dict(type='ResNet2Plus1d',
                  depth=50,
                  pretrained=None,
                  pretrained2d=False,
                  norm_eval=False,
                  conv_cfg=dict(type='Conv2plus1d'),
                  norm_cfg=dict(type='SyncBN', requires_grad=True, eps=1e-3),
                  conv1_kernel=(3, 7, 7),
                  conv1_stride_t=1,
                  pool1_stride_t=1,
                  inflate=(1, 1, 1, 1),
                  spatial_strides=(1, 2, 2, 2),
                  temporal_strides=(1, 2, 2, 2),
                  zero_init_residual=False),
    cls_head=dict(type='I3DHead',
                  num_classes=400,
                  in_channels=512,
                  spatial_type='avg',
                  dropout_ratio=0.5,
                  init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# audio model settings
audio_model = dict(
    type='AudioRecognizer',
    backbone=dict(type='ResNet', depth=50, in_channels=1, norm_eval=False),
    cls_head=dict(type='AudioTSNHead',
                  num_classes=400,
                  in_channels=512,
                  dropout_ratio=0.5,
                  init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

fuse_model = dict(
    type='FullyConnectedHead',
    num_classes=400,
    in_channels=rgb_model['cls_head']['in_channels'] +
    flow_model['cls_head']['in_channels'] +
    audio_model['cls_head']['in_channels'],
    hidden_layer_size=[1024, 512],
    dropout_ratio=0.5,
)
