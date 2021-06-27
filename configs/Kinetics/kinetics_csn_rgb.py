# the data configuration for the rgb dataset

rgb_dataset_type = 'RawframeDataset'
rgb_data_root_train_name = 'rawframes_train'
rgb_data_root_test_name = 'rawframes_val'
rgb_data_root_val_name = 'rawframes_val'
rgb_ann_file_train_name = 'train_list_rawframes.txt'
rgb_ann_file_test_name = 'val_list_rawframes.txt'
rgb_ann_file_val_name = 'val_list_rawframes.txt'
rgb_img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_bgr=False)

rgb_train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='FrameSelector'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
rgb_val_pipeline = [
    dict(type='SampleFrames',
         clip_len=32,
         frame_interval=2,
         num_clips=1,
         test_mode=True),
    dict(type='FrameSelector'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
rgb_test_pipeline = [
    dict(type='SampleFrames',
         clip_len=32,
         frame_interval=2,
         num_clips=10,
         test_mode=True),
    dict(type='FrameSelector'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

rgb_data_train = dict(type=rgb_dataset_type,
                      ann_file=rgb_ann_file_train_name,
                      data_prefix=rgb_data_root_train_name,
                      pipeline=rgb_train_pipeline)

rgb_data_val = dict(type=rgb_dataset_type,
                    ann_file=rgb_ann_file_val_name,
                    data_prefix=rgb_data_root_val_name,
                    pipeline=rgb_val_pipeline)

rgb_data_test = dict(type=rgb_dataset_type,
                     ann_file=rgb_ann_file_test_name,
                     data_prefix=rgb_data_root_test_name,
                     pipeline=rgb_test_pipeline)
