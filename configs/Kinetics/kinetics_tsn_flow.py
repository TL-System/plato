# the data configuration for the flow dataset
""" Note: we did one slightly modification to the original configuration from the tsn config file
        - Converted the SampleFrames to the one used in the kinetics_csn_rgb
    because we want to train a multimodal model
"""

flow_dataset_type = 'RawframeDataset'
flow_data_root_train_name = 'rawframes_train'
flow_data_root_test_name = 'rawframes_test'
flow_data_root_val_name = 'rawframes_val'
flow_ann_file_train_name = 'train_list_rawframes.txt'
flow_ann_file_test_name = 'val_list_rawframes.txt'
flow_ann_file_val_name = 'val_list_rawframes.txt'
flow_img_norm_cfg = dict(mean=[128, 128], std=[128, 128])

flow_train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **flow_img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW_Flow'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
flow_val_pipeline = [
    dict(type='SampleFrames',
         clip_len=32,
         frame_interval=2,
         num_clips=1,
         test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **flow_img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW_Flow'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
flow_test_pipeline = [
    dict(type='SampleFrames',
         clip_len=32,
         frame_interval=2,
         num_clips=10,
         test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='TenCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **flow_img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW_Flow'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]


flow_data_train = dict(
        type=flow_dataset_type,
        ann_file=flow_ann_file_train_name,
        data_prefix=flow_data_root_train_name,
        filename_tmpl='{}_{:05d}.jpg',
        modality='Flow',
        pipeline=flow_train_pipeline),

flow_data_val = dict(
        type=flow_dataset_type,
        ann_file=flow_ann_file_val_name,
        data_prefix=flow_data_root_val_name,
        filename_tmpl='{}_{:05d}.jpg',
        modality='Flow',
        pipeline=flow_val_pipeline),

flow_data_test = dict(
        type=flow_dataset_type,
        ann_file=flow_ann_file_test_name,
        data_prefix=flow_data_root_test_name,
        filename_tmpl='{}_{:05d}.jpg',
        modality='Flow',
        pipeline=flow_test_pipeline))
