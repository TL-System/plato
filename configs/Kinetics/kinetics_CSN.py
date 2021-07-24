# clients settings
clients = dict(
    type="simple",  # Type
    total_clients=50,  # The total number of clients
    per_round=40,  # The number of clients selected in each round
    do_test=True)  # Should the clients compute test accuracy locally?

# server settings
server = dict(address="127.0.0.1", port=8000)

# dataset settings
dataset_type = 'RawframeDataset'
data_root_name = 'rawframes_train'
data_root_test_name = 'rawframes_test'
data_root_val_name = 'rawframes_val'
ann_file_train_name = 'train_list_rawframes.txt'
ann_file_test_name = 'val_list_rawframes.txt'
ann_file_val_name = 'val_list_rawframes.txt'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_bgr=False)
train_pipeline = [
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
val_pipeline = [
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
test_pipeline = [
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
data = dict(
    datasource="Kinetics700",
    data_path="./data/Kinetics",
    download_url=
    "https://storage.googleapis.com/deepmind-media/Datasets/kinetics700_2020.tar.gz",
    num_workers=4,
    failed_save_file="failed_record.txt",
    log_file=None,
    compress=False,
    verbose=False,
    skip=False,
    videos_per_gpu=3,
    workers_per_gpu=4,
    train=dict(type=dataset_type,
               ann_file=ann_file_train_name,
               data_prefix=data_root_name,
               pipeline=train_pipeline),
    val=dict(type=dataset_type,
             ann_file=ann_file_val_name,
             data_prefix=data_root_val_name,
             pipeline=val_pipeline),
    test=dict(type=dataset_type,
              ann_file=ann_file_test_name,
              data_prefix=data_root_test_name,
              pipeline=test_pipeline),
    sampler="dis_noniid")

# Train settings

## optimizer
optimizer = dict(type='SGD', lr=0.000125, momentum=0.9,
                 weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# #learning policy
lr_config = dict(policy='step',
                 step=[32, 48],
                 warmup='linear',
                 warmup_ratio=0.1,
                 warmup_by_epoch=True,
                 warmup_iters=16)
total_epochs = 1000
checkpoint_config = dict(interval=2)
evaluation = dict(interval=5,
                  metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

trainer = dict(type="basic",
               batch_size=24,
               optimizer=optimizer,
               optimizer_config=optimizer_config,
               learning_rate_config=lr_config,
               rounds=total_epochs,
               parallelized=False,
               max_concurrency=clients["per_round"],
               target_accuracy=0.67)
algorithm = dict(
    type="fedavg",  # Aggregation algorithm
)  # The number of local aggregation rounds on edge servers before sending
# aggreagted weights to the central server

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb'  # noqa: E501
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
