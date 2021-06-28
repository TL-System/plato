# the configuration file for the audio dataset of the kinetics70

# dataset settings
audio_dataset_type = 'AudioFeatureDataset'
audio_data_root_train_name = 'audio_feature_train'
audio_data_root_test_name = 'audio_feature_val'
audio_data_root_val_name = 'audio_feature_val'
audio_ann_file_train_name = 'kinetics700_train_list_audio_feature.txt'
audio_ann_file_test_name = 'kinetics700_val_list_audio_feature.txt'
audio_ann_file_val_name = 'kinetics700_val_list_audio_feature.txt'

audio_train_pipeline = [
    dict(type='LoadAudioFeature'),
    dict(type='SampleFrames', clip_len=64, frame_interval=1, num_clips=1),
    dict(type='AudioFeatureSelector'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
audio_val_pipeline = [
    dict(type='LoadAudioFeature'),
    dict(type='SampleFrames',
         clip_len=64,
         frame_interval=1,
         num_clips=1,
         test_mode=True),
    dict(type='AudioFeatureSelector'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
audio_test_pipeline = [
    dict(type='LoadAudioFeature'),
    dict(type='SampleFrames',
         clip_len=64,
         frame_interval=1,
         num_clips=1,
         test_mode=True),
    dict(type='AudioFeatureSelector'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]

audio_data_train = dict(type=audio_dataset_type,
                        ann_file=audio_ann_file_train_name,
                        data_prefix=audio_data_root_train_name,
                        pipeline=audio_train_pipeline)

audio_data_val = dict(type=audio_dataset_type,
                      ann_file=audio_ann_file_val_name,
                      data_prefix=audio_data_root_val_name,
                      pipeline=audio_val_pipeline)

audio_data_test = dict(type=audio_dataset_type,
                       ann_file=audio_ann_file_test_name,
                       data_prefix=audio_data_root_test_name,
                       pipeline=audio_test_pipeline)

audio_data = dict(train=audio_data_train,
                  val=audio_data_val,
                  test=audio_data_test)
