#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np

os.environ['config_file'] = 'configs/Kinetics/kinetics_mm.py'

from plato.config import Config
from plato.models.multimodal import multimodal_module

from mmaction.tests import test_models


def test_full_multimodal_model():
    # support_modalities = ['rgb', "flow", "audio"]
    support_modalities = ['rgb']
    # define the sub-nets to be untrained
    for modality_nm in support_modalities:
        modality_model_nm = modality_nm + "_model"
        if modality_model_nm in Config.multimodal_nets_configs.keys():

            modality_net = Config.multimodal_nets_configs[modality_model_nm]
            modality_net['backbone']['pretrained'] = None

    # define the model
    multi_model = multimodal_module.DynamicMultimodalModule(
        support_modality_names=support_modalities,
        multimodal_nets_configs=Config.multimodal_nets_configs)

    # define the test data
    rgb_input_shape = (1, 3, 3, 8, 32, 32)
    rgb_demo_inputs = test_models.base.generate_recognizer_demo_inputs(
        rgb_input_shape, model_type='3D')

    # flow_input_shape = (1, 3, 3, 8, 32, 32)
    # flow_demo_inputs = test_models.base.generate_recognizer_demo_inputs(
    #     flow_input_shape, model_type='3D')

    # audio_feature_input_shape = (1, 3, 1, 128, 80)
    # audio_demo_inputs = test_models.base.generate_recognizer_demo_inputs(
    #     audio_feature_input_shape, model_type='audio')

    rgb_imgs = rgb_demo_inputs['imgs']
    # flow_imgs = flow_demo_inputs['imgs']
    # audio_feas = flow_demo_inputs['imgs']
    gt_labels = rgb_demo_inputs['gt_labels']

    print("rgb_imgs: ", rgb_imgs.shape)
    # print("flow_imgs: ", flow_imgs.shape)
    # print("audio_feas: ", audio_feas.shape)
    print("gt_labels: ", gt_labels.shape)

    mm_data_container = {
        "rgb": rgb_imgs,
        # "flow": flow_imgs,
        # "audio": audio_feas
    }

    losses = multi_model(
        data_container=mm_data_container,
        label=gt_labels,
        return_loss=True,
    )
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        img_list = [img[None, :] for img in imgs]
        for one_img in img_list:
            recognizer(one_img, None, return_loss=False)


if __name__ == "__main__":
    _ = Config()
    test_full_multimodal_model()