#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np

os.environ['config_file'] = 'configs/Kinetics/kinetics_mm.py'

from plato.config import Config
from plato.models.multimodal import multimodal_module

from mmaction.tests import test_models


def test_full_multimodal_model():

    # support_modalities = ['rgb']
    # support_modalities = ['rgb', 'flow']
    support_modalities = ['rgb', "flow", "audio"]
    # define the sub-nets to be untrained
    for modality_nm in support_modalities:
        modality_model_nm = modality_nm + "_model"
        if modality_model_nm in Config.multimodal_nets_configs.keys():
            modality_net = Config.multimodal_nets_configs[modality_model_nm]
            modality_net['backbone']['pretrained'] = None
            if "pretrained2d" in list(modality_net['backbone'].keys()):
                modality_net['backbone']['pretrained2d'] = False

    # define the model
    multi_model = multimodal_module.DynamicMultimodalModule(
        support_modality_names=support_modalities,
        multimodal_nets_configs=Config.multimodal_nets_configs,
        is_fused_head=False)

    # define the test data
    rgb_input_shape = (1, 3, 3, 8, 32, 32)
    rgb_demo_inputs = test_models.base.generate_recognizer_demo_inputs(
        rgb_input_shape, model_type='3D')

    flow_input_shape = (1, 3, 3, 8, 32, 32)
    flow_demo_inputs = test_models.base.generate_recognizer_demo_inputs(
        flow_input_shape, model_type='3D')

    audio_feature_input_shape = (1, 3, 1, 128, 80)
    audio_demo_inputs = test_models.base.generate_recognizer_demo_inputs(
        audio_feature_input_shape, model_type='audio')

    rgb_imgs = rgb_demo_inputs['imgs']
    flow_imgs = flow_demo_inputs['imgs']
    audio_feas = audio_demo_inputs['imgs']
    gt_labels = rgb_demo_inputs['gt_labels']

    print("rgb_imgs: ", rgb_imgs.shape)
    print("flow_imgs: ", flow_imgs.shape)
    print("audio_feas: ", audio_feas.shape)
    print("gt_labels: ", gt_labels.shape)

    mm_data_container = {
        "rgb": rgb_imgs,
        "flow": flow_imgs,
        "audio": audio_feas
    }

    outputs = multi_model(
        data_container=mm_data_container,
        label=gt_labels,
        return_loss=True,
    )
    print(outputs)


def test_rgb_build():
    from mmaction.models import build_model
    # define the test data

    rgb_input_shape = (1, 3, 3, 8, 32, 32)
    rgb_demo_inputs = test_models.base.generate_recognizer_demo_inputs(
        rgb_input_shape, model_type='3D')

    rbg_model = build_model(Config.multimodal_nets_configs["rgb_model"])

    rgb_imgs = rgb_demo_inputs['imgs']
    # flow_imgs = flow_demo_inputs['imgs']
    # audio_feas = flow_demo_inputs['imgs']
    gt_labels = rgb_demo_inputs['gt_labels']

    print("rgb_imgs: ", rgb_imgs.shape)
    # print("flow_imgs: ", flow_imgs.shape)
    # print("audio_feas: ", audio_feas.shape)
    print("gt_labels: ", gt_labels.shape)

    losses = rbg_model(rgb_imgs, gt_labels)


if __name__ == "__main__":
    _ = Config()
    test_full_multimodal_model()