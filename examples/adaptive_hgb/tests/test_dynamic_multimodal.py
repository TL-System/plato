"""
Test the dynamic multimodal model definition

"""

import os
import logging

from mmaction.tests import test_models
from mmaction.models import build_model

from plato.config import Config
from plato.models.multimodal import multimodal_module

os.environ['config_file'] = 'configs/Kinetics/kinetics_mm.py'


def test_full_multimodal_model():
    """ Test the multimodal model with all modalities """
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
        is_fused_head=True)

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

    logging.debug("rgb_imgs: %s", rgb_imgs.shape)
    logging.debug("flow_imgs: %s", flow_imgs.shape)
    logging.debug("audio_feas: %s", audio_feas.shape)
    logging.debug("gt_labels: %s", gt_labels.shape)

    mm_data_container = {
        "rgb": rgb_imgs,
        "flow": flow_imgs,
        "audio": audio_feas
    }

    opt_scores, opt_losses = multi_model(
        data_container=mm_data_container,
        label=gt_labels,
        return_loss=True,
    )

    for key in list(opt_scores.keys()):
        print(key)
        print(opt_scores[key])

    for key in list(opt_losses.keys()):
        print(key)
        print(opt_losses[key])


def test_rgb_build():
    """ Test the code for building the rgb model"""

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
    print("the rgb losses: ", losses)


if __name__ == "__main__":
    _ = Config()
    test_full_multimodal_model()
