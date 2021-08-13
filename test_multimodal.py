#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ['config_file'] = 'configs/Kinetics/kinetics_mm.py'

from plato.config import Config
from plato.models.multimodal import multimodal_module
from plato.datasources.multimodal import kinetics

support_modalities = ['rgb', "flow", "audio"]


def test_multimodal():
    kinetics_source = kinetics.DataSource()

    multi_model = multimodal_module.DynamicMultimodalModule(
        support_modality_names=support_modalities,
        multimodal_nets_configs=Config.multimodal_nets_configs,
        is_fused_head=True)


if __name__ == "__main__":
    _ = Config()
    test_multimodal()