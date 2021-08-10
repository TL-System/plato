#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ['config_file'] = 'configs/Kinetics/kinetics_mm.py'

from plato.config import Config
from plato.models.multimodal import multimodal_module
from plato.datasources.multimodal import kinetics

kinetics_source = kinetics.DataSource()

multi_model = multimodal_module.DynamicMultimodalModule(
    support_modality_names=['rgb', "flow", "audio"],
    multimodal_nets_configs=Config.multimodal_nets_configs)
