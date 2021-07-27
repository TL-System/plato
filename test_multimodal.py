#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ['config_file'] = 'configs/Kinetics/kinetics_mm_CSN.py'

from plato.config import Config
from plato.models.multimodal import multimodal_net
from plato.datasources.multimodal import kinetics

multi_model = multimodal_net.MM3F(
    multimoda_model_configs=Config.multimodal_data_model)

kinetics_source = kinetics.DataSource()
