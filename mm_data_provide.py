#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

# put the configuration file here:

import numpy as np

from plato.config import Config
from plato.datasources.multimodal import kinetics
from plato.datasources.multimodal import flickr30k_entities
from plato.datasources.multimodal import coco
from plato.datasources.multimodal import referitgame


def test_coco_provide():
    os.environ['config_file'] = 'configs/COCO/coco.yml'
    test_coco = coco.DataSource()


def test_kinetics_provide():
    os.environ['config_file'] = 'configs/Kinetics/kinetics.yml'
    kinetics_data_source = kinetics.DataSource()
    kinetics_data_source.get_train_set()


def test_flickr30k_entities_provide():
    os.environ[
        'config_file'] = 'configs/Flickr30KEntities/flickr30Kentities.yml'
    fe_data_source = flickr30k_entities.DataSource()
    fe_data_source.create_splits_data()
    fe_data_source.integrate_data_to_json()


def test_referitgame_provide():
    os.environ['config_file'] = 'configs/ReferItGame/referitgame.yml'
    rig = referitgame.DataSource()
    rig.get_train_loader(batch_size=Config.trainer.batch_size)


if __name__ == "__main__":
    _ = Config()

    test_coco_provide()
