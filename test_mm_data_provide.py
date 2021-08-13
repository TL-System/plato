#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# put the configuration file here:
os.environ['config_file'] = 'configs/Kinetics/kinetics_mm.py'
# os.environ['config_file'] = 'configs/Gym/gym_mm.py'

import numpy as np

from plato.config import Config
from plato.datasources.multimodal import kinetics
from plato.datasources.multimodal import kinetics_mm
from plato.datasources.multimodal import gym
from plato.datasources.multimodal import flickr30k_entities
from plato.datasources.multimodal import coco
from plato.datasources.multimodal import referitgame


def test_coco_provide():
    os.environ['config_file'] = 'configs/COCO/coco.yml'
    test_coco = coco.DataSource()


def test_kinetics_provide():

    kinetics_data_source = kinetics_mm.DataSource()
    # kinetics_data_source.get_train_set()


def test_gym_provide():

    gym_data_source = gym.DataSource()
    gym_data_source.extract_videos_rgb_flow_audio()


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

    test_kinetics_provide()

    # test_gym_provide()
