"""
Test all datasources

"""

import os

from plato.config import Config
# from plato.mmconfig import mmConfig
from plato.datasources.multimodal import kinetics

from plato.datasources.multimodal import gym
from plato.datasources.multimodal import flickr30k_entities
from plato.datasources.multimodal import coco
from plato.datasources.multimodal import referitgame

# put the configuration file here:
# os.environ['config_file'] = 'configs/Kinetics/kinetics_mm.py'
os.environ['config_file'] = 'configs/Gym/gym_mm.py'


def test_coco_provide():
    """ Test the coco datasource """
    os.environ['config_file'] = 'configs/COCO/coco.yml'
    coco.DataSource()


def test_kinetics_provide():
    """ Test the kinetics datasource """
    kinetics_data_source = kinetics.DataSource()
    kinetics_data_source.get_test_set()


def test_gym_provide():
    """ Test the gym datasource """
    gym_data_source = gym.DataSource()
    gym_data_source.extract_videos_rgb_flow_audio()


def test_flickr30k_entities_provide():
    """ Test the flickr30k_entities datasource """
    os.environ[
        'config_file'] = 'configs/Flickr30KEntities/flickr30Kentities.yml'
    fe_data_source = flickr30k_entities.DataSource()
    fe_data_source.create_splits_data()
    fe_data_source.integrate_data_to_json()


def test_referitgame_provide():
    """ Test the referitgame datasource """
    os.environ['config_file'] = 'configs/ReferItGame/referitgame.yml'
    rig = referitgame.DataSource()
    rig.get_train_loader(batch_size=Config.trainer.batch_size)


if __name__ == "__main__":
    _ = Config()

    # test_kinetics_provide()

    test_gym_provide()
