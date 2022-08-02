"""
A federated learning training session with clients running td3
"""
import logging

import td3_learning_client
import td3_learning_trainer
import td3_learning_server
import td3_learning_model
import td3_learning_algorithm

from plato.config import Config


import torch

import numpy as np

import gym

import pybullet_envs

#to run
#python examples/td3_learning/td3.py -c examples/td3_learning/td3_FashionMNIST_lenet5.yml


def main():
    """ A Plato federated learning training session with clients running TD3. """
    logging.info("Starting RL Environment's process.")

    model = td3_learning_model.Model
    trainer = td3_learning_trainer.Trainer
    algorithm = td3_learning_algorithm.Algorithm
    client = td3_learning_client.RLClient(model = model, trainer=trainer, algorithm = algorithm)
    server = td3_learning_server.TD3Server(Config().algorithm.algorithm_name, Config().algorithm.env_gym_name,
    model=model, algorithm = algorithm, trainer=trainer)

    server.run(client)

if __name__ == "__main__":
    main()
