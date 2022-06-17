"""
A federated learning training session with clients running td3
"""
import logging

from plato.config import Config


import torch

import numpy as np

import gym

import pybullet_envs

import park

import a2c_park_algorithm
import a2c_park_client
import a2c_park_model
import a2c_park_server
import a2c_park_trainer

import imp
#to run
#python examples/park_env/a2c_plato.py -c examples/park_env/a2c_FashionMNIST_lenet5.yml

env = park.make(Config().algorithm.env_park_name)

seed = Config().server.random_seed

env.seed(seed)
env.reset()
torch.manual_seed(seed)
np.random.seed(seed)




def main():
    """ A Plato federated learning training session with clients running TD3. """
    logging.info("Starting RL Environment's process.")
    #def __init__(self, state_dim, n_actions, env_name, rl_algo):
    #TODO instantiate classes

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model = a2c_park_model.Model(state_dim, n_actions, 
    Config().algorithm.env_name, Config().algorithm.algorithm_name)
    trainer = a2c_park_trainer
    algorithm = a2c_park_algorithm
    client = a2c_park_client.RLClient(model=model,trainer=trainer,algorithm=algorithm)
    server = a2c_park_server.A2CServer(Config().algorithm.algorithm_name, Config().algorithm.env_park_name,
    model=model, algorithm=algorithm, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
