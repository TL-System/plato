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

from torch import nn

import torch

import numpy as np

import gym

#to run
#python examples/td3_learning/td3.py -c examples/td3_learning/td3_FashionMNIST_lenet5.yml


def main():
    """ A Plato federated learning training session with clients running TD3. """
    logging.info("Starting RL Environment's process.")

    CONST_ENV_NAME = "halfcheetah"

    CONST_ENV_GYM_NAME = "HalfCheetahBulletEnv-v0"

    CONST_ALGORITHM_NAME = 'td3_'

    """ A Plato federated learning training session using a custom model. """
    env = gym.make(CONST_ENV_GYM_NAME)

    seed = Config().server.random_seed

    env.seed(seed)
    env.reset()
    print('exectued')
    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_episode_steps = env._max_episode_steps

    model = td3_learning_model.Model(state_dim, action_dim, max_action, env, max_episode_steps
    , CONST_ENV_NAME, CONST_ALGORITHM_NAME)
    trainer = td3_learning_trainer.Trainer
    algorithm = td3_learning_algorithm.Algorithm
    client = td3_learning_client.RLClient(model = model, trainer=trainer, algorithm = algorithm)
    server = td3_learning_server.TD3Server(CONST_ALGORITHM_NAME, CONST_ENV_GYM_NAME, model=model, algorithm = algorithm, trainer=trainer)

    server.run(client)

if __name__ == "__main__":
    main()
