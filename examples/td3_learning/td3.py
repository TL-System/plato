"""
A federated learning training session using td3
"""
import logging

import td3_learning_client
import td3_trainer

import gym
import torch
import numpy as np

from td3_learning import td3_learning_server
from torch import nn

env = gym.make("Cartpole-v0")

seed = 0

env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

def main():
    """ A Plato federated learning training session using TD3. """
    logging.info("Starting RL Environment's process.")

    """ A Plato federated learning training session using a custom model. """
    model = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    trainer = td3_trainer.Trainer(state_dim, action_dim)
    client = td3_learning_client.RLClient(trainer=trainer, model=model)
    server = td3_learning_server.TD3Server(model=model)
    server.run(client)

if __name__ == "__main__":
    main()
