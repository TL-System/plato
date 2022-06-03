



import logging

import gym
import torch
import numpy as np

from torch import nn


env = gym.make("BipedalWalker-v3")

seed = 0

env.reset(seed=seed)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])