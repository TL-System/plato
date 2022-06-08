



import logging

import gym
import torch
import numpy as np

from torch import nn
import pybullet_envs


env = gym.make("HalfCheetahBulletEnv-v0")

seed = 1

env.seed(seed)
env.reset()
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
max_episode_steps = env._max_episode_steps