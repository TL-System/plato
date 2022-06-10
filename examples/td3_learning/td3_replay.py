

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pybullet_envs
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from torch.autograd import Variable
from collections import deque
from queue import Queue
import pickle
import math

class ReplayBuffer:
    
    def __init__(self, max_size):
        print("actually being initialized")
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
        else:
            self.storage.append(transition)
            self.ptr = (self.ptr + 1) % self.max_size

    def empty(self):
        self.storage = []
        self.ptr = 0

    def sample(self, batch_size, low = 0, high = None):
        if high is None: high=len(self.storage)
        if low > high: #range of samples overflow buffer
            actual_indices = [i for i in range(low, len(self.storage))]
            actual_indices = actual_indices + [i for i in range(0, high+1)]
            ind_int = np.random.randint(0, len(actual_indices), size=batch_size)
            ind = [actual_indices[i] for i in ind_int.tolist()]
        else:
            ind = np.random.randint(low=low, high=high, size=batch_size)
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind: 
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)