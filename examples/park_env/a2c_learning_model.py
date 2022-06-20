import numpy as np
import torch
import gym
from torch import nn
import matplotlib.pyplot as plt
import park

# Actor module, categorical actions only
class A2CActor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
            nn.Softmax(dim = 0)
        )
    
    def forward(self, X):
        return self.model(X)

# Critic module
class A2CCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, X):
        return self.model(X)

class Model:
    """Wrapper class that holds both models"""
    def __init__(self, state_dim, n_actions, env_name, rl_algo):
        self.state_dim = state_dim
        self.n_action = n_actions
        self.env_name = env_name
        self.rl_algo = rl_algo
        self.actor = A2CActor(state_dim, n_actions)
        self.critic = A2CCritic(state_dim)
    
    def get_env_name(self):
        return self.env_name

    def get_rl_algo(self):
        return self.rl_algo

    def get_state_dim(self):
        return self.state_dim

    def cpu(self):
        self.actor.cpu()
        self.critic.cpu()

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    @staticmethod
    def get_model(*args):
        """ Obtaining an instance of this model. """
        return Model()