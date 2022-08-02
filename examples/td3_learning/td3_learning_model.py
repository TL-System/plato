import copy

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

import gym

import pybullet_envs

from plato.config import Config
class TD3Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(TD3Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        #nn.init.uniform_(self.l1.weight.data)
        self.l2 = nn.Linear(400, 300)
        #nn.init.uniform_(self.l2.weight.data)
        self.l3 = nn.Linear(300, action_dim)
        #nn.init.uniform_(self.l3.weight.data)
        self.max_action = max_action

    def forward(self, x, hidden=None):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        # Normalize/Scaling aggregation weights so that the sum is 1
        #x += 1  # [-1, 1] -> [0, 2]
        #x /= x.sum()
        return x


class TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(TD3Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        # Forward-Propagation on the first Critic Neural Network
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        # Forward-Propagation on the second Critic Neural Network
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

env = gym.make(Config().algorithm.env_gym_name)

seed = Config().server.random_seed

env.seed(seed)
env.reset()
torch.manual_seed(seed)
np.random.seed(seed)
class Model:
    """A wrapper class that holds both actor and critic models"""
    def __init__(self):


        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        max_episode_steps = env._max_episode_steps
            


        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.max_episode_steps = max_episode_steps

        self.env_name = Config().algorithm.env_name
        self.rl_algo = Config().algorithm.algorithm_name
        
        self.actor = TD3Actor(state_dim, action_dim, max_action)
        self.critic = TD3Critic(state_dim, action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
    
    def get_env_name(self):
        return self.env_name

    def get_rl_algo(self):
        return self.rl_algo

    def get_state_dim(self):
        return self.state_dim

    def get_action_dim(self):
        return self.action_dim

    def get_max_action(self):
        return self.max_action

    def get_max_episode_steps(self):
        return self.max_episode_steps

    def cpu(self):
        self.actor.cpu()
        self.critic.cpu()
        self.actor_target.cpu()
        self.critic_target.cpu()

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.actor_target.to(device)
        self.critic_target.to(device)

    def eval(self):
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    @staticmethod
    def get_model(*args):
        """ Obtaining an instance of this model. """
        return Model()