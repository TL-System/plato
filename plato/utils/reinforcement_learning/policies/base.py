import copy
import logging
import os
import random
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from plato.config import Config


class ReplayMemory:
    """A simple example of replay memory buffer."""

    def __init__(self, state_dim, action_dim, capacity, seed):
        random.seed(seed)
        self.device = Config().device()
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.capacity, state_dim))
        self.action = np.zeros((self.capacity, action_dim))
        self.reward = np.zeros((self.capacity, 1))
        self.next_state = np.zeros((self.capacity, state_dim))
        self.done = np.zeros((self.capacity, 1))

    def push(self, data):
        self.state[self.ptr] = data[0]
        self.action[self.ptr] = data[1]
        self.reward[self.ptr] = data[2]
        self.next_state[self.ptr] = data[3]
        self.done[self.ptr] = data[4]

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self):
        ind = np.random.randint(0, self.size, size=int(Config().algorithm.batch_size))

        state = self.state[ind]
        action = self.action[ind]
        reward = self.reward[ind]
        next_state = self.next_state[ind]
        done = self.done[ind]

        return state, action, reward, next_state, done

    def __len__(self):
        return self.size


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class Policy(ABC):
    """A simple example of DRL policy."""

    def __init__(self, state_dim, action_dim):
        self.max_action = Config().algorithm.max_action
        self.device = Config().device()
        self.total_it = 0

        # Initialize NNs
        self.actor = Actor(state_dim, action_dim, self.max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=Config().algorithm.learning_rate
        )

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=Config().algorithm.learning_rate
        )
        # Initialize replay memory
        self.replay_buffer = ReplayMemory(
            state_dim,
            action_dim,
            Config().algorithm.replay_size,
            Config().algorithm.replay_seed,
        )

    def save_model(self, ep=None):
        """Saving the model to a file."""
        model_name = Config().algorithm.model_name
        model_path = f"./models/{model_name}/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if ep is not None:
            model_path += "iter" + str(ep) + "_"

        torch.save(self.actor.state_dict(), model_path + "actor.pth")
        torch.save(
            self.actor_optimizer.state_dict(), model_path + "actor_optimizer.pth"
        )
        torch.save(self.critic.state_dict(), model_path + "critic.pth")
        torch.save(
            self.critic_optimizer.state_dict(), model_path + "critic_optimizer.pth"
        )

        logging.info("[RL Agent] Model saved to %s.", model_path)

    def load_model(self, ep=None):
        """Loading pre-trained model weights from a file."""
        model_name = Config().algorithm.model_name
        model_path = f"./models/{model_name}/"
        if ep is not None:
            model_path += "iter" + str(ep) + "_"

        logging.info("[RL Agent] Loading a model from %s.", model_path)

        self.actor.load_state_dict(torch.load(model_path + "actor.pth"))
        self.actor_optimizer.load_state_dict(
            torch.load(model_path + "actor_optimizer.pth")
        )
        self.critic.load_state_dict(torch.load(model_path + "critic.pth"))
        self.critic_optimizer.load_state_dict(
            torch.load(model_path + "critic_optimizer.pth")
        )

    @abstractmethod
    def select_action(self, state, hidden=None, test=False):
        """Select action from policy."""
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def update(self):
        """Update policy."""
        raise NotImplementedError("Please Implement this method")
