"""
Reference:

https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from plato.config import Config
from plato.utils.reinforcement_learning.policies import base


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


class Policy(base.Policy):
    def __init__(self, state_dim, action_space):
        super().__init__(state_dim, action_space)

        # Initialize NNs
        self.actor = Actor(state_dim, action_space.shape[0],
                           self.max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=Config().algorithm.learning_rate)

        self.critic = Critic(state_dim, action_space.shape[0]).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=Config().algorithm.learning_rate)
        # Initialize replay memory
        self.replay_buffer = base.ReplayMemory(state_dim,
                                               action_space.shape[0],
                                               Config().algorithm.replay_size,
                                               Config().algorithm.replay_seed)

    def select_action(self, state):
        """ Select action from policy. """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):
        """ Update policy. """
        for _ in range(Config().algorithm.update_iteration):
            # Sample replay buffer
            state, action, reward, next_state, done = self.replay_buffer.sample(
            )
            state = torch.FloatTensor(state).to(self.device).unsqueeze(1)
            action = torch.FloatTensor(action).to(self.device).unsqueeze(1)
            reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
            next_state = torch.FloatTensor(next_state).to(
                self.device).unsqueeze(1)
            done = torch.FloatTensor(done).to(self.device).unsqueeze(1)

            # Compute the target Q value
            target_Q = self.critic_target(next_state,
                                          self.actor_target(next_state))
            target_Q = reward + (
                (1 - done) * Config().algorithm.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(Config().algorithm.tau * param.data +
                                        (1 - Config().algorithm.tau) *
                                        target_param.data)

            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(Config().algorithm.tau * param.data +
                                        (1 - Config().algorithm.tau) *
                                        target_param.data)

        return critic_loss.item(), actor_loss.item()
