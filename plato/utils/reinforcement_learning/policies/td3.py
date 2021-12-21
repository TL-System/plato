"""
Reference:

https://github.com/AntoineTheb/RNN-RL
"""
import copy
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)

from plato.config import Config


class ReplayMemory:
    def __init__(self, state_dim, action_dim, hidden_size, capacity, seed):
        random.seed(seed)
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0

        if Config().algorithm.recurrent_actor:
            self.h = np.zeros((self.capacity, hidden_size))
            self.nh = np.zeros((self.capacity, hidden_size))
            self.c = np.zeros((self.capacity, hidden_size))
            self.nc = np.zeros((self.capacity, hidden_size))
            self.state = [0] * self.capacity
            self.action = [0] * self.capacity
            self.reward = [0] * self.capacity
            self.next_state = [0] * self.capacity
            self.done = [0] * self.capacity
        else:
            self.state = np.zeros((self.capacity, state_dim))
            self.action = np.zeros((self.capacity, action_dim))
            self.reward = np.zeros((self.capacity, 1))
            self.next_state = np.zeros((self.capacity, state_dim))
            self.done = np.zeros((self.capacity, 1))

        self.device = Config().device

    def push(self, data):
        self.state[self.ptr] = data[0]
        self.action[self.ptr] = data[1]
        self.reward[self.ptr] = data[2]
        self.next_state[self.ptr] = data[3]
        self.done[self.ptr] = data[4]

        if Config().algorithm.recurrent_actor:
            self.h[self.ptr] = data[5].detach().cpu()
            self.c[self.ptr] = data[6].detach().cpu()
            self.nh[self.ptr] = data[7].detach().cpu()
            self.nc[self.ptr] = data[8].detach().cpu()

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self):
        ind = np.random.randint(0,
                                self.size,
                                size=int(Config().algorithm.batch_size))

        if not Config().algorithm.recurrent_actor:
            state = self.state[ind]
            action = self.action[ind]
            reward = self.reward[ind]
            next_state = self.next_state[ind]
            done = self.done[ind]

            return state, action, reward, next_state, done

        h = torch.tensor(self.h[ind][None, ...],
                         requires_grad=True,
                         dtype=torch.float).to(self.device)
        c = torch.tensor(self.c[ind][None, ...],
                         requires_grad=True,
                         dtype=torch.float).to(self.device)
        nh = torch.tensor(self.nh[ind][None, ...],
                          requires_grad=True,
                          dtype=torch.float).to(self.device)
        nc = torch.tensor(self.nc[ind][None, ...],
                          requires_grad=True,
                          dtype=torch.float).to(self.device)

        state = [torch.FloatTensor(self.state[i]).to(self.device) for i in ind]
        action = [
            torch.FloatTensor(self.action[i]).to(self.device) for i in ind
        ]
        reward = [self.reward[i] for i in ind]
        next_state = [
            torch.FloatTensor(self.next_state[i]).to(self.device) for i in ind
        ]
        done = [self.done[i] for i in ind]

        return state, action, reward, next_state, done, h, c, nh, nc

    def __len__(self):
        return self.size


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, max_action):
        super(Actor, self).__init__()
        self.action_dim = action_dim

        if Config().algorithm.recurrent_actor:
            self.l1 = nn.LSTM(state_dim, hidden_size, batch_first=True)
        else:
            self.l1 = nn.Linear(state_dim, hidden_size)

        self.l2 = nn.Linear(hidden_size, hidden_size)
        if Config().algorithm.recurrent_actor:
            self.l3 = nn.Linear(hidden_size, 1)
        else:
            self.l3 = nn.Linear(hidden_size, action_dim)

        self.max_action = max_action

    def forward(self, state, hidden):
        if Config().algorithm.recurrent_actor:
            if hasattr(Config().clients, 'varied') and Config().clients.varied:
                # Pad the first state to full dims
                if len(state) != 1:
                    pilot = state[0]
                else:
                    pilot = state
                pilot = F.pad(input=pilot,
                              pad=(0, 0, 0, self.action_dim - pilot.shape[-2]),
                              mode='constant',
                              value=0)
                if len(state) != 1:
                    state[0] = pilot
                else:
                    state = pilot
                # Pad variable states
                # Get the length explicitly for later packing sequences
                lens = list(map(len, state))
                # Pad and pack
                padded = pad_sequence(state, batch_first=True)
                state = pack_padded_sequence(padded,
                                             lengths=lens,
                                             batch_first=True,
                                             enforce_sorted=False)
            self.l1.flatten_parameters()
            a, h = self.l1(state, hidden)
        else:
            a, h = F.relu(self.l1(state)), None

        # mini-batch update
        if Config().algorithm.recurrent_actor and hasattr(
                Config().clients,
                'varied') and Config().clients.varied and len(state) != 1:
            a, _ = pad_packed_sequence(a, batch_first=True)

        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        return self.max_action * a, h


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Critic, self).__init__()
        self.action_dim = action_dim

        if Config().algorithm.recurrent_actor:
            self.l1 = nn.LSTM(state_dim + 1, hidden_size, batch_first=True)
            self.l4 = nn.LSTM(state_dim + 1, hidden_size, batch_first=True)

        else:
            self.l1 = nn.Linear(state_dim + action_dim, hidden_size)
            self.l4 = nn.Linear(state_dim + action_dim, hidden_size)

        # Q1 architecture
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

        # Q2 architecture
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l6 = nn.Linear(hidden_size, 1)

    def forward(self, state, action, hidden1, hidden2):
        if Config().algorithm.recurrent_actor and hasattr(
                Config().clients, 'varied') and Config().clients.varied:
            # Pad the first state to full dims
            if len(state) != 1:
                pilot = state[0]
            else:
                pilot = state
            pilot = F.pad(input=pilot,
                          pad=(0, 0, 0, self.action_dim - pilot.shape[-2]),
                          mode='constant',
                          value=0)
            if len(state) != 1:
                state[0] = pilot
            else:
                state = pilot
            # Pad variable states
            # Get the length explicitly for later packing sequences
            lens = list(map(len, state))
            # Pad and pack
            padded = pad_sequence(state, batch_first=True)
            state = padded
        sa = torch.cat([state, action], -1)

        if Config().algorithm.recurrent_actor:
            if hasattr(Config().clients, 'varied') and Config().clients.varied:
                sa = pack_padded_sequence(sa,
                                          lengths=lens,
                                          batch_first=True,
                                          enforce_sorted=False)
            self.l1.flatten_parameters()
            self.l4.flatten_parameters()
            q1, hidden1 = self.l1(sa, hidden1)
            q2, hidden2 = self.l4(sa, hidden2)
        else:
            q1, hidden1 = F.relu(self.l1(sa)), None
            q2, hidden2 = F.relu(self.l4(sa)), None

        if Config().algorithm.recurrent_actor and hasattr(
                Config().clients, 'varied') and Config().clients.varied:
            q1, _ = pad_packed_sequence(q1, batch_first=True)
            q2, _ = pad_packed_sequence(q2, batch_first=True)

        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q1 = torch.mean(q1.reshape(q1.shape[0], -1, 1), 1)

        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        q2 = torch.mean(q2.reshape(q2.shape[0], -1, 1), 1)

        return q1, q2

    def Q1(self, state, action, hidden1):
        if Config().algorithm.recurrent_actor and hasattr(
                Config().clients, 'varied') and Config().clients.varied:
            # Pad variable states
            # Get the length explicitly for later packing sequences
            lens = list(map(len, state))
            # Pad and pack
            padded = pad_sequence(state, batch_first=True)
            state = padded

        sa = torch.cat([state, action], -1)
        if Config().algorithm.recurrent_actor:
            if hasattr(Config().clients, 'varied') and Config().clients.varied:
                sa = pack_padded_sequence(sa,
                                          lengths=lens,
                                          batch_first=True,
                                          enforce_sorted=False)
            self.l1.flatten_parameters()
            q1, hidden1 = self.l1(sa, hidden1)
        else:
            q1, hidden1 = F.relu(self.l1(sa)), None

        if Config().algorithm.recurrent_actor and hasattr(
                Config().clients, 'varied') and Config().clients.varied:
            q1, _ = pad_packed_sequence(q1, batch_first=True)

        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q1 = torch.mean(q1.reshape(q1.shape[0], -1, 1), 1)

        return q1


class Policy(object):
    def __init__(self, state_dim, action_space):
        self.max_action = Config().algorithm.max_action
        self.hidden_size = Config().algorithm.hidden_size
        self.discount = Config().algorithm.gamma
        self.tau = Config().algorithm.tau
        self.policy_noise = Config().algorithm.policy_noise * self.max_action
        self.noise_clip = Config().algorithm.noise_clip * self.max_action
        self.policy_freq = Config().algorithm.policy_freq
        self.lr = Config().algorithm.learning_rate

        self.device = Config().device
        self.on_policy = False

        self.actor = Actor(state_dim, action_space.shape[0], self.hidden_size,
                           self.max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr)

        self.critic = Critic(state_dim, action_space.shape[0],
                             self.hidden_size)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.lr)

        self.total_it = 0
        self.replay_buffer = ReplayMemory(state_dim, action_space.shape[0],
                                          self.hidden_size,
                                          Config().algorithm.replay_size,
                                          Config().algorithm.seed)

    def get_initial_states(self):
        h_0, c_0 = None, None
        if Config().algorithm.recurrent_actor:
            h_0 = torch.zeros(
                (self.actor.l1.num_layers, 1, self.actor.l1.hidden_size),
                dtype=torch.float)
            # h_0 = h_0.to(self.device)

            c_0 = torch.zeros(
                (self.actor.l1.num_layers, 1, self.actor.l1.hidden_size),
                dtype=torch.float)
            # c_0 = c_0.to(self.device)
        return (h_0, c_0)

    def select_action(self, state, hidden=None, test=True):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        action, hidden = self.actor(state, hidden)
        return action.cpu().data.numpy().flatten(), hidden

    def update(self):
        self.total_it += 1

        # Sample replay buffer
        if Config().algorithm.recurrent_actor:
            state, action, reward, next_state, done, h, c, nh, nc = self.replay_buffer.sample(
            )
            if hasattr(Config().clients, 'varied') and Config().clients.varied:
                # Pad variable actions
                padded = pad_sequence(action, batch_first=True)
                action = padded
            reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
            done = torch.FloatTensor(done).to(self.device).unsqueeze(1)
            hidden = (h, c)
            next_hidden = (nh, nc)
        else:
            state, action, reward, next_state, done = self.replay_buffer.sample(
            )
            state = torch.FloatTensor(state).to(self.device).unsqueeze(1)
            next_state = torch.FloatTensor(next_state).to(
                self.device).unsqueeze(1)
            action = torch.FloatTensor(action).to(self.device).unsqueeze(1)
            reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
            done = torch.FloatTensor(done).to(self.device).unsqueeze(1)
            hidden, next_hidden = (None, None), (None, None)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state, next_hidden)[0] +
                           noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action,
                                                      next_hidden, next_hidden)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action, hidden, hidden)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = critic_loss

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state,
                                         self.actor(state, hidden)[0],
                                         hidden).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data +
                                        (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data +
                                        (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def train_mode(self):
        self.actor.train()
        self.critic.train()

    def save_model(self, ep=None):
        """Saving the model to a file."""
        model_name = Config().algorithm.model_name
        model_path = f'./models/{model_name}/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if ep is not None:
            model_path += 'iter' + str(ep) + '_'

        torch.save(self.actor.state_dict(), model_path + 'actor.pth')
        torch.save(self.actor_optimizer.state_dict(),
                   model_path + "actor_optimizer.pth")
        torch.save(self.critic.state_dict(), model_path + 'critic.pth')
        torch.save(self.critic_optimizer.state_dict(),
                   model_path + "critic_optimizer.pth")

        logging.info("[RL Agent] Model saved to %s.", model_path)

    def load_model(self, ep=None):
        """Loading pre-trained model weights from a file."""
        model_name = Config().algorithm.model_name
        model_path = f'./models/{model_name}/'
        if ep is not None:
            model_path += 'iter' + str(ep) + '_'

        logging.info("[RL Agent] Loading a model from %s.", model_path)

        self.actor.load_state_dict(torch.load(model_path + 'actor.pth'))
        self.actor_optimizer.load_state_dict(
            torch.load(model_path + 'actor_optimizer.pth'))
        self.critic.load_state_dict(torch.load(model_path + 'critic.pth'))
        self.critic_optimizer.load_state_dict(
            torch.load(model_path + 'critic_optimizer.pth'))
