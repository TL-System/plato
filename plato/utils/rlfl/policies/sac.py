"""
Reference:

https://github.com/pranz24/pytorch-soft-actor-critic
"""
import logging
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from plato.utils import csv_processor
from rlfl.config import SACConfig as Config

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) +
                                param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, data):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        batch = random.sample(self.buffer, Config().batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample(
        )  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


class Policy(object):
    def __init__(self, state_dim, action_space):
        self.device = Config().device
        self.alpha = Config().alpha
        self.automatic_entropy_tuning = Config().automatic_entropy_tuning

        self.critic = QNetwork(state_dim, action_space.shape[0],
                               Config().hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(),
                                 lr=Config().learning_rate)

        self.critic_target = QNetwork(state_dim, action_space.shape[0],
                                      Config().hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if Config().policy == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(
                    torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1,
                                             requires_grad=True,
                                             device=self.device)
                self.alpha_optim = Adam([self.log_alpha],
                                        lr=Config().learning_rate)

            self.actor = GaussianPolicy(state_dim, action_space.shape[0],
                                        Config().hidden_size,
                                        action_space).to(self.device)
            self.actor_optim = Adam(self.actor.parameters(),
                                    lr=Config().learning_rate)

        elif Config().policy == "Deterministic":
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.actor = DeterministicPolicy(state_dim, action_space.shape[0],
                                             Config().hidden_size,
                                             action_space).to(self.device)
            self.actor_optim = Adam(self.actor.parameters(),
                                    lr=Config().learning_rate)

        self.replay_buffer = ReplayMemory(Config().replay_size, Config().seed)
        self.writer = SummaryWriter(Config().result_dir)

        self.num_update_iteration = 0

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)
        return action.detach().cpu().numpy().flatten()

    def update(self):
        for it in range(Config().update_iteration):
            # Sample a batch from memory
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.replay_buffer.sample(
            )

            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(
                self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(
                self.device).unsqueeze(1)
            mask_batch = torch.FloatTensor(mask_batch).to(
                self.device).unsqueeze(1)

            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.actor.sample(
                    next_state_batch)
                qf1_next_target, qf2_next_target = self.critic_target(
                    next_state_batch, next_state_action)
                min_qf_next_target = torch.min(
                    qf1_next_target,
                    qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + (1 - mask_batch) * Config().gamma * (
                    min_qf_next_target)
            qf1, qf2 = self.critic(
                state_batch, action_batch
            )  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = F.mse_loss(qf1, next_q_value)
            qf2_loss = F.mse_loss(qf2, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()

            pi, log_pi, _ = self.actor.sample(state_batch)

            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()

            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha *
                               (log_pi + self.target_entropy).detach()).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp()
                alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
            else:
                alpha_loss = torch.tensor(0.).to(self.device)
                alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

            soft_update(self.critic_target, self.critic, Config().tau)

            self.writer.add_scalar('loss/critic_1', qf1_loss.item(),
                                   self.num_update_iteration)
            self.writer.add_scalar('loss/critic_2', qf2_loss.item(),
                                   self.num_update_iteration)
            self.writer.add_scalar('loss/policy', policy_loss.item(),
                                   self.num_update_iteration)
            self.writer.add_scalar('loss/entropy_loss', alpha_loss.item(),
                                   self.num_update_iteration)
            self.writer.add_scalar('loss/alpha',
                                   alpha_tlogs.item(),
                                   self.num_update_iteration)

            self.num_update_iteration += 1

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(
        ), alpha_loss.item(), alpha_tlogs.item()

    def save_model(self, ep=None):
        """Saving the model to a file."""
        model_name = Config().model_name
        model_dir = Config().model_dir

        model_path = f'{model_dir}/{model_name}/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if ep is not None:
            model_path += 'iter' + str(ep) + '_'

        torch.save(self.actor.state_dict(), model_path + 'actor.pth')
        torch.save(self.critic.state_dict(), model_path + 'critic.pth')

        logging.info("[RL Agent] Model saved to %s.", model_path)

    def load_model(self, ep=None):
        """Loading pre-trained model weights from a file."""
        model_name = Config().model_name
        model_dir = Config().model_dir

        model_path = f'{model_dir}/{model_name}/'
        if ep is not None:
            model_path += 'iter' + str(ep) + '_'

        logging.info("[RL Agent] Loading a model from %s.", model_path)

        self.actor.load_state_dict(torch.load(model_path + 'actor.pth'))
        self.critic.load_state_dict(torch.load(model_path + 'critic.pth'))
