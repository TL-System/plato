import copy
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from plato.utils.rlfl.config import TD3Config as Config
from plato.utils.rlfl.policies.memory import ReplayMemory
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)


class Actor(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim,
                 max_action,
                 is_recurrent=False,
                 varied_per_round=False):
        super(Actor, self).__init__()
        self.recurrent = is_recurrent
        self.varied_per_round = varied_per_round
        self.action_dim = action_dim

        if self.recurrent:
            self.l1 = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        else:
            self.l1 = nn.Linear(state_dim, hidden_dim)

        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        if self.recurrent:
            self.l3 = nn.Linear(hidden_dim, 1)
        else:
            self.l3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

    def forward(self, state, hidden):
        if self.recurrent:
            # if self.varied_per_round and len(state) != 1:
            if self.varied_per_round:
                # # Pad the first state to full dims
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
        if self.recurrent and self.varied_per_round and len(state) != 1:
            a, _ = pad_packed_sequence(a, batch_first=True)

        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        return self.max_action * a, h


class Critic(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim,
                 is_recurrent=False,
                 varied_per_round=False):
        super(Critic, self).__init__()
        self.recurrent = is_recurrent
        self.varied_per_round = varied_per_round
        self.action_dim = action_dim

        if self.recurrent:
            self.l1 = nn.LSTM(state_dim + 1, hidden_dim, batch_first=True)
            self.l4 = nn.LSTM(state_dim + 1, hidden_dim, batch_first=True)

        else:
            self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
            self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)

        # Q1 architecture
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action, hidden1, hidden2):
        if self.recurrent and self.varied_per_round:
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

        if self.recurrent:
            if self.varied_per_round:
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

        if self.recurrent and self.varied_per_round:
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
        if self.recurrent and self.varied_per_round:
            # Pad variable states
            # Get the length explicitly for later packing sequences
            lens = list(map(len, state))
            # Pad and pack
            padded = pad_sequence(state, batch_first=True)
            state = padded

        sa = torch.cat([state, action], -1)
        if self.recurrent:
            if self.varied_per_round:
                sa = pack_padded_sequence(sa,
                                          lengths=lens,
                                          batch_first=True,
                                          enforce_sorted=False)
            self.l1.flatten_parameters()
            q1, hidden1 = self.l1(sa, hidden1)
        else:
            q1, hidden1 = F.relu(self.l1(sa)), None

        if self.recurrent and self.varied_per_round:
            q1, _ = pad_packed_sequence(q1, batch_first=True)

        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q1 = torch.mean(q1.reshape(q1.shape[0], -1, 1), 1)

        return q1


class Policy(object):
    def __init__(self, state_dim, action_space):
        self.max_action = Config().max_action
        self.hidden_dim = Config().hidden_size
        self.discount = Config().gamma
        self.tau = Config().tau
        self.policy_noise = Config().policy_noise
        self.noise_clip = Config().noise_clip
        self.policy_freq = Config().policy_freq
        self.lr = Config().learning_rate

        self.device = Config().device
        self.on_policy = False
        self.recurrent = Config().recurrent_actor
        self.varied_per_round = Config().varied_per_round
        # self.action_len = action_space.shape[0]

        self.actor = Actor(state_dim,
                           action_space.shape[0],
                           self.hidden_dim,
                           self.max_action,
                           is_recurrent=Config().recurrent_actor,
                           varied_per_round=Config().varied_per_round).to(
                               self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr)

        self.critic = Critic(state_dim,
                             action_space.shape[0],
                             self.hidden_dim,
                             is_recurrent=Config().recurrent_critic,
                             varied_per_round=Config().varied_per_round).to(
                                 self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.lr)

        self.total_it = 0
        self.replay_buffer = ReplayMemory(state_dim, action_space.shape[0],
                                          self.hidden_dim,
                                          Config().replay_size,
                                          Config().seed, self.recurrent,
                                          self.varied_per_round)

    def get_initial_states(self):
        h_0, c_0 = None, None
        if self.actor.recurrent:
            h_0 = torch.zeros(
                (self.actor.l1.num_layers, 1, self.actor.l1.hidden_size),
                dtype=torch.float)
            h_0 = h_0.to(device=self.device)

            c_0 = torch.zeros(
                (self.actor.l1.num_layers, 1, self.actor.l1.hidden_size),
                dtype=torch.float)
            c_0 = c_0.to(device=self.device)
        return (h_0, c_0)

    def select_action(self, state, hidden=None, test=True):
        # if self.recurrent:
        #     state = np.array(torch.FloatTensor(state).to(self.device).unsqueeze(0))
        # else:
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        action, hidden = self.actor(state, hidden)
        return action.cpu().data.numpy().flatten(), hidden

    def update(self):
        self.total_it += 1

        # Sample replay buffer
        if self.recurrent:
            state, action, reward, next_state, done, h, c, nh, nc = self.replay_buffer.sample(
            )
            if self.varied_per_round:
                # Pad variable states
                # Get the length explicitly for later packing sequences
                # temp = []
                # for item in (state, action, next_state):
                #     lens = list(map(len, item))
                #     # Pad and pack
                #     padded = pad_sequence(item, batch_first=True)
                #     temp.append(
                #         pack_padded_sequence(padded,
                #                              lengths=lens,
                #                              batch_first=True,
                #                              enforce_sorted=False))
                # state, action, next_state = temp
                # lens = list(map(len, state))
                # padded = pad_sequence(state, batch_first=True)
                # state = pack_padded_sequence(padded,
                #                              lengths=lens,
                #                              batch_first=True)
                # lens = list(map(len, next_state))
                # padded = pad_sequence(next_state, batch_first=True)
                # next_state = pack_padded_sequence(padded,
                #                              lengths=lens,
                #                              batch_first=True)

                lens = list(map(len, action))
                padded = pad_sequence(action, batch_first=True)
                action = padded
            reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
            done = torch.FloatTensor(done).to(self.device).unsqueeze(1)
            # h = torch.tensor(h, requires_grad=True,
            #                  dtype=torch.float).to(self.device)[0]
            # c = torch.tensor(c, requires_grad=True,
            #                  dtype=torch.float).to(self.device)[0]
            # nh = torch.tensor(nh, requires_grad=True,
            #                   dtype=torch.float).to(self.device)[0]
            # nc = torch.tensor(nc, requires_grad=True,
            #                   dtype=torch.float).to(self.device)[0]
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
        model_name = Config().model_name
        model_dir = Config().model_dir

        model_path = f'{model_dir}/{model_name}/'
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
        model_name = Config().model_name
        model_dir = Config().model_dir

        model_path = f'{model_dir}/{model_name}/'
        if ep is not None:
            model_path += 'iter' + str(ep) + '_'

        logging.info("[RL Agent] Loading a model from %s.", model_path)

        self.actor.load_state_dict(torch.load(model_path + 'actor.pth'))
        self.actor_optimizer.load_state_dict(
            torch.load(model_path + 'actor_optimizer.pth'))
        self.critic.load_state_dict(torch.load(model_path + 'critic.pth'))
        self.critic_optimizer.load_state_dict(
            torch.load(model_path + 'critic_optimizer.pth'))
