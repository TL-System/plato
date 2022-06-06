"""
A customized trainer for td3.
"""
import logging
import os
import time

import copy

import numpy as np
import torch
import torch.nn.functional as F
import globals

from torch import nn
from plato.utils.reinforcement_learning.policies import base
from opacus.privacy_engine import PrivacyEngine
from plato.config import Config
from plato.models import registry as models_registry
from plato.trainers import basic
from plato.utils import optimizers



class TD3Actor(base.Actor):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__(state_dim, action_dim, max_action)

    def forward(self, x, hidden=None):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        # Normalize/Scaling aggregation weights so that the sum is 1
        x += 1  # [-1, 1] -> [0, 2]
        x /= x.sum()
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

    def forward(self, state, action, hidden1=None, hidden2=None):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action, hidden=None):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class Trainer(base.Policy):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        #Create actor and critic
        #Could have used the base class given's but for convenient sake we declare our own
        self.actor = TD3Actor(state_dim, action_dim, self.max_action).to(self.device)
        self.critic = TD3Critic(state_dim, action_dim).to(self.device)

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr = Config().algorithm.learning_rate)

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr = Config().algorithm.learning_rate)

        #replay buffer initialization
        self.replay_buffer = base.ReplayMemory(
            state_dim, action_dim, 
            Config().algorithm.replay_size, 
            Config().algorithm.replay_seed)
        
        self.policy_noise = Config().algorithm.policy_noise * self.max_action
        self.noise_clip = Config().algorithm.noise_clip * self.max_action

    def select_action(self, state):
        """Select action from policy"""
        state = torch.FloatTensor(state.reshape(1,-1)).to(self.device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    #add to replay buffer
    def add(self, transition):
            #adds to ReplayMemory, it always updates the pointer in the push method
            self.replay_buffer.push(transition)


    def update(self):
        """Training"""

        for it in range(Config().algorithm.iterations):

            #sample from replay buffer
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = self.replay_buffer.sample()
            state = torch.FloatTensor(batch_states).to(self.device)
            action = torch.FloatTensor(batch_actions).to(self.device)
            reward = torch.FloatTensor(batch_rewards).to(self.device)
            next_state = torch.FloatTensor(batch_next_states).to(self.device)
            done = torch.FloatTensor(batch_dones).to(self.device)

            #with torch.no_grad():
            #from next state s' get next action a'
            next_action = self.actor_target(next_state)

            #add gaussian noise to this next action a' and clamp it in range of values supported by env
            noise = torch.FloatTensor(batch_actions).data.normal_(0, self.policy_noise).to(self.device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            #Two critics take the couple (s', a') as input and return two Q values Qt1(s',a') & Qt2 as outputs
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            #Keep minimum of the two Q values: min(Q1, Q2)
            target_Q = torch.min(target_Q1, target_Q2)

            #Get final target of the two critic models
            target_Q = reward + (1-done) * Config().algorithm.gamma * target_Q

            #Two critics take each couple (s,a) as input and return two Q-values
            current_Q1, current_Q2 = self.critic(state, action)

            #Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + \
                F.mse_loss(current_Q2, target_Q)

            #optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if it % Config().algorithm.policy_freq == 0:

                #Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                #optimize actor
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
                
                
    def evaluate_policy(self, policy, eval_episodes = 10):
        avg_reward = 0
        for _ in range(eval_episodes):
            obs = globals.env.reset()
            done = False
            while not done:
                action = policy.select_action(np.array(obs))
                obs, reward, done, _ = globals.env.step(action)
                avg_reward += reward
        avg_reward /= eval_episodes
        print ("---------------------------------------")
        print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
        print ("---------------------------------------")
        return avg_reward



