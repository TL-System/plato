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
from plato.trainers import basic
from opacus.privacy_engine import PrivacyEngine
from plato.config import Config
from plato.models import registry as models_registry
from plato.trainers import basic
from plato.utils import optimizers

class Trainer(basic.Trainer):
    def __init__(self, state_dim, action_dim, max_action=None, model=None):
        #super().__init__(state_dim, action_dim, max_action, model)
        super().__init__(model=model)
        self.env = globals.env
        #Create actor and critic
        #Could have used the base class given's but for convenient sake we declare our own
        self.max_action = max_action            
        self.model = model
        self.actor = self.model.actor
        self.critic = self.model.critic
        self.actor_target = self.model.actor_target
        self.critic_target = self.model.critic_target
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr = Config().algorithm.learning_rate)
        
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


    def train_loop(self, config, trainset, sampler, cut_layer):
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

        print("one client update done")    
                
            
    def load_model(self, filename=None, location=None):
        """Loading pre-trained model weights from a file."""
        model_path = Config(
        ).params['model_path'] if location is None else location
        model_name = Config().trainer.model_name

        if filename is not None:
            model_path = f'{model_path}/{filename}'
        else:
            model_path = f'{model_path}/{model_name}.pth'

        if self.client_id == 0:
            logging.info("[Server #%d] Loading a model from %s.", os.getpid(),
                         model_path)
        else:
            logging.info("[Client #%d] Loading a model from %s.",
                         self.client_id, model_path)


        #self.model.load_state_dict(torch.load(model_path), strict=True)
        self.actor.load_state_dict(torch.load(model_path), strict=True)

        self.critic.load_state_dict(torch.load(model_path), strict=True)

        self.actor_target.load_state_dict(torch.load(model_path), strict=True)

        self.critic_target.load_state_dict(torch.load(model_path), strict=True)




