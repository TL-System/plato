"""
A customized trainer for td3.
"""
import logging
import os
from pyexpat import model
import time
import pandas as pd
import copy

import numpy as np
import torch
import torch.nn.functional as F
import globals

from torch import nn
from plato.utils.reinforcement_learning.policies import base
from plato.trainers import basic
from plato.config import Config
from plato.trainers import basic

import td3_learning_client as client

import pickle

import random

#TODO: think again about global variables
file_name = "TD3_RL"
models_dir = "./pytorch_models" # TODO: models are not stored here
results_dir = "examples/td3_learning/results"


class ReplayMemory(base.ReplayMemory):
    """ A simple example of replay memory buffer. """
    def __init__(self, state_dim, action_dim, capacity, seed, client_id):
        super().__init__(state_dim, action_dim, capacity, seed)
        self.client_id = client_id

    def save_buffer(self, dir):
        size_np = np.array([self.size])
        
        np.savez(self.make_filename(dir, "replay_buffer"), a=self.state, b=self.action, c=self.reward, d=self.next_state, e=self.done, f = size_np)

    def load_buffer(self, dir):
        data = np.load(self.make_filename(dir, "replay_buffer"))

        self.state = data['a']
        self.action = data['b']
        self.reward = data['c']
        self.next_state = data['d']
        self.done = data['e']
        self.size = int((data['f'])[0]) # single element array

    def make_filename(self, dir, name):
        file_name = "%s_%s.npz" % (name, str(self.client_id)) 
        file_path = os.path.join(dir, file_name)
        return file_path

class Trainer(basic.Trainer):
    def __init__(self, model=None):
        #super().__init__(state_dim, action_dim, max_action, model)
        super().__init__()
        self.env = globals.env
        #Create actor and critic
        #Could have used the base class given's but for convenient sake we declare our own

        print("are we consturcting more than once??") #no it does'nt get claled more than once
        self.max_action = globals.max_action            
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
        self.replay_buffer = ReplayMemory(
            globals.state_dim, globals.action_dim, 
            Config().algorithm.max_replay_size, 
            Config().clients.random_seed, self.client_id)
        
        self.policy_noise = Config().algorithm.policy_noise 
        self.noise_clip = Config().algorithm.noise_clip

        self.evaluations = []
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        if Config().algorithm.save_models and not os.path.exists(models_dir):
            os.makedirs(models_dir)

        self.timesteps_since_eval = 0
        self.episode_num = 0
        self.total_timesteps = 0
        self.done = True

        self.episode_reward = 0
        
        self.actor_state_dict = None
        self.critic_state_dict = None
        self.actor_target_state_dict = None
        self.critic_target_state_dict = None

    def select_action(self, state):
        """Select action from policy"""
        state = torch.FloatTensor(state.reshape(1,-1)).to(self.device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    #add to replay buffer
    def add(self, transition):
            #adds to ReplayMemory, it always updates the pointer in the push method
            self.replay_buffer.push(transition)


    def train_model(self, config, trainset, sampler, cut_layer):
        """Main Training"""
        #print("in line 87 of td3_learning_trainer!")
        episode_timesteps = 0 #fixing error about using before assignment
        obs = 0 #fixing error about using before assignment
        round_episode_steps = 0
        if self.total_timesteps > Config().algorithm.max_steps:
            # TODO: when max number of steps is hit, we should stop training and terminate the process. How?
            print("Done training")
            return
        while round_episode_steps < Config().algorithm.max_round_episodes * globals.max_episode_steps:
            if self.done:
                #evaluate episode and save policy
                if self.timesteps_since_eval >= Config().algorithm.eval_freq * globals.max_episode_steps:
                    self.timesteps_since_eval %= Config().algorithm.eval_freq * globals.max_episode_steps
                    self.evaluations.append(client.evaluate_policy(self, self.env))
                    print(self.evaluations)
                    np.save("./results/%s" % (file_name), self.evaluations)
                
                #When the training step is done, we reset the state of the env
                obs = self.env.reset()

                #Set done to false
                self.done = False

                # Set rewards and episode timesteps to zero
                self.episode_reward = 0
                episode_timesteps = 0
                self.episode_num += 1

            #Before the number of specified timesteps from config file we sample random actions
            if self.total_timesteps < Config().algorithm.start_steps:
                action = self.env.action_space.sample()
            else: #after we pass the threshold we switch model
                action = self.select_action(np.array(obs))

                #if not 0 we add noise
                if Config().algorithm.expl_noise != 0:
                    expl_noise = Config().algorithm.expl_noise
                    action = (action + np.random.normal(0, expl_noise, size = self.env.action_space.shape[0])).clip(
                        self.env.action_space.low, self.env.action_space.high
                    )
            

            #performs action in environment, then reaches next state and receives the reward
            new_obs, reward, self.done, _ = self.env.step(action)
            #print(reward)

            #is episode done?
            done_bool = 0 if episode_timesteps + 1 == self.env._max_episode_steps else float(self.done)
            
            #update total reward
            self.episode_reward += reward
           
            #add to replay buffer in this order due to push method in replay buffer
            new = (obs, action, reward, new_obs, done_bool)
            self.add(new)

            #Update state, episode time_step, total timesteps, and timesteps since last eval
            obs = new_obs
            episode_timesteps += 1
            self.total_timesteps += 1
            round_episode_steps += 1
            self.timesteps_since_eval += 1

            #If episode is done
            if self.done:
                if self.total_timesteps != 0:
                    logging.info("Total Timesteps: {} Episode Num: {} Reward: {}".format(self.total_timesteps, self.episode_num, self.episode_reward))
                    #train here call td3_trainer
                    self.train_helper()
        
        #Add the last policy evaluation to our list of evaluations and save evaluations
        self.evaluations.append(client.evaluate_policy(self, self.env))
        np.save("./results/%s" % (file_name), self.evaluations)


    def train_helper(self):
        """Training Loop"""
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
            target_Q = reward + ((1-done) * Config().algorithm.gamma * target_Q).detach()

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
        # TODO: here load replay buffer

        self.replay_buffer.load_buffer(results_dir)
        model_path = Config(
        ).params['model_path'] if location is None else location
        actor_model_name = 'actor_model'
        critic_model_name = 'critic_model'
        actor_target_model_name = 'actor_target_model'
        critic_target_model_name = 'critic_target_model'
        env_algorithm = globals.env_name+ globals.algorithm_name

        #going in here for some reason
        #fixed by making it none
        if filename is None:
            actor_filename = filename + '_actor'
            actor_model_path = f'{model_path}/{actor_filename}'
            critic_filename = filename + '_critic'
            critic_model_path = f'{model_path}/{critic_filename}'
            actor_target_filename = filename + '_actor_target'
            actor_target_model_path = f'{model_path}/{actor_target_filename}'
            critic_target_filename = filename + '_critic_target'
            critic_target_model_path = f'{model_path}/{critic_target_filename}'
        else:
            actor_model_path = f'{model_path}/{env_algorithm+actor_model_name}.pth'
            critic_model_path = f'{model_path}/{env_algorithm+critic_model_name}.pth'
            actor_target_model_path = f'{model_path}/{env_algorithm+actor_target_model_name}.pth'
            critic_target_model_path = f'{model_path}/{env_algorithm+critic_target_model_name}.pth'

        if self.client_id == 0:
            logging.info("[Server #%d] Loading models from %s, %s, %s and %s.", os.getpid(),
                         actor_model_path, critic_model_path, actor_target_model_path, critic_target_model_path)
        else:
            logging.info("[Client #%d] Loading a model from %s, %s, %s and %s.",
                         self.client_id, actor_model_path, critic_model_path, actor_target_model_path, critic_target_model_path)

        # Load episode_num and total_timesteps
        file_name = "%s_%s.npz" % ("training_status", str(self.client_id)) 
        file_path = os.path.join(model_path, file_name)
        data = np.load(file_name)
        self.episode_num = int((data['a'])[0])
        self.total_timesteps = int((data['b'])[0])

        self.actor.load_state_dict(torch.load(actor_model_path), strict=True)

        self.critic.load_state_dict(torch.load(critic_model_path), strict=True)

        self.actor_target.load_state_dict(torch.load(actor_target_model_path), strict=True)

        self.critic_target.load_state_dict(torch.load(critic_target_model_path), strict=True)

        self.replay_buffer.load_buffer(model_path)


        # TODO: do we need those?
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr = Config().algorithm.learning_rate)
        
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr = Config().algorithm.learning_rate)

    def save_model(self, filename=None, location=None):
        """Saving the model to a file."""
        model_path = Config(
        ).params['model_path'] if location is None else location
        actor_model_name = 'actor_model'
        critic_model_name = 'critic_model'
        actor_target_model_name = 'actor_target_model'
        critic_target_model_name = 'critic_target_model'
        env_algorithm = globals.env_name+ globals.algorithm_name

        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except FileExistsError:
            pass
        
        #going in here for some reason
        #fixed by making it none
        if filename is None:
           # model_path = f'{model_path}/{filename}'
           # model_filename = filename + _'model'
           # model path = Config().params stuff
           actor_filename = filename + '_actor'
           critic_filename = filename + '_critic'
           actor_target_filename = filename + '_actor_target'
           critic_target_filename = filename + '_critic_target'
           actor_model_path = f'{model_path}/{actor_filename}'
           critic_model_path = f'{model_path}/{critic_filename}'
           actor_target_model_path = f'{model_path}/{actor_target_filename}'
           critic_target_model_path = f'{model_path}/{critic_target_filename}'
        else:
            actor_model_path = f'{model_path}/{env_algorithm+actor_model_name}.pth'
            critic_model_path = f'{model_path}/{env_algorithm+critic_model_name}.pth'
            actor_target_model_path = f'{model_path}/{env_algorithm+actor_target_model_name}.pth'
            critic_target_model_path = f'{model_path}/{env_algorithm+critic_target_model_name}.pth'

        if self.model_state_dict is None:
            #torch.save(self.model.state_dict(), model_path)
            torch.save(self.actor.state_dict(), actor_model_path)
            torch.save(self.critic.state_dict(),critic_model_path)
            torch.save(self.actor_target.state_dict(), actor_target_model_path)
            torch.save(self.critic_target.state_dict(), critic_target_model_path)

        else:
            #torch.save(self.model_state_dict, model_path)
            torch.save(self.actor_state_dict, actor_model_path)
            torch.save(self.critic_state_dict, critic_model_path)
            torch.save(self.actor_target_state_dict, actor_target_model_path)
            torch.save(self.critic_target_state_dict, critic_target_model_path)
        
        # Need to save buffer
        self.replay_buffer.save_buffer(model_path)

        # Need to save total_timesteps and episode_num that we stopped at (to resume training)
        file_name = "%s_%s.npz" % ("training_status", str(self.client_id)) 
        file_path = os.path.join(model_path, file_name)
        np.savez(file_name, a=np.array([self.episode_num]), b=np.array([self.total_timesteps]))


        #TODO What is the difference between .state_dict() & _state_dict

        if self.client_id == 0:
            logging.info("[Server #%d] Saving models to %s, %s, %s and %s.", os.getpid(),
                         actor_model_path, critic_model_path, actor_target_model_path, critic_target_model_path)
        else:
            logging.info("[Client #%d] Saving a model to %s, %s, %s and %s.",
                         self.client_id, actor_model_path, critic_model_path, actor_target_model_path, critic_target_model_path)

    async def server_test(self, testset, sampler=None, **kwargs):
        return (client.evaluate_policy(self, self.env))



