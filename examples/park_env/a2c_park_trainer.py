import numpy as np
import torch
import gym
from torch import nn
import matplotlib.pyplot as plt
import park

from plato.utils.reinforcement_learning.policies import base
from plato.trainers import basic
from plato.config import Config
from plato.trainers import basic

import os
import logging

# Memory
# Stores results from the networks, instead of calculating the operations again from states, etc.
class Memory():
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()  
    
    def _zip(self):
        return zip(self.log_probs,
                self.values,
                self.rewards,
                self.dones)
    
    def __iter__(self):
        for data in self._zip():
            return data
    
    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data
    
    def __len__(self):
        return len(self.rewards)

    #def save_buffer(self, dir, client_id):
        #size_np = np.array([self.size])
        
        #np.savez(self.make_filename(dir, "memory" + str(client_id)), a=self.state, b=self.action, c=self.reward, d=self.next_state, e=self.done, f = size_np)

    #def load_buffer(self, dir, client_id):
     #   data = np.load(self.make_filename(dir, "memory" + str(client_id)))

      #  self.state = data['a']
       # self.action = data['b']
        #self.reward = data['c']
        #self.next_state = data['d']
        #self.done = data['e']
        #self.size = int((data['f'])[0]) # single element array

    def make_filename(self, dir, name):
        file_name = "%s.npz" % (name) 
        file_path = os.path.join(dir, file_name)
        return file_path


class Trainer(basic.Trainer):
    def __init__(self, model=None):
        super().__init__()
        #pass

        #TODO INITIALIZE NECESSARY THINGS!!
        self.env = 2 #make it park env soon

        self.env_name = model.get_env_name()
        self.algorithm_name = model.get_rl_algo()



        self.model = model
        self.actor = model.actor
        self.critic = model.critic
        self.adam_actor = torch.optim.Adam(self.actor.parameters(), lr=Config().algorithm.learning_rate)
        self.adam_critic = torch.optim.Adam(self.critic.parameters(), lr=Config().algorithm.learning_rate)

        self.memory = Memory()

        self.episode_reward = []
        self.server_reward = []
        self.episode_num = 0
        self.total_reward = 0
        self.done = True
        self.trace_idx = 0
        self.steps = 0

        self.actor_state_dict = None
        self.critic_state_dict = None

        

              
        if not os.path.exists(Config().results.results_dir):
            os.makedirs(Config().results.results_dir)


    def t(self, x): 
        return torch.from_numpy(x).float()


    def train_model(self, config, trainset, sampler, cut_layer):
        """Main Training"""
        #We will put what exectues in the "main function of a2c_abr_sim.py here"
        while True:
            self.done = False
            self.total_reward = 0
            self.trace_idx = 0
            if (self.episode_num % 700 == 0):
                self.trace_idx = int(self.episode_num / 700)
                print( "change trace to: ", self.trace_idx )
            state = self.env.reset(trace_idx=self.trace_idx)
            self.steps = 0

            while not self.done:
                probs = self.actor(self.t(state))
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample()

                next_state, reward, self.done, info = self.env.step(action.detach().data.numpy())

                self.total_reward += reward
                self.steps += 1
                self.memory.add(dist.log_prob(action), self.critic(self.t(state)), reward, self.done)

                state = next_state

                if self.done or (self.steps % Config().algorithm.max_steps == 0):
                    last_q_val = self.critic(self.t(next_state)).detach().data.numpy()
                    self.train_helper(self.memory, last_q_val)
                    self.memory.clear()

            self.episode_num += 1
            self.episode_reward.append(self.total_reward)
            print("Episode number: %d, Reward: %d" % (self.episode_num, self.total_reward))

    def train_helper(self, memory, last_q_val):
        #We will put the train loop here
        values = torch.stack(memory.values)
        q_vals = np.zeros((len(memory), 1))

        #target values calculated backward
        #important to handle correctly done states
        #for those cases we want our target to be equal to the reward only
        for i, (_, _, reward, done) in enumerate(memory.reversed()):
            q_val = reward + Config().algorithm.gamma*q_val*(1.0-done)
            q_vals[len(memory)-1 - i] = q_val #store values from end to the start
        
        #advantage function!!
        advantage = torch.Tensor(q_vals) - values

        critic_loss = advantage.pow(2).mean()
        self.adam_critic.zero_grad()
        critic_loss.backward()
        self.adam_critic.step()

        actor_loss = (-torch.stack(memory.log_probs)*advantage.detach()).mean()
        self.adam_actor.zero_grad()
        actor_loss.backward()
        self.adam_actor.step()



                
    def load_model(self, filename=None, location=None):
        """Loading pre-trained model weights from a file."""
        #We will load actor and critic models here
        model_path = Config(
        ).params['model_path'] if location is None else location
        actor_model_name = 'actor_model'
        critic_model_name = 'critic_model'
        env_algorithm = self.env_name+ self.algorithm_name

        if filename is None:
            actor_filename = filename + '_actor'
            actor_model_path = f'{model_path}/{actor_filename}'
            critic_filename = filename + '_critic'
            critic_model_path = f'{model_path}/{critic_filename}'
        else:
            actor_model_path = f'{model_path}/{env_algorithm+actor_model_name}.pth'
            critic_model_path = f'{model_path}/{env_algorithm+critic_model_name}.pth'
    
        if self.client_id == 0:
            logging.info("[Server #%d] Loading models from %s, and %s.", os.getpid(),
                         actor_model_path, critic_model_path)
        else:
            logging.info("[Client #%d] Loading a model from %s, %s, %s and %s.",
                         self.client_id, actor_model_path, critic_model_path)

        if self.client_id != 0:
            #do we need to load memory here TODO
            #self.memory.

            file_name = "%s_%s.npz" % ("training_status", str(self.client_id)) 
            file_path = os.path.join(model_path, file_name)
            data = np.load(file_path)
            self.episode_num = int((data['a'])[0])
            #self.steps = int((data['b'])[0])

            self.actor.load_state_dict(torch.load(actor_model_path), strict=True)
            self.critic.load_state_dict(torch.load(critic_model_path), strict=True)

            #load episode_reward so it doesn't overwrite
            arr = np.load("%s.npz" %(Config().results.results_dir +"/"+Config().results.file_name+"_"+str(self.client_id)))
            self.episode_reward = list(arr['a'])


            #unsure if we need tehse
            self.adam_actor = torch.optim.Adam(self.actor.parameters(), lr=Config().algorithm.learning_rate)
            self.adam_critic = torch.optim.Adam(self.critic.parameters(), lr=Config().algorithm.learning_rate)



    def save_model(self, filename=None, location=None):
        """Saving the model to a file."""
        #We will save actor and critic models here
        """Saving the model to a file."""
        model_path = Config(
        ).params['model_path'] if location is None else location
        actor_model_name = 'actor_model'
        critic_model_name = 'critic_model'
        env_algorithm = self.env_name+ self.algorithm_name

        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except FileExistsError:
            pass

        if filename is None:
           # model_path = f'{model_path}/{filename}'
           # model_filename = filename + _'model'
           # model path = Config().params stuff
           actor_filename = filename + '_actor'
           critic_filename = filename + '_critic'
           actor_model_path = f'{model_path}/{actor_filename}'
           critic_model_path = f'{model_path}/{critic_filename}'
        else:
            actor_model_path = f'{model_path}/{env_algorithm+actor_model_name}.pth'
            critic_model_path = f'{model_path}/{env_algorithm+critic_model_name}.pth'

        if self.model_state_dict is None:
            torch.save(self.actor.state_dict(), actor_model_path)
            torch.save(self.critic.state_dict(),critic_model_path)
        else:
            torch.save(self.actor_state_dict, actor_model_path)
            torch.save(self.critic_state_dict, critic_model_path)

        if self.client_id != 0:
            # Need to save buffer and some variables
            #TODO do we need to save memory?
            #TODO do we need to save steps?
            file_name = "%s_%s.npz" % ("training_status", str(self.client_id)) 
            file_path = os.path.join(model_path, file_name)
            np.savez(file_path, a=np.array([self.episode_num]))#, b=np.array([self.steps]))        

        if self.client_id == 0:
            logging.info("[Server #%d] Saving models to %s, and %s.", os.getpid(),
                         actor_model_path, critic_model_path)
        else:
            logging.info("[Client #%d] Saving a model to %s, and %s.",
                         self.client_id, actor_model_path, critic_model_path)


    async def server_test(self, testset, sampler=None, **kwargs):
        #We will return the average reward here
        self.server_reward.append(self.total_reward)
        file_name = "A2C_RL_SERVER"
        np.savetxt("%s.csv" %(Config().results.results_dir +"/"+file_name), self.server_reward, delimiter=",")
        return self.total_reward
