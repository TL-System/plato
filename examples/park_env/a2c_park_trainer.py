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


class Trainer(basic.Trainer):
    def __init__(self, model=None):
        super().__init__()
        #pass

        #TODO INITIALIZE NECESSARY THINGS!!

    
    def add(self, log_prob, value, reward, done):
        #We will add to the memory here
        pass


    def train_model(self, config, trainset, sampler, cut_layer):
        """Main Training"""
        #We will put what exectues in the "main function of a2c_abr_sim.py here"
        pass

    def train_helper(self, memmory, last_q_val):
        #We will put the train loop here
        pass

                
    def load_model(self, filename=None, location=None):
        """Loading pre-trained model weights from a file."""
        #We will load actor and critic models here
        pass

    def save_model(self, filename=None, location=None):
        """Saving the model to a file."""
        #We will save actor and critic models here
        pass

    async def server_test(self, testset, sampler=None, **kwargs):
        #We will return the average reward here
        pass
