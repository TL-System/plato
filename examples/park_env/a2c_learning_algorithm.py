"""
The federated averaging algorithm for Actor-Critic model.
"""
from collections import OrderedDict

from plato.algorithms import fedavg
from plato.trainers.base import Trainer
import torch
from plato.config import Config
import numpy as np
import random


class Algorithm(fedavg.Algorithm):
    """ Federated averaging algorithm for Actor-Critic models, used by both the client and the server. """

    def __init__(self, trainer: Trainer):
        super().__init__(trainer)
        self.actor = self.model.actor
        self.critic = self.model.critic

    def compute_weight_deltas(self, weights_received):
        """ Extract the weights received from a client and compute the updates. """

        baseline_weights_actor, baseline_weights_critic = self.extract_weights()

        deltas = []
        
        for weight_actor, weight_critic in weights_received:
            delta_actor = OrderedDict()
            for name, current_weight in weight_actor.items():
                baseline = baseline_weights_actor[name]

                delta = current_weight - baseline
                delta_actor[name] = delta

            delta_critic = OrderedDict()
            for name, current_weight in weight_critic.items():
                baseline = baseline_weights_critic[name]

                delta = current_weight - baseline
                delta_critic[name] = delta

            deltas.append((delta_actor, delta_critic))

        return deltas

    def update_weights(self, deltas):
        """ Update the existing model weights. """
 
        baseline_weights_actor, baseline_weights_critic = self.extract_weights()
        update_actor, update_critic = deltas

        updated_weights_actor = OrderedDict()
        for name, weight in baseline_weights_actor.items():
            updated_weights_actor[name] = weight + update_actor[name]

        updated_weights_critic = OrderedDict()
        for name, weight in baseline_weights_critic.items():
            updated_weights_critic[name] = weight + update_critic[name]

        return updated_weights_actor, updated_weights_critic
    
    def extract_weights(self, model=None):
        """ Extract weights from the model. """
      
        actor = self.actor
        critic = self.critic
        if model is not None:
            actor = model.actor
            critic = model.critic

        actor_weight = actor.cpu().state_dict()
        critic_weight = critic.cpu().state_dict()

        return actor_weight, critic_weight

    def load_weights(self, weights):
        """ Load the model weights passed in as a parameter. """
        weights_actor, weights_critic = weights
        # The client might only receive one or none of the Actor
        # and Critic model weight.
        if weights_actor is not None:
            self.actor.load_state_dict(weights_actor, strict=True)
        if weights_critic is not None:
            self.critic.load_state_dict(weights_critic, strict=True)
