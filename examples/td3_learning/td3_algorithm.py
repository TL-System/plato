
"""Federated averaging algorithm for td3"""

from collections import OrderedDict

from plato.algorithms import fedavg
from plato.trainers.base import Trainer


class Algorithm(fedavg.Algorithm):

    def __init__(self, trainer: Trainer):
        super().__init__(trainer=trainer)
        self.actor = self.model.actor
        self.critic = self.model.critic

    def compute_weight_deltas(self, weights_received):
        """ Extract the weights received from a client and compute the updates. """
        actor_weights, critic_weights = self.extract_weights()

        deltas = []

        for weight_actor, weight_critic in weights_received:
            delta_actor = OrderedDict()
            for name, current_weight in weight_actor.items():
                baseline = actor_weights[name]

                delta = current_weight - baseline
                delta_actor[name] = delta
            
            delta_critic = OrderedDict()

            for name, current_weight in weight_critic.items():
                baseline = critic_weights[name]

                delta = current_weight - baseline
                delta_critic[name] = delta
            deltas.append((delta_actor, delta_critic))
        
        return deltas

    def update_weights(self, deltas):
        """Update the existing model weights"""
        actor_weights, critic_weights = self.extract_weights()
        
        update_actor, update_critic = deltas

        updated_weights_actor = OrderedDict()

        for name, weight in actor_weights.items():
            updated_weights_actor[name] = weight + update_actor[name]

        updated_weights_critic = OrderedDict()

        for name, weight in critic_weights.items():
            updated_weights_critic[name] = weight + update_critic[name]

        return updated_weights_actor, updated_weights_critic

    def extract_weights(self, model=None):
        """Extract weights from model"""
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

        if weights_actor is not None:
            self.actor.load_state_dict(weights_actor, strict=True)

        if weights_critic is not None:
            self.actor.load_state_dict(weights_critic, strict=True)

        

