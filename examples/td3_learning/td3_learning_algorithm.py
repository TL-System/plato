"""
The federated averaging algorithm for Actor-Critic model.
"""
from collections import OrderedDict

from plato.algorithms import fedavg
from plato.trainers.base import Trainer


class Algorithm(fedavg.Algorithm):
    """ Federated averaging algorithm for Actor-Critic models, used by both the client and the server. """

    def __init__(self, trainer: Trainer):
        super().__init__(trainer)
        self.actor = self.model.actor
        self.critic = self.model.critic
        self.actor_target = self.model.actor_target
        self.critic_target = self.model.critic_target
    
    def compute_weight_deltas(self, weights_received):
        """ Extract the weights received from a client and compute the updates. """
        print("we need to compuete weight deltas")
        baseline_weights_actor, baseline_weights_critic, baseline_weights_actor_target, baseline_weights_critic_target = self.extract_weights()

        deltas = []
        
        for weight_actor, weight_critic, weight_actor_target,weight_critic_target in weights_received:
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

            delta_actor_target = OrderedDict()
            for name, current_weight in weight_actor_target.items():
                baseline = baseline_weights_actor_target[name]

                delta = current_weight - baseline
                delta_actor_target[name] = delta

            delta_critic_target = OrderedDict()
            for name, current_weight in weight_critic_target.items():
                baseline = baseline_weights_critic_target[name]

                delta = current_weight - baseline
                delta_critic_target[name] = delta

            deltas.append((delta_actor, delta_critic, delta_actor_target, delta_critic_target))

        return deltas

    def update_weights(self, deltas):
        """ Update the existing model weights. """
        print("line 61 is being exectued in update weights")
        baseline_weights_actor, baseline_weights_critic, baseline_weights_actor_target, baseline_weights_critic_target = self.extract_weights()
        update_actor, update_critic, update_actor_target, update_critic_target = deltas

        updated_weights_actor = OrderedDict()
        for name, weight in baseline_weights_actor.items():
            updated_weights_actor[name] = weight + updated_weights_actor[name]
        
        updated_weights_actor_target = OrderedDict()
        for name, weight in baseline_weights_actor_target.items():
            updated_weights_actor_target[name] = weight + updated_weights_actor_target[name]

        updated_weights_critic = OrderedDict()
        for name, weight in baseline_weights_critic.items():
            updated_weights_critic[name] = weight + updated_weights_critic [name]

        updated_weights_critic_target = OrderedDict()
        for name, weight in baseline_weights_critic_target.items():
            updated_weights_critic_target[name] = weight + updated_weights_critic_target [name]

        return updated_weights_actor, updated_weights_critic, updated_weights_actor_target, updated_weights_critic_target
    
    def extract_weights(self, model=None):
        """ Extract weights from the model. """
        print("line 84 in algorithm is being exectued")
        actor = self.actor
        critic = self.critic
        actor_target = self.actor_target
        critic_target = self.critic_target
        if model is not None:
            actor = model.actor
            critic = model.critic
            actor_target = model.actor_target
            critic_target = model.critic_target

        actor_weight = actor.cpu().state_dict()
        critic_weight = critic.cpu().state_dict()
        actor_target_weight = actor_target.cpu().state_dict()
        critic_target_weight = critic_target.cpu().state_dict()

        return actor_weight, critic_weight, actor_target_weight, critic_target_weight

    def load_weights(self, weights):
        """ Load the model weights passed in as a parameter. """
        print("weights are being loaded in line 104")
        weights_actor, weights_critic, weights_actor_target, weights_critic_target = weights
        # The client might only receive one or none of the Actor
        # and Critic model weight.
        if weights_actor is not None:
            self.actor.load_state_dict(weights_actor, strict=True)
        if weights_critic is not None:
            self.critic.load_state_dict(weights_critic, strict=True)
        if weights_actor_target is not None:
            self.actor_target.load_state_dict(weights_actor_target, strict=True)
        if weights_critic_target is not None:
            self.critic_target.load_state_dict(weights_critic_target, strict=True)