"""
The federated learning trainer for Adaptive Parameter Freezing.

Reference:

C. Chen, et al. "Communication-Efficient Federated Learning with Adaptive
Parameter Freezing," found in docs/papers.
"""

import copy
import torch
import logging

from models.base import Model
from trainers import trainer
from config import Config


class Trainer(trainer.Trainer):
    """The federated learning trainer for Adaptive Synchronization Frequency,
       used by the server.
    """
    def __init__(self, model: Model):
        super().__init__(model)

        self.average_positive_deltas = {
            name: torch.zeros(param.data.shape)
            for name, param in model.named_parameters()
        }
        self.average_negative_deltas = {
            name: torch.zeros(param.data.shape)
            for name, param in model.named_parameters()
        }

    def compute_weight_updates(self, weights_received):
        """Extract the weights received from a client and compute the updates."""
        # Extract baseline model weights
        baseline_weights = self.extract_weights()

        self.update_sync_frequency(baseline_weights, weights_received)

        # Calculate updates from the received weights
        updates = []
        for weight in weights_received:
            update = []
            for i, (name, current_weight) in enumerate(weight):
                bl_name, baseline = baseline_weights[i]

                # Ensure correct weight is being updated
                assert name == bl_name

                # Calculate updates
                delta = current_weight - baseline
                update.append((name, delta))
            updates.append(update)

        return updates

    def moving_average(self, previous_value, new_value):
        """Compute the exponential moving average."""
        alpha = Config().trainer.moving_average_alpha
        return previous_value * alpha + new_value * (1 - alpha)

    def update_sync_frequency(self, previous_weights, weights_received):
        """Update the sychronization frequency between the clients and
        the server.
        """
        model_size = 0
        consistency_count = 0
        for previous_name, previous_weight in previous_weights:
            model_size += previous_weight.numel()
            positive_deltas = torch.zeros(previous_weight.shape)
            negative_deltas = torch.zeros(previous_weight.shape)

            for weights in weights_received:
                for name, current_weight in weights:
                    if previous_name == name:
                        delta = current_weight - previous_weight

                        # Calculate both positive and negative updates
                        positive_deltas += torch.where(
                            delta > 0, delta, torch.zeros(delta.shape))

                        negative_deltas += torch.where(
                            delta < 0, delta, torch.zeros(delta.shape))

                        self.average_positive_deltas[
                            name] = self.moving_average(
                                self.average_positive_deltas[name],
                                positive_deltas)

                        self.average_negative_deltas[
                            name] = self.moving_average(
                                self.average_negative_deltas[name],
                                negative_deltas)
            divide_base = self.average_positive_deltas[
                previous_name] - self.average_negative_deltas[previous_name]
            divide_base = torch.where(divide_base == 0,
                                      torch.ones(previous_weight.shape),
                                      divide_base)
            layer_consistency_rate = torch.abs(
                self.average_positive_deltas[previous_name] +
                self.average_negative_deltas[previous_name]) / divide_base
            consistency_count += torch.sum(layer_consistency_rate)
        consistency_rate = consistency_count / model_size
        print(f"Consistent_rate: {consistency_rate}")

    def load_weights(self, weights):
        """Loading the server model onto this client."""
        updated_state_dict = {}
        for name, weight in weights:
            updated_state_dict[name] = weight

        self.model.load_state_dict(updated_state_dict, strict=False)
