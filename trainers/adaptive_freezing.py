"""
The federated learning trainer for Adaptive Parameter Freezing.

Reference:

C. Chen, et al. "Communication-Efficient Federated Learning with Adaptive
Parameter Freezing," found in docs/papers.
"""

import copy
import logging
import torch

from models.base import Model
from trainers import trainer
from config import Config


class Trainer(trainer.Trainer):
    """The federated learning trainer for Adaptive Parameter Freezing,
       used by both the client and the server.
    """
    def __init__(self, model: Model):
        super().__init__(model)
        self.sync_mask = {}
        self.moving_average_deltas = {}
        self.moving_average_abs_deltas = {}
        self.frozen_durations = {}
        self.wake_up_round = {}
        self.current_round = 0
        self.stability_threshold = Config().trainer.stability_threshold

        # Initialize the synchronization mask
        if not self.sync_mask:
            for name, weight in model.to(
                    torch.device('cpu')).named_parameters():
                if weight.requires_grad:
                    self.sync_mask[name] = torch.ones(weight.data.shape).bool()

        # Initialize the preserved weights
        self.previous_weights = None

    def compress_weights(self):
        """Extract weights from the model, and apply the sync mask
           to make sure that frozen parameters will not be transmitted.
        """
        weights = []
        for name, weight in self.model.to(
                torch.device('cpu')).named_parameters():
            if weight.requires_grad:
                # Rolling back model parameters that should be frozen
                weight.data = torch.where(self.sync_mask[name], weight.data,
                                          self.previous_weights[name].data)
                # Removing model weights that should not be synced with ther server
                weights_to_sync = torch.masked_select(weight.data,
                                                      self.sync_mask[name])
                weights.append((name, weights_to_sync))

        return weights

    def compute_weight_updates(self, weights_received):
        """Extract the weights received from a client and compute the updates."""
        # Extract baseline model weights
        baseline_weights = self.extract_weights()

        # Calculate updates from the received weights
        updates = []
        for weight in weights_received:
            update = []
            for i, (name, current_weight) in enumerate(weight):
                bl_name, baseline = baseline_weights[i]

                # Ensure correct weight is being updated
                assert name == bl_name

                # Expand the received weights using the sync mask
                updated_weight = copy.deepcopy(baseline)
                updated_weight[self.sync_mask[name]] = current_weight

                # Calculate update
                delta = updated_weight - baseline
                update.append((name, delta))
            updates.append(update)

        return updates

    def preserve_weights(self):
        """Making a copy of the model weights for later use."""
        self.previous_weights = {
            name: copy.deepcopy(weight)
            for name, weight in self.model.named_parameters()
        }

    def moving_average(self, previous_value, new_value):
        """Compute the exponential moving average."""
        alpha = Config().trainer.moving_average_alpha
        return previous_value * alpha + new_value * (1 - alpha)

    def update_sync_mask(self, name, weights):
        """Update the synchronization mask.

        Arguments:

        name: The name of the model layer (conv1.bias, etc.)
        weights: The tensor containing all the weights in the layer.
        """
        deltas = self.previous_weights[name] - weights
        self.sync_mask[name] = torch.ones(weights.shape).bool()
        indices = self.sync_mask[name].nonzero(as_tuple=True)

        if not name in self.moving_average_deltas:
            self.moving_average_deltas[name] = torch.zeros(weights.shape)
            self.moving_average_abs_deltas[name] = torch.zeros(weights.shape)
            self.frozen_durations[name] = torch.zeros(weights.shape).int()
            self.wake_up_round[name] = torch.zeros(weights.shape).int()

        self.moving_average_deltas[name][indices] = self.moving_average(
            self.moving_average_deltas[name][indices], deltas[indices])

        self.moving_average_abs_deltas[name][indices] = self.moving_average(
            self.moving_average_abs_deltas[name][indices],
            torch.abs(deltas[indices]))

        effective_perturbation = torch.abs(
            self.moving_average_deltas[name]
            [indices]) / self.moving_average_abs_deltas[name][indices]

        # Additive increase, multiplicative decrease for the frozen durations
        self.frozen_durations[name][indices] = torch.where(
            effective_perturbation < self.stability_threshold,
            self.frozen_durations[name][indices] + 1,
            self.frozen_durations[name][indices] // 2)

        self.wake_up_round[name][
            indices] = self.current_round + self.frozen_durations[name][
                indices] + 1

        if Config().trainer.random_freezing:
            rand = torch.rand(self.wake_up_round[name][indices].shape) * 100
            rand_frozen = torch.where(rand < self.current_round / 20.0,
                                      rand.int(),
                                      torch.zeros(rand.shape).int())

            self.wake_up_round[name][
                indices] = self.wake_up_round[name][indices] + rand_frozen

        # Updating the synchronization mask
        self.sync_mask[name] = (self.wake_up_round[name] >= self.current_round)

    def update_stability_threshold(self, inactive_ratio):
        """Tune the stability threshold adaptively if necessary."""
        logging.info('current ratio of stable parameters: {:.2f}'.format(
            inactive_ratio))

        if inactive_ratio > Config().trainer.tight_threshold:
            self.stability_threshold /= 2.0

    def load_weights(self, weights):
        """Loading the server model onto this client."""

        # Masking the weights received and load them into the model
        weights_received = []

        total_active_weights = 0
        total_weights = 0

        for name, weight in weights:
            # Expanding the compressed weights using the sync mask
            weight.data[self.sync_mask[name]] = weight.data.view(-1)

            if self.previous_weights is not None:
                self.update_sync_mask(name, weight.data)

            total_active_weights += self.sync_mask[name].sum()
            total_weights += torch.numel(weight)

            weights_received.append((name, weight.data))

        # Preserve the model weights for the next round
        self.preserve_weights()

        # Update the stability threshold, if necessary
        inactive_ratio = 1 - total_active_weights / total_weights
        self.update_stability_threshold(inactive_ratio)

        self.current_round += 1

        updated_state_dict = {}
        for name, weight in weights_received:
            updated_state_dict[name] = weight

        self.model.load_state_dict(updated_state_dict, strict=False)
