"""
The federated learning algorithm for Adaptive Parameter Freezing.

Reference:

C. Chen, H. Xu, W. Wang, B. Li, B. Li, L. Chen, G. Zhang. “Communication-
Efficient Federated Learning with Adaptive Parameter Freezing,” in the
Proceedings of the 41st IEEE International Conference on Distributed Computing
Systems (ICDCS 2021), Online, July 7-10, 2021 (found in papers/).
"""

import copy
import logging
from collections import OrderedDict

import torch
from plato.config import Config
from plato.trainers.base import Trainer

from plato.algorithms import fedavg


class Algorithm(fedavg.Algorithm):
    """The federated learning trainer for Adaptive Parameter Freezing,
       used by both the client and the server.
    """
    def __init__(self, trainer: Trainer = None):
        super().__init__(trainer)
        self.sync_mask = {}
        self.moving_average_deltas = {}
        self.moving_average_abs_deltas = {}
        self.frozen_durations = {}
        self.wake_up_round = {}
        self.current_round = 0
        self.stability_threshold = Config().algorithm.stability_threshold

        # Layers in ResNet models with the following suffixes in their names need to be skipped
        self.skipped_layers = [
            'running_var',
            'running_mean',
            'num_batches_tracked',
        ]

        # Initialize the synchronization mask
        if not self.sync_mask:
            for name, weight in self.model.cpu().state_dict().items():
                self.sync_mask[name] = torch.ones(weight.data.shape).bool()

        # Initialize the preserved weights
        self.previous_weights = None

    def compress_weights(self):
        """Extract weights from the model, and apply the sync mask
           to make sure that frozen parameters will not be transmitted.
        """
        weights = {}
        for name, weight in self.model.cpu().state_dict().items():
            # Rolling back model parameters that should be frozen
            weight.data = torch.where(self.sync_mask[name], weight.data,
                                      self.previous_weights[name].data)
            # Removing model weights that should not be synced with ther server
            weights_to_sync = torch.masked_select(weight.data,
                                                  self.sync_mask[name])
            weights[name] = weights_to_sync
        return weights

    def compute_weight_updates(self, weights_received):
        """Extract the weights received from a client and compute the updates."""
        # Extract baseline model weights
        baseline_weights = self.extract_weights()

        # Calculate updates from the received weights
        updates = []
        for weight in weights_received:
            update = OrderedDict()
            for name, current_weight in weight.items():
                baseline = baseline_weights[name]

                # Expand the received weights using the sync mask
                updated_weight = copy.deepcopy(baseline)
                updated_weight[self.sync_mask[name]] = current_weight

                # Calculate update
                delta = updated_weight - baseline
                update[name] = delta
            updates.append(update)

        return updates

    def preserve_weights(self):
        """Making a copy of the model weights for later use."""
        self.previous_weights = {
            name: copy.deepcopy(weight)
            for name, weight in self.model.cpu().state_dict().items()
        }

    def moving_average(self, previous_value, new_value):
        """Compute the exponential moving average."""
        alpha = Config().algorithm.moving_average_alpha
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

        if Config().algorithm.random_freezing:
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
        logging.info('Current ratio of stable parameters: {:.2f}'.format(
            inactive_ratio))

        if inactive_ratio > Config().algorithm.tight_threshold:
            self.stability_threshold /= 2.0

    def load_weights(self, weights):
        """Loading the server model onto this client."""

        # Masking the weights received and load them into the model
        weights_received = []

        total_active_weights = 0
        total_weights = 0

        for name, weight in weights.items():
            # Expanding the compressed weights using the sync mask
            weight.data[self.sync_mask[name]] = weight.data.clone().view(-1)

            if self.previous_weights is not None and name.split(
                    '.')[-1] not in self.skipped_layers:
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
