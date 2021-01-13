"""
The federated learning trainer for Adaptive Parameter Freezing.
"""

import copy
import torch

from models.base import Model
from trainers import trainer


class Trainer(trainer.Trainer):
    """The federated learning trainer for Adaptive Parameter Freezing,
       used by both the client and the server.
    """
    def __init__(self, model: Model):
        super().__init__(model)
        self.sync_mask = {}

        # Initialize the synchronization mask
        if not self.sync_mask:
            for name, weight in model.to(
                    torch.device('cpu')).named_parameters():
                if weight.requires_grad:
                    self.sync_mask[name] = torch.ones(weight.data.shape).bool()

        # Initialize the preserved weights
        self.previous_weights = None
        self.preserve_weights()

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

    def load_weights(self, weights):
        """Loading the server model onto this client."""
        # Masking the weights received and load them into the model
        weights_received = []

        for name, weight in weights:
            weight.data[self.sync_mask[name]] = weight.data.view(-1)
            weights_received.append((name, weight.data))

        updated_state_dict = {}
        for name, weight in weights_received:
            updated_state_dict[name] = weight

        self.model.load_state_dict(updated_state_dict, strict=False)
