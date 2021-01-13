"""
The federated learning trainer for Adaptive Parameter Freezing.
"""

import torch

from trainers import trainer


class Trainer(trainer.Trainer):
    """The federated learning trainer for Adaptive Parameter Freezing, 
       used by both the client and the server.
    """
    def __init__(self):
        super().__init__()
        self.sync_mask = {}

    def extract_weights(self):
        """Extract weights from a model passed in as a parameter, and apply the sync mask."""
        weights = []

        for name, weight in self.model.to(
                torch.device('cpu')).named_parameters():
            if weight.requires_grad:
                # Rolling back model parameters that should be frozen
                weight.data = torch.where(self.sync_mask[name], weight.data,
                                          self.previous_weights[name].data)
                # Removing model weights that should not be synced with ther server
                weights_to_sync = torch.masked_select(
                    weight.data,
                    self.sync_mask[name]).reshape(weight.data.shape)
                weights.append((name, weights_to_sync))

        return weights

    def load_weights(self, weights):
        """Loading the server model onto this client."""
        # Initialize the synchronization mask if necessary
        self.trainer.initialize_sync()
        if not self.sync_mask:
            for name, weight in self.model.to(
                    torch.device('cpu')).named_parameters():
                if weight.requires_grad:
                    self.sync_mask[name] = torch.ones(weight.data.shape).bool()

        # Masking the weights received and load them into the model
        weights_received = []

        for name, weight in weights:
            weight.data[self.sync_mask[name]] = weight.data.view(-1)
            weights_received.append((name, weight.data))

        updated_state_dict = {}
        for name, weight in weights_received:
            updated_state_dict[name] = weight

        self.model.load_state_dict(updated_state_dict, strict=False)
