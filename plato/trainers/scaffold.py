"""
A customized trainer for SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
(https://arxiv.org/pdf/1910.06378.pdf)
"""
import torch
from plato.utils import optimizers

from plato.trainers import basic


class Trainer(basic.Trainer):
    """The federated learning trainer for the SCAFFOLD client. """
    def __init__(self, model=None):
        """Initializing the trainer with the provided model.

        Arguments:
        model: The model to train.
        client_id: The ID of the client using this trainer (optional).
        """
        super().__init__(model)

        self.server_update_direction = None
        self.client_update_direction = None
        self.new_client_update_direction = None

    def pause_training(self):
        c_file = f"new_client_update_direction_{self.client_id}.pth"
        self.new_client_update_direction = torch.load(c_file)
        super().pause_training()

    def get_optimizer(self, model):
        """Initialize the SCAFFOLD optimizer."""
        optimizer = optimizers.get_optimizer(model)

        optimizer.server_update_direction = self.server_update_direction
        optimizer.client_update_direction = self.client_update_direction
        optimizer.client_id = self.client_id
        optimizer.update_flag = True

        return optimizer
