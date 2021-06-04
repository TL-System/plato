"""
A customized trainer for FedSarah.
"""
import torch
from plato.config import Config
from plato.trainers import basic
from plato.utils import optimizers


class Trainer(basic.Trainer):
    """ The federated learning trainer for the FedSarah client. """
    def __init__(self, client_id=0, model=None):
        """ Initializing the trainer with the provided model.

        Arguments:
            client_id: The ID of the client using this trainer (optional).
            model: The model to train (optional).
        """
        super().__init__(client_id, model)

        self.server_control_variates = None
        self.client_control_variates = None
        self.new_client_control_variates = None
        self.fl_round_counter = None
        self.adjustment = True

    def pause_training(self):
        c_file = f"new_client_control_variates_{self.client_id}.pth"
        self.new_client_control_variates = torch.load(c_file)
        super().pause_training()

    def get_optimizer(self, model):
        """Initialize the FedSarah optimizer."""
        optimizer = optimizers.get_optimizer(model)
        optimizer.server_control_variates = self.server_control_variates
        optimizer.client_control_variates = self.client_control_variates
        optimizer.client_id = self.client_id
        optimizer.max_counter = Config().trainer.epochs

        return optimizer
