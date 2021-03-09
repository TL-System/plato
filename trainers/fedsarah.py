"""
A customized trainer for FedSarah.
"""
import torch
from models.base import Model
from trainers import trainer
from utils import optimizers
from config import Config
import numpy as np


class Trainer(trainer.Trainer):
    """The federated learning trainer for the FedSarah client"""
    def __init__(self, model: Model, client_id=0):
        """Initializing the trainer with the provided model.

        Arguments:
        model: The model to train. Must be a models.base.Model subclass.
        client_id: The ID of the client using this trainer (optional).
        """
        super().__init__(model, client_id)

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

        if self.adjustment:
            optimizer.epsilon = optimizer.max_epsilon - (
                optimizer.max_epsilon - optimizer.min_epsilon) * np.exp(
                    -1 * optimizer.epsilon_decay * self.fl_round_counter)
            #optimizer.epsilon = optimizer.min_epsilon + (
            #optimizer.max_epsilon - optimizer.min_epsilon) * np.exp(
            #   -1 * optimizer.epsilon_decay * self.fl_round_counter)
        else:
            optimizer.epsilon = optimizer.min_epsilon

        return optimizer
