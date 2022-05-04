"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
import logging
import os
import torch
import scaffold_optimizer

from plato.config import Config
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

        self.server_control_variate = None
        self.client_control_variate = None

    def get_optimizer(self, model):
        """Initialize the SCAFFOLD optimizer."""
        optimizer = scaffold_optimizer.ScaffoldOptimizer(
            model.parameters(),
            lr=Config().trainer.learning_rate,
            momentum=Config().trainer.momentum,
            weight_decay=Config().trainer.weight_decay)

        optimizer.server_control_variate = self.server_control_variate
        optimizer.client_control_variate = self.client_control_variate
        optimizer.device = self.device

        return optimizer

    def save_model(self, filename=None, location=None):
        """Saving the model to a file."""
        super().save_model(filename=filename, location=location)

        if self.client_id == 0:
            # Also save the control variate
            model_dir = Config(
            ).params['model_dir'] if location is None else location

            if filename is not None:
                control_variate_path = f'{model_dir}/{filename}'.replace(
                    ".pth", "_control_variate.pth")
            else:
                model_name = Config().trainer.model_name
                control_variate_path = f'{model_dir}/{model_name}_control_variate.pth'

            logging.info("[Server #%d] Saving the control variate to %s.",
                         os.getpid(), control_variate_path)
            torch.save(self.server_control_variate, control_variate_path)
            logging.info("[Server #%d] Control variate saved to %s.",
                         os.getpid(), control_variate_path)

    def load_model(self, filename=None, location=None):
        """Loading pre-trained model weights from a file."""
        super().load_model(filename=filename, location=location)

        # The server loads its control variate
        if self.client_id == 0:
            model_dir = Config(
            ).params['model_dir'] if location is None else location

            if filename is not None:
                control_variate_path = f'{model_dir}/{filename}'.replace(
                    ".pth", "_control_variate.pth")
            else:
                model_name = Config().trainer.model_name
                control_variate_path = f'{model_dir}/{model_name}_control_variate.pth'

            if os.path.exists(control_variate_path):
                logging.info("[Server #%d] Loading a control variate from %s.",
                             os.getpid(), control_variate_path)
                self.server_control_variate = torch.load(control_variate_path)
                logging.info(
                    "[Server #%d] Loaded its control variate from %s.",
                    os.getpid(), control_variate_path)
