"""
A personalized federated learning trainer using APFL.
"""
import logging
import os

import numpy as np
import torch

from plato.config import Config
from plato.models import registry as models_registry
from plato.trainers import basic


class Trainer(basic.Trainer):
    """
    A trainer using the APFL algorithm to train both global and personalized models.
    """

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        # The alpha used in the APFL paper
        self.alpha = Config().algorithm.alpha

        # A personalized model and its optimizer
        if model is None:
            self.personalized_model = models_registry.get()
        else:
            self.personalized_model = model()

        self.personalized_optimizer = None

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Performing forward and backward passes in the training loop."""
        super().perform_forward_and_backward_passes(config, examples, labels)

        # Perform the local update on self.personalized_model
        self.optimizer.zero_grad()
        self.personalized_optimizer.zero_grad()

        output1 = self.personalized_model(examples)
        output2 = self.model(examples)
        output = self.alpha * output1 + (1 - self.alpha) * output2
        personalized_loss = self._loss_criterion(output, labels)

        personalized_loss.backward()
        self.personalized_optimizer.step()

        return personalized_loss

    def train_run_start(self, config):
        """Load the alpha before starting the training."""
        super().train_run_start(config)

        # Load the alpha from the saved file
        model_path = Config().params["model_path"]
        filename = f"client_{self.client_id}_alpha.pth"
        save_path = os.path.join(model_path, filename)
        if os.path.exists(save_path):
            self.alpha = torch.load(save_path)

        # Load the personalized model
        model_name = Config().trainer.model_name
        model_path = (
            f"{model_path}/{model_name}_{self.client_id}_personalized_model.pth"
        )
        if os.path.exists(model_path):
            self.personalized_model.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu")), strict=True
            )
            logging.info("[Client #%d] Loaded the personalized model.", self.client_id)

        self.personalized_optimizer = self.get_optimizer(self.personalized_model)

        self.personalized_model.to(self.device)
        self.personalized_model.train()

    def train_run_end(self, config):
        """Saves alpha to a file identified by the client id, and saves the personalized model."""
        super().train_run_end(config)

        # Save the alpha to the file
        model_path = Config().params["model_path"]
        filename = f"client_{self.client_id}_alpha.pth"
        save_path = os.path.join(model_path, filename)
        torch.save(self.alpha, save_path)

        # Save the personalized model
        model_name = Config().trainer.model_name
        model_path = (
            f"{model_path}/{model_name}_{self.client_id}_personalized_model.pth"
        )
        torch.save(self.personalized_model.state_dict(), model_path)

    def train_step_end(self, config, batch=None, loss=None):
        """Updates alpha in APFL before each iteration."""
        super().train_step_end(config, batch, loss)

        # Update alpha based on Eq. 10 in the paper
        if Config().algorithm.adaptive_alpha and self.current_epoch == 1 and batch == 0:
            # 0.1 / np.sqrt(1 + args.local_index))
            lr = self.lr_scheduler.get_lr()[0]
            self._update_alpha(lr)

    def _update_alpha(self, eta):
        """Updates alpha based on Eq. 10 in the paper."""
        grad_alpha = 0

        for l_params, p_params in zip(
            self.model.parameters(), self.personalized_model.parameters()
        ):
            dif = p_params.data - l_params.data
            grad = (
                self.alpha * p_params.grad.data + (1 - self.alpha) * l_params.grad.data
            )
            grad_alpha += dif.view(-1).T.dot(grad.view(-1))

        grad_alpha += 0.02 * self.alpha

        alpha_n = self.alpha - eta * grad_alpha
        self.alpha = np.clip(alpha_n.item(), 0.0, 1.0)
