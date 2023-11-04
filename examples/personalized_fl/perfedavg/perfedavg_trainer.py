"""
A personalized federated learning trainer using Per-FedAvg.
"""

import copy

from plato.config import Config
from plato.trainers import basic


class Trainer(basic.Trainer):
    """A personalized federated learning trainer using the Per-FedAvg algorithm."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        self.iter_trainloader = None

    def train_epoch_start(self, config):
        """Runs at the start of each epoch."""
        super().train_epoch_start(config)

        self.iter_trainloader = iter(self.train_loader)

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Performs forward and backward passes in the training loop."""

        if self.current_round > Config().trainer.rounds:
            # During the final personalization round, the normal training loop is used
            return super().perform_forward_and_backward_passes(config, examples, labels)
        else:
            # Save a copy of the current model weights
            past_model_params = copy.deepcopy(list(self.model.parameters()))

            # Step 1: Update the model with a fixed learning rate, alpha, in Algorithm 1
            for g in self.optimizer.param_groups:
                g["lr"] = Config().algorithm.alpha

            self.optimizer.zero_grad()
            logits = self.model(examples)
            loss = self._loss_criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            # Step 2: Compute the meta gradients with a fixed learning rate, beta, in Algorithm 1
            for g in self.optimizer.param_groups:
                g["lr"] = Config().algorithm.beta

            self.optimizer.zero_grad()
            examples, labels = next(self.iter_trainloader)
            examples, labels = examples.to(self.device), labels.to(self.device)
            logits = self.model(examples)
            loss = self._loss_criterion(logits, labels)
            self._loss_tracker.update(loss, labels.size(0))
            loss.backward()

            # Step 3: Restore the model weights saved before step 1
            for model_param, past_model_param in zip(
                self.model.parameters(), past_model_params
            ):
                model_param.data = past_model_param.data.clone()

            # Update the model with the meta gradients from step 2
            self.optimizer.step()

            return loss
