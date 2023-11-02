"""
A personalized federated learning trainer using Per-FedAvg
"""

import copy

from plato.config import Config
from plato.trainers import basic
from plato.models import registry as models_registry


class Trainer(basic.Trainer):
    """A personalized federated learning trainer using the Per-FedAvg algorithm."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        # the iterator for the dataloader
        self.iter_trainloader = None

    def train_epoch_start(self, config):
        """Defining the iterator for the train dataloader."""
        super().train_epoch_start(config)
        self.iter_trainloader = iter(self.train_loader)

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Perform forward and backward passes in the training loop."""

        if self.current_round > Config().trainer.rounds:
            # No meta learning in the fine-tuning in the final personalization round
            return super().perform_forward_and_backward_passes(config, examples, labels)
        else:
            alpha = Config().algorithm.alpha
            beta = Config().algorithm.beta

            # Put the current model weights into the other meta model
            past_model_params = copy.deepcopy(list(self.model.parameters()))

            # Step 1
            # Update model with learning rate alpha.
            for g in self.optimizer.param_groups:
                g["lr"] = alpha

            self.optimizer.zero_grad()
            logits = self.model(examples)
            loss = self._loss_criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            # Step 2
            # Calculate the meta gradients
            for g in self.optimizer.param_groups:
                g["lr"] = beta

            self.optimizer.zero_grad()
            examples, labels = next(self.iter_trainloader)
            examples, labels = examples.to(self.device), labels.to(self.device)

            logits = self.model(examples)

            loss = self._loss_criterion(logits, labels)
            self._loss_tracker.update(loss, labels.size(0))
            loss.backward()

            # Step 3
            # Update model weights with meta model's gradients
            # The model parameter is only updated here, in each iteration.
            # Use the gradients by step 2 to update the weights before step 1.
            for model_param, past_model_param in zip(
                self.model.parameters(), past_model_params
            ):
                model_param.data = past_model_param.data.clone()

            self.optimizer.step()

            return loss
