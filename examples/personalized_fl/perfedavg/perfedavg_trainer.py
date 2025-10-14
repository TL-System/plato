"""
A personalized federated learning trainer using Per-FedAvg.

This trainer uses the composable trainer architecture with custom strategies
to implement meta-learning style training with dual learning rates.
"""

import copy

import torch

from plato.callbacks.trainer import TrainerCallback
from plato.config import Config
from plato.trainers import basic
from plato.trainers.strategies.base import TrainingContext, TrainingStepStrategy


class PerFedAvgTrainingStepStrategy(TrainingStepStrategy):
    """
    Training step strategy for Per-FedAvg algorithm.

    Per-FedAvg uses a meta-learning approach with two learning rates:
    1. Alpha: for initial model update
    2. Beta: for meta-gradient computation

    During regular training, it performs:
    1. Update model with learning rate alpha
    2. Compute meta-gradients with learning rate beta on a different batch
    3. Restore original weights and apply meta-gradients

    During personalization phase, it uses standard training.
    """

    def __init__(self):
        """Initialize the Per-FedAvg training step strategy."""
        self.iter_trainloader = None

    def training_step(
        self,
        model,
        optimizer,
        examples,
        labels,
        loss_criterion: callable,
        context: TrainingContext,
    ) -> torch.Tensor:
        """Perform Per-FedAvg training step based on current phase."""
        current_round = context.current_round

        # Personalization phase: use standard training
        if current_round > Config().trainer.rounds:
            optimizer.zero_grad()
            outputs = model(examples)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            return loss

        # Regular training phase: use Per-FedAvg meta-learning
        else:
            # Save a copy of the current model weights
            past_model_params = copy.deepcopy(list(model.parameters()))

            # Step 1: Update the model with a fixed learning rate, alpha
            for g in optimizer.param_groups:
                g["lr"] = Config().algorithm.alpha

            optimizer.zero_grad()
            logits = model(examples)
            loss = loss_criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Step 2: Compute the meta gradients with a fixed learning rate, beta
            for g in optimizer.param_groups:
                g["lr"] = Config().algorithm.beta

            optimizer.zero_grad()

            # Get next batch from train loader
            train_loader = context.state.get("train_loader")
            if self.iter_trainloader is None and train_loader is not None:
                self.iter_trainloader = iter(train_loader)

            try:
                examples, labels = next(self.iter_trainloader)
            except StopIteration:
                # Restart iterator if we've exhausted the dataset
                self.iter_trainloader = iter(train_loader)
                examples, labels = next(self.iter_trainloader)

            examples, labels = examples.to(context.device), labels.to(context.device)
            logits = model(examples)
            loss = loss_criterion(logits, labels)
            loss.backward()

            # Step 3: Restore the model weights saved before step 1
            for model_param, past_model_param in zip(
                model.parameters(), past_model_params
            ):
                model_param.data = past_model_param.data.clone()

            # Update the model with the meta gradients from step 2
            optimizer.step()

            return loss


class PerFedAvgCallback(TrainerCallback):
    """Callback to reset the training step strategy's iterator at each epoch."""

    def on_train_epoch_start(self, trainer, config, **kwargs):
        """Reset the data loader iterator at the start of each epoch."""
        if hasattr(trainer.training_step_strategy, "iter_trainloader"):
            train_loader = trainer.context.state.get("train_loader")
            if train_loader is not None:
                trainer.training_step_strategy.iter_trainloader = iter(train_loader)


class Trainer(basic.Trainer):
    """
    A personalized federated learning trainer using Per-FedAvg algorithm.

    Per-FedAvg uses a meta-learning approach with two learning rates (alpha and beta)
    to personalize models for each client. During regular training rounds, it performs
    a two-step process:
    1. Update with learning rate alpha
    2. Compute meta-gradients with learning rate beta on a different batch
    3. Apply meta-gradients to the original model

    It uses the composable trainer architecture with a custom training step strategy.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the Per-FedAvg trainer with custom training step strategy.

        Arguments:
            model: The model to train
            callbacks: List of callback classes or instances
        """
        # Create Per-FedAvg callback
        perfedavg_callback = PerFedAvgCallback()

        # Combine with provided callbacks
        all_callbacks = [perfedavg_callback]
        if callbacks is not None:
            all_callbacks.extend(callbacks)

        # Initialize parent trainer with combined callbacks
        super().__init__(model=model, callbacks=all_callbacks)

        # Replace the training step strategy with Per-FedAvg strategy
        self.training_step_strategy = PerFedAvgTrainingStepStrategy()
