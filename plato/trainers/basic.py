"""
The training and testing loops for PyTorch.

This module provides basic trainers using the composable trainer architecture.
The Trainer class uses the ComposableTrainer with default strategies, leveraging
the strategy design pattern.
"""

import copy
import logging
import os
import re
import time
from typing import Optional

import torch

from plato.callbacks.trainer import TrainerCallback
from plato.config import Config
from plato.models import registry as models_registry
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.base import TrainingContext
from plato.trainers.strategies.lr_scheduler import TimmLRSchedulerStrategy


class LegacyHookBridgeCallback(TrainerCallback):
    """
    Bridge callback that calls legacy hook methods for backward compatibility.

    This callback ensures that trainers overriding the old hook methods
    (train_run_start, train_epoch_start, etc.) continue to work with the
    new ComposableTrainer architecture.
    """

    def on_train_run_start(self, trainer, config, **kwargs):
        """Call legacy train_run_start hook."""
        if hasattr(trainer, "train_run_start"):
            trainer.train_run_start(config)

    def on_train_run_end(self, trainer, config, **kwargs):
        """Call legacy train_run_end hook."""
        if hasattr(trainer, "train_run_end"):
            trainer.train_run_end(config, **kwargs)

    def on_train_epoch_start(self, trainer, config, **kwargs):
        """Call legacy train_epoch_start hook."""
        if hasattr(trainer, "train_epoch_start"):
            trainer.train_epoch_start(config)

    def on_train_epoch_end(self, trainer, config, **kwargs):
        """Call legacy train_epoch_end hook."""
        if hasattr(trainer, "train_epoch_end"):
            trainer.train_epoch_end(config)

    def on_train_step_start(self, trainer, config, batch, **kwargs):
        """Call legacy train_step_start hook."""
        if hasattr(trainer, "train_step_start"):
            trainer.train_step_start(config, batch=batch)

    def on_train_step_end(self, trainer, config, batch, loss, **kwargs):
        """Call legacy train_step_end hook."""
        if hasattr(trainer, "train_step_end"):
            trainer.train_step_end(config, batch=batch, loss=loss)


class Trainer(ComposableTrainer):
    """
    A basic federated learning trainer using the composable architecture.

    This trainer extends ComposableTrainer with default strategies.

    For advanced customization, use ComposableTrainer directly with custom strategies.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the basic trainer with default strategies.

        Arguments:
            model: The model to train (class or instance)
            callbacks: List of callback classes or instances
        """
        # Add bridge callback to support legacy hook methods
        callbacks_with_bridge = [LegacyHookBridgeCallback]
        if callbacks is not None:
            callbacks_with_bridge.extend(callbacks)

        # Initialize with default strategies
        super().__init__(
            model=model,
            callbacks=callbacks_with_bridge,
            loss_strategy=None,  # Uses DefaultLossCriterionStrategy
            optimizer_strategy=None,  # Uses DefaultOptimizerStrategy
            training_step_strategy=None,  # Uses DefaultTrainingStepStrategy
            lr_scheduler_strategy=None,  # Uses DefaultLRSchedulerStrategy
            model_update_strategy=None,  # Uses NoOpUpdateStrategy
            data_loader_strategy=None,  # Uses DefaultDataLoaderStrategy
            testing_strategy=None,  # Uses DefaultTestingStrategy
        )

        # Legacy attributes for backward compatibility
        self._loss_criterion = None

    @property
    def loss_criterion(self):
        """Legacy property for accessing loss criterion."""
        if self._loss_criterion is None:
            # Create loss criterion using the strategy
            def compute_loss_fn(outputs, labels):
                return self.loss_strategy.compute_loss(outputs, labels, self.context)

            self._loss_criterion = compute_loss_fn
        return self._loss_criterion

    # Legacy hook methods for backward compatibility
    # These are no-ops since ComposableTrainer handles them via strategies
    def train_run_start(self, config):
        """Method called at the start of training run (legacy hook)."""
        pass

    def train_run_end(self, config):
        """Method called at the end of a training run (legacy hook)."""
        pass

    def train_epoch_start(self, config):
        """Method called at the beginning of a training epoch (legacy hook)."""
        pass

    def train_epoch_end(self, config):
        """Method called at the end of a training epoch (legacy hook)."""
        pass

    def train_step_start(self, config, batch=None):
        """Method called at the beginning of a training step (legacy hook)."""
        pass

    def train_step_end(self, config, batch=None, loss=None):
        """Method called at the end of a training step (legacy hook)."""
        pass

    # Legacy methods for old obtain_model_update behavior
    def obtain_model_update_legacy(self, client_id, requested_time):
        """
        Obtain a saved model for a particular epoch that finishes just after the provided
        wall clock time is reached.

        This is a legacy method for asynchronous training with wall-clock simulation.
        """
        # Constructing a list of epochs and training times
        models_per_epoch = {}

        for filename in os.listdir(Config().params["model_path"]):
            split = re.match(
                r"(?P<client_id>\d+)_(?P<epoch>\d+)_(?P<training_time>\d+.\d+).pth$",
                filename,
            )

            if split is not None:
                epoch = split.group("epoch")
                training_time = split.group("training_time")
                if client_id == int(split.group("client_id")):
                    models_per_epoch[epoch] = {
                        "training_time": float(training_time),
                        "model_checkpoint": filename,
                    }

        # Locate the model at a specific wall clock time
        for epoch in sorted(models_per_epoch, reverse=True):
            model_training_time = models_per_epoch[epoch]["training_time"]
            model_checkpoint = models_per_epoch[epoch]["model_checkpoint"]

            if model_training_time < requested_time:
                model_path = f"{Config().params['model_path']}/{model_checkpoint}"

                pretrained = None
                if torch.cuda.is_available():
                    pretrained = torch.load(model_path)
                else:
                    pretrained = torch.load(
                        model_path, map_location=torch.device("cpu")
                    )

                model = models_registry.get()
                model.load_state_dict(pretrained, strict=True)

                logging.info(
                    "[Client #%s] Responding to the server with the model after "
                    "epoch %s finished, at time %s.",
                    client_id,
                    epoch,
                    model_training_time,
                )

                return model

        raise ValueError(
            f"[Client #{client_id}] Cannot find an epoch that matches the wall-clock time provided."
        )

    @staticmethod
    def process_outputs(outputs):
        """
        Method called after model outputs are generated.

        This is a legacy method for backward compatibility.
        Override this in subclasses if output processing is needed.
        """
        return outputs


class TimmSchedulerCallback(TrainerCallback):
    """
    Callback that handles timm scheduler-specific hooks.

    This callback calls the on_epoch_start() and on_step() methods
    on TimmLRSchedulerStrategy to handle timm's step_update() functionality.
    """

    def on_train_epoch_start(self, trainer, config, **kwargs):
        """Call timm scheduler's epoch start hook."""
        if isinstance(trainer.lr_scheduler_strategy, TimmLRSchedulerStrategy):
            trainer.lr_scheduler_strategy.on_epoch_start(
                trainer.lr_scheduler, trainer.context
            )

    def on_train_step_end(self, trainer, config, batch, loss, **kwargs):
        """Call timm scheduler's step hook after each training step."""
        if isinstance(trainer.lr_scheduler_strategy, TimmLRSchedulerStrategy):
            trainer.lr_scheduler_strategy.on_step(trainer.lr_scheduler, trainer.context)


class TrainerWithTimmScheduler(Trainer):
    """
    Trainer that works with timm schedulers using the composable architecture.

    This trainer uses a custom TimmLRSchedulerStrategy to handle timm's
    step_update() method that needs to be called after each training step.
    The timm-specific hooks are handled via TimmSchedulerCallback.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize trainer with timm scheduler strategy.

        Arguments:
            model: The model to train (class or instance)
            callbacks: List of callback classes or instances
        """
        # Create timm scheduler strategy
        timm_scheduler_strategy = TimmLRSchedulerStrategy()

        # Add both TimmSchedulerCallback and LegacyHookBridgeCallback
        # to support timm-specific hooks and legacy hook methods
        callbacks_with_timm = [LegacyHookBridgeCallback, TimmSchedulerCallback]
        if callbacks is not None:
            callbacks_with_timm.extend(callbacks)

        # Initialize parent with timm strategy
        # We need to bypass Trainer.__init__ and call ComposableTrainer directly
        ComposableTrainer.__init__(
            self,
            model=model,
            callbacks=callbacks_with_timm,
            loss_strategy=None,
            optimizer_strategy=None,
            training_step_strategy=None,
            lr_scheduler_strategy=timm_scheduler_strategy,
            model_update_strategy=None,
            data_loader_strategy=None,
            testing_strategy=None,
        )

        # Legacy attributes for backward compatibility
        self._loss_criterion = None
