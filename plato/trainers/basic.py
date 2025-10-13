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


class TrainerWithTimmScheduler(Trainer):
    """
    Trainer that works with timm schedulers using the composable architecture.

    This trainer uses a custom TimmLRSchedulerStrategy to handle timm's
    step_update() method that needs to be called after each training step.
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

        # Initialize parent with timm strategy
        # We need to bypass Trainer.__init__ and call ComposableTrainer directly
        ComposableTrainer.__init__(
            self,
            model=model,
            callbacks=callbacks,
            loss_strategy=None,
            optimizer_strategy=None,
            training_step_strategy=None,
            lr_scheduler_strategy=timm_scheduler_strategy,
            model_update_strategy=None,
            data_loader_strategy=None,
        )

        # Legacy attributes for backward compatibility
        self._loss_criterion = None

    def train_model(self, config, trainset, sampler, **kwargs):
        """Override to inject epoch start and step hooks for timm scheduler."""
        # Store reference to strategy for hook calls
        timm_strategy = self.lr_scheduler_strategy

        # Call epoch start hook
        original_train_model = super().train_model

        # We need to override the training loop to call timm-specific hooks
        batch_size = config["batch_size"]
        self.sampler = sampler
        self.context.config = config
        self.context.current_round = self.current_round

        # Reset tracking
        self.run_history.reset()
        self._loss_tracker.reset()

        # Callbacks: train run start
        self.callback_handler.call_event("on_train_run_start", self, config)

        # Strategy hook: on_train_start
        self.model_update_strategy.on_train_start(self.context)

        # Create data loader using strategy
        self.train_loader = self.data_loader_strategy.create_train_loader(
            trainset, sampler, batch_size, self.context
        )

        # Store train_loader in context for potential use by strategies
        self.context.state["train_loader"] = self.train_loader

        # Create optimizer using strategy
        self.optimizer = self.optimizer_strategy.create_optimizer(
            self.model, self.context
        )

        # Create LR scheduler using strategy (timm-aware)
        self.lr_scheduler = self.lr_scheduler_strategy.create_scheduler(
            self.optimizer, self.context
        )

        # Move model to device
        self.model.to(self.device)
        self.model.train()

        # Training epochs
        total_epochs = config["epochs"]
        tic = time.perf_counter()

        for self.current_epoch in range(1, total_epochs + 1):
            self.context.current_epoch = self.current_epoch
            self._loss_tracker.reset()

            # Timm-specific: epoch start hook
            timm_strategy.on_epoch_start(self.lr_scheduler, self.context)

            # Callbacks: epoch start
            self.callback_handler.call_event("on_train_epoch_start", self, config)

            # Training steps
            for batch_id, (examples, labels) in enumerate(self.train_loader):
                # Store current batch in context
                self.context.state["current_batch"] = batch_id

                # Callbacks: step start
                self.callback_handler.call_event(
                    "on_train_step_start", self, config, batch=batch_id
                )

                # Strategy hook: before_step
                self.model_update_strategy.before_step(self.context)

                # Move data to device
                examples = examples.to(self.device)
                labels = labels.to(self.device)

                # Create loss criterion callable
                def compute_loss(outputs, labels_inner):
                    return self.loss_strategy.compute_loss(
                        outputs, labels_inner, self.context
                    )

                # Perform training step using strategy
                loss = self.training_step_strategy.training_step(
                    model=self.model,
                    optimizer=self.optimizer,
                    examples=examples,
                    labels=labels,
                    loss_criterion=compute_loss,
                    context=self.context,
                )

                # Track loss
                self._loss_tracker.update(loss, labels.size(0))

                # Store last loss in context
                self.context.state["last_loss"] = loss.item()

                # Strategy hook: after optimizer step
                self.optimizer_strategy.on_optimizer_step(self.optimizer, self.context)

                # Timm-specific: step hook
                timm_strategy.on_step(self.lr_scheduler, self.context)

                # Strategy hook: after_step
                self.model_update_strategy.after_step(self.context)

                # Callbacks: step end
                self.callback_handler.call_event(
                    "on_train_step_end", self, config, batch=batch_id, loss=loss
                )

            # LR scheduler epoch step (timm-aware)
            self.lr_scheduler_strategy.step(self.lr_scheduler, self.context)

            # Handle optimizer params state update if needed
            if hasattr(self.optimizer, "params_state_update"):
                self.optimizer.params_state_update()

            # Simulate client's speed
            if (
                self.client_id != 0
                and hasattr(Config().clients, "speed_simulation")
                and Config().clients.speed_simulation
            ):
                self.simulate_sleep_time()

            # Save model for asynchronous mode
            if (
                hasattr(Config().server, "request_update")
                and Config().server.request_update
            ):
                self.model.cpu()
                training_time = time.perf_counter() - tic
                filename = f"{self.client_id}_{self.current_epoch}_{training_time}.pth"
                self.save_model(filename)
                self.model.to(self.device)

            # Update metrics
            self.run_history.update_metric("train_loss", self._loss_tracker.average)

            # Callbacks: epoch end
            self.callback_handler.call_event("on_train_epoch_end", self, config)

        # Strategy hook: on_train_end
        self.model_update_strategy.on_train_end(self.context)

        # Callbacks: train run end
        self.callback_handler.call_event("on_train_run_end", self, config)
