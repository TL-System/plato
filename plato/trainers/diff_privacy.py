"""
Differentially private federated learning trainer using composable architecture.

This module provides a differential privacy trainer that uses the composable
trainer pattern with custom strategies and callbacks instead of inheritance.
"""

import logging
import time
from typing import Optional

import torch
from opacus import GradSampleModule
from opacus.privacy_engine import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, Subset

from plato.callbacks.trainer import TrainerCallback
from plato.config import Config
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.base import (
    DataLoaderStrategy,
    OptimizerStrategy,
    TrainingContext,
    TrainingStepStrategy,
)


class DifferentialPrivacyCallback(TrainerCallback):
    """
    Callback to handle differential privacy setup and cleanup.

    This callback wraps the model with GradSampleModule at training start
    and cleans up the state dict at training end.
    """

    def on_train_run_start(self, trainer, config, **kwargs):
        """Wrap model with GradSampleModule for differential privacy."""
        trainer.model = GradSampleModule(trainer.model)
        logging.info(
            "[Client #%s] Model wrapped with GradSampleModule for differential privacy.",
            trainer.client_id,
        )

    def on_train_run_end(self, trainer, config, **kwargs):
        """
        Clean up GradSampleModule wrapper from state dict.

        After GradSampleModule conversion, state_dict names have a '_module' prefix.
        We need to save weights with the original layer names without the prefix.
        """
        trainer.model_state_dict = {
            k[8:] if "_module." in k else k: v
            for k, v in trainer.model.state_dict().items()
        }
        logging.info(
            "[Client #%s] Cleaned up GradSampleModule wrapper from state dict.",
            trainer.client_id,
        )


class DPDataLoaderStrategy(DataLoaderStrategy):
    """
    Data loader strategy for differential privacy.

    Creates a data loader using a Subset of the original dataset
    (based on the sampler) to enable Opacus poisson sampling.
    """

    def create_train_loader(
        self, trainset, sampler, batch_size: int, context: TrainingContext
    ) -> DataLoader:
        """
        Create data loader using Subset for DP compatibility.

        Args:
            trainset: Training dataset
            sampler: Data sampler (will be used to create subset, not for DataLoader)
            batch_size: Batch size
            context: Training context

        Returns:
            DataLoader without sampler (Opacus will add poisson sampling)
        """
        # Convert sampler to subset indices
        trainset_subset = Subset(trainset, list(sampler))

        # Create DataLoader without sampler - Opacus will recreate it with poisson sampling
        return DataLoader(
            dataset=trainset_subset, shuffle=False, batch_size=batch_size, sampler=None
        )


class DPOptimizerStrategy(OptimizerStrategy):
    """
    Optimizer strategy that wraps the optimizer with Opacus PrivacyEngine.

    This strategy creates a privacy engine and makes the optimizer, model,
    and data loader differentially private.
    """

    def __init__(
        self,
        target_epsilon: float = 10.0,
        target_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        max_physical_batch_size: int = 128,
    ):
        """
        Initialize DP optimizer strategy.

        Args:
            target_epsilon: Target epsilon for differential privacy
            target_delta: Target delta for differential privacy
            max_grad_norm: Maximum gradient norm for clipping
            max_physical_batch_size: Maximum physical batch size for memory management
        """
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.max_physical_batch_size = max_physical_batch_size
        self.privacy_engine = None

    def create_optimizer(
        self, model: torch.nn.Module, context: TrainingContext
    ) -> torch.optim.Optimizer:
        """
        Create and wrap optimizer with PrivacyEngine.

        Args:
            model: The model to optimize
            context: Training context

        Returns:
            Optimizer wrapped with differential privacy
        """
        # Import locally to get the optimizer from registry
        from plato.trainers import optimizers

        # Create base optimizer
        optimizer = optimizers.get(model)

        # Get config values, with defaults
        config = context.config
        target_epsilon = config.get("dp_epsilon", self.target_epsilon)
        target_delta = config.get("dp_delta", self.target_delta)
        max_grad_norm = config.get("dp_max_grad_norm", self.max_grad_norm)
        epochs = config.get("epochs", 1)

        # Get train loader from context
        train_loader = context.state.get("train_loader")
        if train_loader is None:
            raise ValueError("Train loader must be created before optimizer")

        logging.info(
            "[Client #%s] Using differential privacy during training.",
            context.client_id,
        )

        # Create privacy engine
        self.privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)

        # Make model, optimizer, and data loader private
        private_model, private_optimizer, private_train_loader = (
            self.privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                epochs=epochs,
                max_grad_norm=max_grad_norm,
            )
        )

        # Update context with private train loader
        context.state["train_loader"] = private_train_loader
        context.state["max_physical_batch_size"] = self.max_physical_batch_size

        # Update model in context (it's now wrapped by privacy engine)
        context.model = private_model

        return private_optimizer


class DPTrainingStepStrategy(TrainingStepStrategy):
    """
    Training step strategy for differential privacy.

    Uses BatchMemoryManager to handle memory-efficient batching with DP.
    """

    def training_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        examples: torch.Tensor,
        labels: torch.Tensor,
        loss_criterion: callable,
        context: TrainingContext,
    ) -> torch.Tensor:
        """
        Perform one DP training step.

        Note: This is called within BatchMemoryManager context,
        so it follows standard training step pattern.

        Args:
            model: The model to train
            optimizer: The DP-wrapped optimizer
            examples: Input batch
            labels: Target labels
            loss_criterion: Loss computation function
            context: Training context

        Returns:
            Loss value for this step
        """
        optimizer.zero_grad(set_to_none=True)

        outputs = model(examples)
        loss = loss_criterion(outputs, labels)

        # Check if create_graph is needed
        config = context.config
        if config.get("create_graph", False):
            loss.backward(create_graph=True)
        else:
            loss.backward()

        optimizer.step()

        return loss


class Trainer(ComposableTrainer):
    """
    A differentially private federated learning trainer using composable architecture.

    This trainer uses the ComposableTrainer with custom strategies and callbacks
    to implement differential privacy without inheritance.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize DP trainer with custom strategies and callbacks.

        Args:
            model: The model to train (class or instance)
            callbacks: Additional callback classes or instances
        """
        # Get DP configuration with safe defaults
        try:
            max_physical_batch_size = (
                Config().trainer.max_physical_batch_size
                if hasattr(Config().trainer, "max_physical_batch_size")
                else 128
            )
        except (ValueError, AttributeError):
            # Config not initialized or trainer not in config
            max_physical_batch_size = 128

        # Create DP-specific strategies
        dp_data_loader_strategy = DPDataLoaderStrategy()
        dp_optimizer_strategy = DPOptimizerStrategy(
            max_physical_batch_size=max_physical_batch_size
        )
        dp_training_step_strategy = DPTrainingStepStrategy()

        # Create DP callback
        dp_callback = DifferentialPrivacyCallback()

        # Combine with user callbacks
        callbacks_with_dp = [dp_callback]
        if callbacks is not None:
            callbacks_with_dp.extend(callbacks)

        # Initialize with DP strategies
        super().__init__(
            model=model,
            callbacks=callbacks_with_dp,
            loss_strategy=None,  # Uses DefaultLossCriterionStrategy
            optimizer_strategy=dp_optimizer_strategy,
            training_step_strategy=dp_training_step_strategy,
            lr_scheduler_strategy=None,  # Uses DefaultLRSchedulerStrategy
            model_update_strategy=None,  # Uses NoOpUpdateStrategy
            data_loader_strategy=dp_data_loader_strategy,
        )

        # Make model compatible with differential privacy
        self.make_model_private()

    def make_model_private(self):
        """Make the model private for use with the differential privacy engine."""
        errors = ModuleValidator.validate(self.model, strict=False)
        if len(errors) > 0:
            self.model = ModuleValidator.fix(self.model)
            errors = ModuleValidator.validate(self.model, strict=False)
            assert len(errors) == 0
            logging.info("Model validated and fixed for differential privacy.")

    def train_model(self, config, trainset, sampler, **kwargs):
        """
        Training loop with BatchMemoryManager for differential privacy.

        This override is needed to wrap the training loop with BatchMemoryManager,
        which is required for memory-efficient DP training.
        """
        batch_size = config["batch_size"]
        self.trainset = trainset
        self.sampler = sampler
        self.context.config = config
        self.context.current_round = self.current_round

        # Reset tracking
        self.run_history.reset()
        self._loss_tracker.reset()

        # Callbacks: train run start (wraps model with GradSampleModule)
        self.callback_handler.call_event("on_train_run_start", self, config)

        # Strategy hook: on_train_start
        self.model_update_strategy.on_train_start(self.context)

        # Create data loader using strategy (creates Subset)
        self.train_loader = self.data_loader_strategy.create_train_loader(
            trainset, sampler, batch_size, self.context
        )

        # Store train_loader in context
        self.context.state["train_loader"] = self.train_loader
        sampled_size = 0
        if sampler is not None and hasattr(sampler, "num_samples"):
            try:
                sampled_size = sampler.num_samples()
            except TypeError:
                sampled_size = 0
        if sampled_size == 0 and self.train_loader is not None:
            loader_sampler = getattr(self.train_loader, "sampler", None)
            if loader_sampler is not None and hasattr(loader_sampler, "__len__"):
                try:
                    sampled_size = len(loader_sampler)
                except TypeError:
                    sampled_size = 0
        if sampled_size == 0 and trainset is not None and hasattr(trainset, "__len__"):
            try:
                sampled_size = len(trainset)
            except TypeError:
                sampled_size = 0
        self.context.state["num_samples"] = sampled_size

        # Create optimizer using strategy (wraps with PrivacyEngine)
        # This also updates train_loader with poisson sampling
        self.optimizer = self.optimizer_strategy.create_optimizer(
            self.model, self.context
        )

        # Get the updated train loader with poisson sampling
        train_loader = self.context.state["train_loader"]
        max_physical_batch_size = self.context.state["max_physical_batch_size"]

        # Update model reference (it's now wrapped by privacy engine)
        self.model = self.context.model

        # Create LR scheduler using strategy
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

            # Wrap with BatchMemoryManager for DP
            with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=max_physical_batch_size,
                optimizer=self.optimizer,
            ) as memory_safe_train_loader:
                self._loss_tracker.reset()

                # Callbacks: epoch start
                self.callback_handler.call_event("on_train_epoch_start", self, config)

                # Training steps
                for batch_id, (examples, labels) in enumerate(memory_safe_train_loader):
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
                    self.optimizer_strategy.on_optimizer_step(
                        self.optimizer, self.context
                    )

                    # Strategy hook: after_step
                    self.model_update_strategy.after_step(self.context)

                    # Callbacks: step end
                    self.callback_handler.call_event(
                        "on_train_step_end", self, config, batch=batch_id, loss=loss
                    )

            # LR scheduler step
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

        # Callbacks: train run end (cleans up GradSampleModule wrapper)
        self.callback_handler.call_event("on_train_run_end", self, config)
