"""
Customized Trainer for PerFedRLNAS using the composable trainer architecture.

This module implements custom strategies and callbacks for Neural Architecture Search
in federated learning, including memory-aware batch size adjustment and subnet
configuration management.
"""

import logging
import os
import pickle
import random
import re

import fednas_specific
import fedtools
import torch
from model.mobilenetv3_supernet import NasDynamicModel

from plato.callbacks.trainer import TrainerCallback
from plato.config import Config
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.base import (
    DataLoaderStrategy,
    LossCriterionStrategy,
    OptimizerStrategy,
    TrainingStepStrategy,
)
from plato.trainers.strategies.lr_scheduler import TimmLRSchedulerStrategy


class SimuRuntimeError(RuntimeError):
    """Simulated runtime error for memory overflow simulation."""


# ============================================================================
# Custom Strategies
# ============================================================================


class FedNASLossStrategy(LossCriterionStrategy):
    """Loss criterion strategy for FedNAS using custom NAS loss."""

    def setup(self, context):
        """Initialize the NAS-specific loss criterion."""
        self._criterion = fednas_specific.get_nasvit_loss_criterion()

    def compute_loss(self, outputs, labels, context):
        """Compute loss using the NAS-specific criterion."""
        return self._criterion(outputs, labels)


class FedNASOptimizerStrategy(OptimizerStrategy):
    """Optimizer strategy for FedNAS using custom optimizer."""

    def create_optimizer(self, model, context):
        """Create the NAS-specific optimizer."""
        return fednas_specific.get_optimizer(model)


class MemoryTrackingTrainingStepStrategy(TrainingStepStrategy):
    """Training step strategy with GPU memory tracking for async training."""

    def __init__(self):
        """Initialize the strategy."""
        self.max_mem_allocated = 0

    def training_step(
        self, model, optimizer, examples, labels, loss_criterion, context
    ):
        """Perform training step with memory tracking."""
        device = context.device
        device_type = device.type if hasattr(device, "type") else str(device)

        # Synchronize or reset memory tracking
        if device_type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        elif device_type == "mps":
            torch.mps.synchronize()

        # Standard training step
        optimizer.zero_grad()
        outputs = model(examples)
        loss = loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Post-training synchronization and memory tracking
        if device_type == "cuda":
            torch.cuda.synchronize(device)
            max_mem = torch.cuda.max_memory_allocated(device) / 1024**3
            self.max_mem_allocated = max(max_mem, self.max_mem_allocated)
        elif device_type == "mps":
            torch.mps.synchronize()

        return loss


class DynamicBatchSizeDataLoaderStrategy(DataLoaderStrategy):
    """Data loader strategy with dynamic batch size adjustment."""

    def __init__(self):
        """Initialize the strategy."""
        self.batch_size = None
        self.unavailable_batch = 1024

    def create_train_loader(self, trainset, sampler, batch_size, context):
        """Create training data loader with potentially adjusted batch size."""
        # Use adjusted batch size if set, otherwise use config batch size
        if self.batch_size is None:
            self.batch_size = batch_size

        # Handle different sampler types properly
        if sampler is not None:
            if isinstance(sampler, torch.utils.data.Sampler):
                # It's already a PyTorch Sampler object
                sampler_obj = sampler
            elif isinstance(sampler, (list, range)):
                # It's a list of indices, create SubsetRandomSampler
                sampler_obj = torch.utils.data.SubsetRandomSampler(sampler)
            elif hasattr(sampler, "get"):
                # It's a Plato Sampler, call get() to obtain PyTorch sampler
                sampler_obj = sampler.get()
            else:
                # Unknown type, try to use it directly
                sampler_obj = sampler
        else:
            sampler_obj = None

        return torch.utils.data.DataLoader(
            dataset=trainset,
            shuffle=False,
            batch_size=self.batch_size,
            sampler=sampler_obj,
        )

    def adjust_batch_size_down(self):
        """Decrease the batch size after memory overflow."""
        self.unavailable_batch = min(self.unavailable_batch, self.batch_size)
        self.batch_size = max(self.batch_size // 2, 1)

    def adjust_batch_size_up(self, config):
        """Increase the batch size if memory allows."""
        if self.batch_size * 2 <= self.unavailable_batch:
            self.batch_size *= 2


# ============================================================================
# Custom Callbacks
# ============================================================================


class MemorySimulationCallback(TrainerCallback):
    """Callback for simulating memory constraints in async training."""

    def __init__(self):
        """Initialize the callback."""
        self.sim_mem = None
        self.max_mem = None
        self.min_mem = None
        self.exceed_memory = False

        if hasattr(Config().parameters, "simulate"):
            self.max_mem = Config().parameters.simulate.max_mem
            self.min_mem = Config().parameters.simulate.min_mem

    def on_train_run_start(self, trainer, config, **kwargs):
        """Initialize simulated memory at the start of training."""
        if hasattr(Config().parameters, "simulate"):
            self.sim_mem = (
                random.random() * (self.max_mem - self.min_mem) + self.min_mem
            )
            if isinstance(
                trainer.training_step_strategy, MemoryTrackingTrainingStepStrategy
            ):
                trainer.training_step_strategy.max_mem_allocated = 0

        # Reset batch size tracking
        if isinstance(trainer.data_loader_strategy, DynamicBatchSizeDataLoaderStrategy):
            trainer.data_loader_strategy.unavailable_batch = 1024
            trainer.data_loader_strategy.batch_size = config["batch_size"]

    def on_train_step_end(self, trainer, config, batch, loss, **kwargs):
        """Check memory usage and adjust batch size after each step."""
        if not isinstance(
            trainer.training_step_strategy, MemoryTrackingTrainingStepStrategy
        ):
            return

        max_mem_allocated = trainer.training_step_strategy.max_mem_allocated

        # Check if memory limit exceeded
        if max_mem_allocated > self.sim_mem:
            raise SimuRuntimeError("Simulated memory overflow")

        # Check if we can increase batch size
        if max_mem_allocated < Config().trainer.mem_usage * self.sim_mem:
            if isinstance(
                trainer.data_loader_strategy, DynamicBatchSizeDataLoaderStrategy
            ):
                trainer.data_loader_strategy.adjust_batch_size_up(config)


class SubnetConfigCallback(TrainerCallback):
    """Callback for saving subnet configurations for NAS models."""

    def __init__(self):
        """Initialize the callback."""
        self.current_config = None

    def on_train_epoch_end(self, trainer, config, **kwargs):
        """Save subnet configuration at the end of each epoch."""
        if (
            hasattr(Config().server, "request_update")
            and Config().server.request_update
        ):
            filename = f"{trainer.client_id}.pkl"
            model_path = Config().params["model_path"]
            full_path = f"{model_path}/{filename}"
            with open(full_path, "wb") as history_file:
                pickle.dump(self.current_config, history_file)


class MemorySavingCallback(TrainerCallback):
    """Callback for saving memory statistics."""

    def on_train_run_end(self, trainer, config, **kwargs):
        """Save memory statistics at the end of training."""
        if "max_concurrency" not in config:
            return

        if not isinstance(
            trainer.training_step_strategy, MemoryTrackingTrainingStepStrategy
        ):
            return

        max_mem_allocated = trainer.training_step_strategy.max_mem_allocated
        exceed_memory = False  # Can be tracked by MemorySimulationCallback if needed
        sim_mem = None

        # Get sim_mem from MemorySimulationCallback if present
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, MemorySimulationCallback):
                sim_mem = callback.sim_mem
                exceed_memory = callback.exceed_memory
                break

        model_name = config["model_name"]
        filename = f"{model_name}_{trainer.client_id}_{config['run_id']}.mem"
        save_memory((max_mem_allocated, exceed_memory, sim_mem), filename)


# ============================================================================
# Helper Functions
# ============================================================================


def save_memory(memory, filename=None):
    """Save memory statistics to a file."""
    model_path = Config().params["model_path"]
    model_name = Config().trainer.model_name

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if filename is not None:
        memory_path = f"{model_path}/{filename}"
    else:
        memory_path = f"{model_path}/{model_name}.mem"

    with open(memory_path, "wb") as file:
        pickle.dump(memory, file)


def load_memory(filename=None):
    """Load memory statistics from a file."""
    model_path = Config().params["model_path"]
    model_name = Config().trainer.model_name

    if filename is not None:
        memory_path = f"{model_path}/{filename}"
    else:
        memory_path = f"{model_path}/{model_name}.mem"

    with open(memory_path, "rb") as file:
        memory = pickle.load(file)

    return memory


# ============================================================================
# Trainer Classes
# ============================================================================


class TrainerSync(ComposableTrainer):
    """Synchronous trainer for FedNAS using custom loss and optimizer."""

    def __init__(self, model=None, callbacks=None):
        """Initialize the synchronous trainer with FedNAS strategies."""
        # Create FedNAS-specific strategies
        loss_strategy = FedNASLossStrategy()
        optimizer_strategy = FedNASOptimizerStrategy()

        # Determine if we need timm scheduler
        lr_scheduler_strategy = None
        if Config().trainer.lr_scheduler == "timm":
            lr_scheduler_strategy = TimmLRSchedulerStrategy()

        # Initialize with strategies
        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=loss_strategy,
            optimizer_strategy=optimizer_strategy,
            lr_scheduler_strategy=lr_scheduler_strategy,
        )

    @staticmethod
    def save_memory(memory, filename=None):
        """Save memory statistics to a file."""
        return save_memory(memory, filename)

    @staticmethod
    def load_memory(filename=None):
        """Load memory statistics from a file."""
        return load_memory(filename)


class TrainerAsync(ComposableTrainer):
    """Asynchronous trainer for FedNAS with memory simulation and dynamic batch size."""

    def __init__(self, model=None, callbacks=None):
        """Initialize the async trainer with FedNAS strategies and callbacks."""
        # Create FedNAS-specific strategies
        loss_strategy = FedNASLossStrategy()
        optimizer_strategy = FedNASOptimizerStrategy()
        training_step_strategy = MemoryTrackingTrainingStepStrategy()
        data_loader_strategy = DynamicBatchSizeDataLoaderStrategy()

        # Determine if we need timm scheduler
        lr_scheduler_strategy = None
        if Config().trainer.lr_scheduler == "timm":
            lr_scheduler_strategy = TimmLRSchedulerStrategy()

        # Create callbacks for async training
        async_callbacks = [
            MemorySimulationCallback,
            SubnetConfigCallback,
            MemorySavingCallback,
        ]
        if callbacks is not None:
            async_callbacks.extend(callbacks)

        # Initialize with strategies and callbacks
        super().__init__(
            model=model,
            callbacks=async_callbacks,
            loss_strategy=loss_strategy,
            optimizer_strategy=optimizer_strategy,
            training_step_strategy=training_step_strategy,
            lr_scheduler_strategy=lr_scheduler_strategy,
            data_loader_strategy=data_loader_strategy,
        )

        # Store reference to strategies for easy access
        self.memory_tracking_strategy = training_step_strategy
        self.dynamic_batch_strategy = data_loader_strategy

    def train_process(self, config, trainset, sampler, **kwargs):
        """Training process with retry logic for memory overflow."""
        while True:
            try:
                self.train_model(config, trainset, sampler.get(), **kwargs)
                break
            except SimuRuntimeError:
                # Adjust batch size and retry
                self.dynamic_batch_strategy.adjust_batch_size_down()
                logging.info(
                    "[Client #%d] Memory overflow, reducing batch size to %d",
                    self.client_id,
                    self.dynamic_batch_strategy.batch_size,
                )
            except Exception as training_exception:
                logging.info("Training on client #%d failed.", self.client_id)
                raise training_exception

        # Save model after successful training
        if "max_concurrency" in config:
            self.model.cpu()
            model_name = config["model_name"]
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.pth"
            self.save_model(filename)

    def obtain_model_at_time(self, client_id, requested_time):
        """
        Obtain a saved NAS model for a particular epoch at the requested wall-clock time.

        This override handles loading NAS-specific models with subnet configurations.
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

        # Load subnet configuration
        with open(
            f"{Config().params['model_path']}/{client_id}.pkl", "rb"
        ) as history_file:
            subnet_config = pickle.load(history_file)

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

                # Create NAS model with subnet configuration
                model = fedtools.sample_subnet_w_config(
                    NasDynamicModel(), subnet_config, False
                )
                model.load_state_dict(pretrained, strict=True)

                logging.info(
                    "[Client #%s] Responding to the server with the model after "
                    "epoch %s finished, at time %s.",
                    client_id,
                    epoch,
                    model_training_time,
                )

                return model

        # If no matching epoch found, return the last available model
        model_path = f"{Config().params['model_path']}/{model_checkpoint}"

        pretrained = None
        if torch.cuda.is_available():
            pretrained = torch.load(model_path)
        else:
            pretrained = torch.load(model_path, map_location=torch.device("cpu"))

        model = fedtools.sample_subnet_w_config(NasDynamicModel(), subnet_config, False)
        model.load_state_dict(pretrained, strict=True)

        logging.info(
            "[Client #%s] Responding to the server with the model after "
            "epoch %s finished, at time %s.",
            client_id,
            epoch,
            model_training_time,
        )

        return model

    @staticmethod
    def save_memory(memory, filename=None):
        """Save memory statistics to a file."""
        return save_memory(memory, filename)

    @staticmethod
    def load_memory(filename=None):
        """Load memory statistics from a file."""
        return load_memory(filename)


# ============================================================================
# Trainer Selection
# ============================================================================

if hasattr(Config().server, "synchronous") and not Config().server.synchronous:
    Trainer = TrainerAsync
else:
    Trainer = TrainerSync
