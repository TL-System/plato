"""
Composable trainer using strategy pattern for extensibility.

This module provides the ComposableTrainer class that uses composition and
dependency injection instead of inheritance for customization. Strategies are
injected via the constructor to customize different aspects of training.

Example:
    >>> from plato.trainers.composable import ComposableTrainer
    >>> from plato.trainers.strategies import (
    ...     CrossEntropyLossStrategy,
    ...     AdamOptimizerStrategy,
    ... )
    >>>
    >>> trainer = ComposableTrainer(
    ...     loss_strategy=CrossEntropyLossStrategy(),
    ...     optimizer_strategy=AdamOptimizerStrategy(lr=0.001)
    ... )
"""

import copy
import logging
import multiprocessing as mp
import os
import pickle
import re
import time
from typing import Any, Callable, List, Optional, Union

import torch
import torch.nn as nn

from plato.callbacks.handler import CallbackHandler
from plato.callbacks.trainer import LogProgressCallback
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.models import registry as models_registry
from plato.trainers import base, tracking
from plato.trainers.strategies.base import (
    DataLoaderStrategy,
    LossCriterionStrategy,
    LRSchedulerStrategy,
    ModelUpdateStrategy,
    OptimizerStrategy,
    TestingStrategy,
    TrainingContext,
    TrainingStepStrategy,
)
from plato.trainers.strategies.data_loader import DefaultDataLoaderStrategy
from plato.trainers.strategies.loss_criterion import DefaultLossCriterionStrategy
from plato.trainers.strategies.lr_scheduler import DefaultLRSchedulerStrategy
from plato.trainers.strategies.model_update import NoOpUpdateStrategy
from plato.trainers.strategies.optimizer import DefaultOptimizerStrategy
from plato.trainers.strategies.testing import DefaultTestingStrategy
from plato.trainers.strategies.training_step import DefaultTrainingStepStrategy


class ComposableTrainer(base.Trainer):
    """
    A composable trainer that uses strategies for extensibility.

    Instead of overriding methods, this trainer accepts strategy objects
    that define specific behaviors. This enables composition, makes testing
    easier, and allows combining multiple algorithms.

    Args:
        model: Model class or instance to train
        callbacks: List of callback classes or instances
        loss_strategy: Strategy for computing loss
        optimizer_strategy: Strategy for creating optimizer
        training_step_strategy: Strategy for training step logic
        lr_scheduler_strategy: Strategy for LR scheduling
        model_update_strategy: Strategy for model updates and state management
        data_loader_strategy: Strategy for creating data loaders
        testing_strategy: Strategy for model testing/evaluation

    Example:
        >>> from plato.trainers.strategies import (
        ...     FedProxLossStrategy,
        ...     AdamOptimizerStrategy,
        ... )
        >>>
        >>> trainer = ComposableTrainer(
        ...     loss_strategy=FedProxLossStrategy(mu=0.01),
        ...     optimizer_strategy=AdamOptimizerStrategy(lr=0.001)
        ... )
    """

    def __init__(
        self,
        model: Optional[Union[nn.Module, Callable[[], nn.Module]]] = None,
        callbacks: Optional[List[Any]] = None,
        loss_strategy: Optional[LossCriterionStrategy] = None,
        optimizer_strategy: Optional[OptimizerStrategy] = None,
        training_step_strategy: Optional[TrainingStepStrategy] = None,
        lr_scheduler_strategy: Optional[LRSchedulerStrategy] = None,
        model_update_strategy: Optional[ModelUpdateStrategy] = None,
        data_loader_strategy: Optional[DataLoaderStrategy] = None,
        testing_strategy: Optional[TestingStrategy] = None,
    ):
        """Initialize composable trainer with strategies."""
        super().__init__()

        # Initialize training context
        self.context = TrainingContext()
        self.context.device = self.device
        self.context.client_id = self.client_id

        # Initialize model
        if model is None:
            self.model = models_registry.get()
        elif isinstance(model, nn.Module):
            # Model instance passed directly
            self.model = model
        elif callable(model):
            # Model factory/constructor passed
            self.model = model()
        else:
            self.model = model

        self.context.model = self.model

        # Initialize strategies with defaults
        self.loss_strategy = loss_strategy or DefaultLossCriterionStrategy()
        self.optimizer_strategy = optimizer_strategy or DefaultOptimizerStrategy()
        self.training_step_strategy = (
            training_step_strategy or DefaultTrainingStepStrategy()
        )
        self.lr_scheduler_strategy = (
            lr_scheduler_strategy or DefaultLRSchedulerStrategy()
        )
        self.model_update_strategy = model_update_strategy or NoOpUpdateStrategy()
        self.data_loader_strategy = data_loader_strategy or DefaultDataLoaderStrategy()
        self.testing_strategy = testing_strategy or DefaultTestingStrategy()

        # Setup all strategies
        self._setup_strategies()

        # Initialize callbacks
        self.callbacks = [LogProgressCallback]
        if callbacks is not None:
            self.callbacks.extend(callbacks)
        self.callback_handler = CallbackHandler(self.callbacks)

        # Initialize tracking
        self.run_history = tracking.RunHistory()
        self._loss_tracker = tracking.LossTracker()

        # Training state
        self.trainset = None
        self.train_loader = None
        self.sampler = None
        self.optimizer = None
        self.lr_scheduler = None
        self.current_epoch = 0
        self.current_round = 0
        self.training_start_time = time.time()
        self.model_state_dict = None

    def _setup_strategies(self):
        """Setup all strategies."""
        strategies = [
            self.loss_strategy,
            self.optimizer_strategy,
            self.training_step_strategy,
            self.lr_scheduler_strategy,
            self.model_update_strategy,
            self.data_loader_strategy,
            self.testing_strategy,
        ]

        for strategy in strategies:
            if strategy is not None:
                strategy.setup(self.context)

    def _teardown_strategies(self):
        """Teardown all strategies."""
        strategies = [
            self.loss_strategy,
            self.optimizer_strategy,
            self.training_step_strategy,
            self.lr_scheduler_strategy,
            self.model_update_strategy,
            self.data_loader_strategy,
            self.testing_strategy,
        ]

        for strategy in strategies:
            if strategy is not None:
                strategy.teardown(self.context)

    def set_client_id(self, client_id):
        """Set client ID for both trainer and context."""
        super().set_client_id(client_id)
        self.context.client_id = client_id

    def zeros(self, shape):
        """Returns a PyTorch zero tensor with the given shape."""
        assert self.client_id == 0
        return torch.zeros(shape)

    def save_model(self, filename=None, location=None):
        """Save the model to a file."""
        model_path = Config().params["model_path"] if location is None else location
        model_name = Config().trainer.model_name

        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except FileExistsError:
            pass

        if filename is not None:
            model_path = f"{model_path}/{filename}"
        else:
            model_path = f"{model_path}/{model_name}.pth"

        if self.model_state_dict is None:
            torch.save(self.model.state_dict(), model_path)
        else:
            torch.save(self.model_state_dict, model_path)

        with open(model_path + ".pkl", "wb") as history_file:
            pickle.dump(self.run_history, history_file)

        if self.client_id == 0:
            logging.info("[Server #%d] Model saved to %s.", os.getpid(), model_path)
        else:
            logging.info("[Client #%d] Model saved to %s.", self.client_id, model_path)

    def load_model(self, filename=None, location=None):
        """Load pre-trained model weights from a file."""
        model_path = Config().params["model_path"] if location is None else location
        model_name = Config().trainer.model_name

        if filename is not None:
            model_path = f"{model_path}/{filename}"
        else:
            model_path = f"{model_path}/{model_name}.pth"

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, weights_only=False))

            logging.info(
                "[Client #%d] Model loaded from %s.", self.client_id, model_path
            )

            if os.path.exists(model_path + ".pkl"):
                with open(model_path + ".pkl", "rb") as history_file:
                    self.run_history = pickle.load(history_file)
        else:
            raise OSError(f"Model file not found: {model_path}")

    def simulate_sleep_time(self):
        """Simulate client's speed variation."""
        if (
            hasattr(Config().clients, "sleep_simulation")
            and Config().clients.sleep_simulation
        ):
            sleep_seconds = Config.client_sleep_times[self.client_id - 1]
            sleep_seconds = max(0, sleep_seconds)

            if sleep_seconds > 0:
                logging.info(
                    "[Client #%d] Simulating stragglers by sleeping for %.2f seconds.",
                    self.client_id,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

    def train_process(self, config, trainset, sampler, **kwargs):
        """The training process in a federated learning workload."""
        self.train_model(config, trainset, sampler, **kwargs)

        model_name = Config().trainer.model_name
        filename = f"{model_name}_{self.client_id}_{config['run_id']}.pth"
        self.save_model(filename)

    def train_model(self, config, trainset, sampler, **kwargs):
        """The main training loop using strategies."""
        batch_size = config["batch_size"]
        self.trainset = trainset
        self.sampler = sampler
        self.context.config = config
        self.context.current_round = self.current_round

        # Ensure training step strategy respects higher-order gradient settings
        if self.training_step_strategy is not None:
            if hasattr(self.training_step_strategy, "create_graph"):
                create_graph = config.get("create_graph")
                if create_graph is not None:
                    self.training_step_strategy.create_graph = create_graph
            if hasattr(self.training_step_strategy, "retain_graph"):
                retain_graph = config.get("retain_graph")
                if retain_graph is None and config.get("create_graph"):
                    retain_graph = True
                if retain_graph is not None:
                    self.training_step_strategy.retain_graph = retain_graph

        if trainset is None:
            logging.warning(
                "[Client #%d] No training dataset received in worker process; "
                "reloading from data source.",
                self.client_id,
            )
            try:
                datasource = datasources_registry.get(client_id=self.client_id)
                trainset = datasource.get_train_set()
                self.trainset = trainset
            except Exception as exc:
                logging.error(
                    "[Client #%d] Failed to reload training dataset: %s",
                    self.client_id,
                    exc,
                )
                self.callback_handler.call_event("on_train_run_end", self, config)
                raise

        if sampler is None:
            logging.warning(
                "[Client #%d] No sampler provided; defaulting to full dataset.",
                self.client_id,
            )

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

        # Create optimizer using strategy
        self.optimizer = self.optimizer_strategy.create_optimizer(
            self.model, self.context
        )

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
            self._loss_tracker.reset()

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

        # Callbacks: train run end
        self.callback_handler.call_event("on_train_run_end", self, config)

    def train(self, trainset, sampler, **kwargs) -> float:
        """
        The main training loop in a federated learning workload.

        Args:
            trainset: The training dataset
            sampler: The sampler that extracts a partition for this client
            **kwargs: Additional keyword arguments

        Returns:
            Training time in seconds
        """
        config = Config().trainer._asdict()
        config["run_id"] = Config().params["run_id"]

        # Set the start time of training in absolute time
        self.training_start_time = time.time()

        if "max_concurrency" in config:
            tic = time.perf_counter()

            if mp.get_start_method(allow_none=True) != "spawn":
                mp.set_start_method("spawn", force=True)

            train_proc = mp.Process(
                target=self.train_process,
                args=(config, trainset, sampler),
                kwargs=kwargs,
            )
            train_proc.start()
            train_proc.join()

            model_name = Config().trainer.model_name
            filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}.pth"

            try:
                self.load_model(filename)
            except OSError as error:
                logging.error(
                    "[Client #%d] Failed to load model from %s: %s",
                    self.client_id,
                    filename,
                    error,
                )
                raise ValueError(
                    f"Training on client {self.client_id} failed."
                ) from error
            except Exception as error:
                logging.error(
                    "[Client #%d] Unexpected error loading model: %s",
                    self.client_id,
                    error,
                )
                raise ValueError(
                    f"Training on client {self.client_id} failed."
                ) from error

            toc = time.perf_counter()
            self.pause_training()
        else:
            tic = time.perf_counter()
            self.train_process(config, trainset, sampler, **kwargs)
            toc = time.perf_counter()

        training_time = toc - tic
        return training_time

    def test_process(self, config, testset, sampler=None, **kwargs):
        """The testing loop, run in a separate process."""
        self.test_model(config, testset, sampler, **kwargs)

        model_name = Config().trainer.model_name
        filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
        self.save_accuracy(self.accuracy, filename)

    def test(self, testset, sampler=None, **kwargs) -> float:
        """
        Test the model using the provided test dataset.

        Args:
            testset: The test dataset
            sampler: The sampler for the test dataset
            **kwargs: Additional keyword arguments

        Returns:
            Accuracy on test set
        """
        config = Config().trainer._asdict()
        config["run_id"] = Config().params["run_id"]

        if "max_concurrency" in config:
            self.model.cpu()

            if mp.get_start_method(allow_none=True) != "spawn":
                mp.set_start_method("spawn", force=True)

            test_proc = mp.Process(
                target=self.test_process,
                args=(config, testset, sampler),
                kwargs=kwargs,
            )
            test_proc.start()
            test_proc.join()

            model_name = Config().trainer.model_name
            filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}.acc"

            try:
                accuracy = self.load_accuracy(filename)
            except OSError as error:
                raise ValueError(
                    f"Testing on client {self.client_id} failed."
                ) from error

            self.pause_training()
            return accuracy
        else:
            return self.test_model(config, testset, sampler, **kwargs)

    def test_model(self, config, testset, sampler=None, **kwargs):
        """
        Test the model using the configured testing strategy.

        Args:
            config: Testing configuration dictionary
            testset: Test dataset
            sampler: Optional data sampler for test set
            **kwargs: Additional keyword arguments

        Returns:
            Test accuracy or other metric as float
        """
        # Use testing strategy to perform evaluation
        accuracy = self.testing_strategy.test_model(
            self.model, config, testset, sampler, self.context
        )

        # Store accuracy for compatibility with existing code
        self.accuracy = accuracy

        return accuracy

    def obtain_model_update(self, config, trainset, sampler):
        """
        Obtain model updates from training.

        Returns model weights and any additional payload from strategies.
        """
        # Perform training
        self.train_model(config, trainset, sampler)

        # Get model weights
        model_update = copy.deepcopy(self.model.state_dict())

        # Get additional payload from model update strategy
        additional_payload = self.model_update_strategy.get_update_payload(self.context)

        # Combine model update with additional payload
        if additional_payload:
            return {
                "model_update": model_update,
                **additional_payload,
            }
        else:
            return model_update

    def obtain_model_at_time(self, client_id, requested_time):
        """
        Obtain a saved model for a particular epoch that finishes just after
        the provided wall clock time is reached.

        This method is used for asynchronous training with wall-clock simulation.
        It searches through saved model checkpoints and returns the model from
        the latest epoch that finished before the requested time.

        Subclasses can override this method to provide custom model retrieval logic
        (e.g., loading models with specific architectures or configurations).

        Args:
            client_id: The client ID whose model to retrieve
            requested_time: The wall clock time threshold

        Returns:
            The model corresponding to the requested time

        Raises:
            ValueError: If no model checkpoint matches the wall-clock time provided
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

    def __del__(self):
        """Teardown strategies when trainer is destroyed."""
        try:
            self._teardown_strategies()
        except:
            # Ignore errors during cleanup
            pass
