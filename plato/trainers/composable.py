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
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, List, Optional, Union, cast

import torch
import torch.nn as nn

from plato.callbacks.handler import CallbackHandler
from plato.callbacks.trainer import LogProgressCallback
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.evaluators.runner import (
    EVALUATION_PRIMARY_KEY,
    EVALUATION_RESULTS_KEY,
    run_configured_evaluation,
)
from plato.models import registry as models_registry
from plato.serialization.safetensor import deserialize_tree, serialize_tree
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
from plato.utils.checkpoint_paths import (
    checkpoint_name,
    checkpoint_path,
    checkpoint_sidecar,
    snapshot_details,
)


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
        model: nn.Module | Callable[[], nn.Module] | None = None,
        callbacks: list[Any] | None = None,
        loss_strategy: LossCriterionStrategy | None = None,
        optimizer_strategy: OptimizerStrategy | None = None,
        training_step_strategy: TrainingStepStrategy | None = None,
        lr_scheduler_strategy: LRSchedulerStrategy | None = None,
        model_update_strategy: ModelUpdateStrategy | None = None,
        data_loader_strategy: DataLoaderStrategy | None = None,
        testing_strategy: TestingStrategy | None = None,
    ):
        """Initialize composable trainer with strategies."""
        super().__init__()

        # Initialize training context
        self.context = TrainingContext()
        device = getattr(self, "device", None)
        if isinstance(device, str):
            device = torch.device(device)
        self.context.device = device
        self.context.client_id = self.client_id

        # Initialize model
        if model is None:
            module: Any = models_registry.get()
        elif isinstance(model, nn.Module):
            # Model instance passed directly
            module = model
        elif callable(model):
            # Model factory/constructor passed
            module = model()
        else:
            module = model

        self.model = module
        self._has_torch_model = isinstance(module, nn.Module)
        self.context.model = module

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
        self.training_start_time = time.time()
        self.model_state_dict = None

    def _require_model(self) -> nn.Module:
        """Return the underlying model, ensuring it is available."""
        if not getattr(self, "_has_torch_model", False):
            raise RuntimeError(
                "ComposableTrainer model has not been initialised correctly."
            )
        return cast(nn.Module, self.model)

    @staticmethod
    def _persisted_test_state_keys() -> tuple[str, ...]:
        """State keys that must survive spawned test subprocesses."""
        return (
            EVALUATION_RESULTS_KEY,
            EVALUATION_PRIMARY_KEY,
            "nanochat_core_results",
        )

    def _test_accuracy_filename(self, run_id: str) -> str:
        model_name = Config().trainer.model_name
        return checkpoint_name(model_name, self.client_id, run_id, suffix=".acc")

    def _test_state_filename(self, run_id: str) -> str:
        model_name = Config().trainer.model_name
        return checkpoint_name(model_name, self.client_id, run_id, suffix=".eval.pkl")

    def _save_test_state(self, filename: str) -> None:
        """Persist evaluation-related context state from a test subprocess."""
        model_path = Config().params["model_path"]
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        state = getattr(self.context, "state", {})
        payload = {
            key: copy.deepcopy(state[key])
            for key in self._persisted_test_state_keys()
            if key in state
        }
        with open(checkpoint_path(model_path, filename), "wb") as state_file:
            pickle.dump(payload, state_file)

    def _load_test_state(self, filename: str) -> None:
        """Restore evaluation-related context state after a spawned test subprocess."""
        model_path = Config().params["model_path"]
        state = getattr(self.context, "state", None)
        if not isinstance(state, dict):
            return

        for key in self._persisted_test_state_keys():
            state.pop(key, None)

        state_path = checkpoint_path(model_path, filename)
        if not os.path.exists(state_path):
            return

        with open(state_path, "rb") as state_file:
            payload = pickle.load(state_file)
        if not isinstance(payload, dict):
            raise TypeError("Persisted test state must be a mapping.")

        for key in self._persisted_test_state_keys():
            if key in payload:
                state[key] = payload[key]

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
        previous_id = self.client_id
        super().set_client_id(client_id)
        self.context.client_id = client_id
        if previous_id != client_id:
            for strategy in (
                self.model_update_strategy, self.loss_strategy,
                self.optimizer_strategy, self.training_step_strategy,
                self.lr_scheduler_strategy, self.testing_strategy,
                self.data_loader_strategy,
            ):
                strategy.on_client_id_changed(self.context)

    def zeros(self, shape):
        """Returns a PyTorch zero tensor with the given shape."""
        assert self.client_id == 0
        return torch.zeros(shape)

    def save_model(self, filename=None, location=None):
        """Save the model to a file."""
        model_path = Config().params["model_path"] if location is None else location
        model_root = model_path
        model_name = Config().trainer.model_name

        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except FileExistsError:
            pass

        model_path = checkpoint_path(
            model_path,
            filename if filename is not None else checkpoint_name(
                model_name, suffix=".safetensors"
            ),
        )
        history_path = checkpoint_sidecar(model_root, model_path)

        model = self._require_model()
        state_dict = (
            self.model_state_dict
            if self.model_state_dict is not None
            else model.state_dict()
        )

        history_payload = pickle.dumps(self.run_history)

        if not model_path.endswith(".safetensors"):
            raise ValueError(
                f"ComposableTrainer.save_model requires a '.safetensors' filename: {model_path}"
            )

        serialized = serialize_tree(state_dict)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as model_file:
            model_file.write(serialized)

        with open(history_path, "wb") as history_file:
            history_file.write(history_payload)

        if self.client_id == 0:
            logging.info("[Server #%d] Model saved to %s.", os.getpid(), model_path)
        else:
            logging.info("[Client #%d] Model saved to %s.", self.client_id, model_path)

    def load_model(self, filename=None, location=None):
        """Load pre-trained model weights from a file."""
        model_path = Config().params["model_path"] if location is None else location
        model_root = model_path
        model_name = Config().trainer.model_name

        model_path = checkpoint_path(
            model_path,
            filename if filename is not None else checkpoint_name(
                model_name, suffix=".safetensors"
            ),
        )
        history_path = checkpoint_sidecar(model_root, model_path)

        if not model_path.endswith(".safetensors"):
            raise ValueError(
                f"ComposableTrainer.load_model requires a '.safetensors' filename: {model_path}"
            )

        if not os.path.exists(model_path):
            raise OSError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as model_file:
            serialized = model_file.read()
        state_dict_raw = deserialize_tree(serialized)
        if not isinstance(state_dict_raw, dict):
            raise TypeError("Deserialised state dict is not a mapping.")
        state_dict = OrderedDict(state_dict_raw.items())
        model = self._require_model()
        model.load_state_dict(state_dict, strict=True)

        logging.info("[Client #%d] Model loaded from %s.", self.client_id, model_path)

        if os.path.exists(history_path):
            with open(history_path, "rb") as history_file:
                self.run_history = pickle.load(history_file)

    def simulate_sleep_time(self):
        """Simulate or enforce wall-clock sleep for straggler emulation."""
        if not (
            hasattr(Config().clients, "speed_simulation")
            and Config().clients.speed_simulation
        ):
            return

        sleep_times = Config.client_sleep_times
        if sleep_times is None:
            sleep_times = Config.simulate_client_speed()

        index = max(self.client_id - 1, 0)
        if index >= len(sleep_times):
            return

        sleep_seconds = max(0.0, float(sleep_times[index]))
        if sleep_seconds <= 0:
            return

        simulate_only = getattr(Config().clients, "sleep_simulation", False)
        if simulate_only:
            # Legacy behaviour: do not block execution, just account for the time.
            return

        logging.info(
            "[Client #%d] Simulating stragglers by sleeping for %.2f seconds.",
            self.client_id,
            sleep_seconds,
        )
        time.sleep(sleep_seconds)

    def train_process(self, config, trainset, sampler, **kwargs):
        """The training process in a federated learning workload."""
        try:
            process_config = {**config, "_defer_strategy_commit": True}
            self.train_model(process_config, trainset, sampler, **kwargs)
            model_name = Config().trainer.model_name
            filename = checkpoint_name(
                model_name, self.client_id, config["run_id"], suffix=".safetensors"
            )
            self.save_model(filename)
            token = config.get("worker_state_token")
            if token is not None and self.model_update_strategy.requires_worker_state:
                state = {
                    "token": token, "client_id": self.client_id,
                    "state": self.model_update_strategy.get_worker_state(self.context),
                }
                with open(self._training_state_path(config["run_id"]), "wb") as state_file:
                    pickle.dump(state, state_file)
            if token is None:
                self.model_update_strategy.on_train_result_accepted(self.context)
        except BaseException:
            self.model_update_strategy.on_train_cleanup(self.context, successful=False)
            raise

    def _training_state_path(self, run_id):
        return checkpoint_path(
            Config().params["model_path"],
            checkpoint_name(Config().trainer.model_name, self.client_id, run_id,
                            suffix=".train.pkl"),
        )

    def train_model(self, config, trainset, sampler, **kwargs):
        """Run local training and always release transient step state."""
        self.context.config = config
        self.context.current_round = self.current_round
        self.context.state.pop("optimizer", None)
        successful = False
        try:
            self.training_step_strategy.on_train_start(self.context)
            result = self._train_model(config, trainset, sampler, **kwargs)
            if not config.get("_defer_strategy_commit"):
                self.model_update_strategy.on_train_result_accepted(self.context)
            successful = True
            return result
        finally:
            self.training_step_strategy.on_train_end(self.context)
            self.model_update_strategy.on_train_cleanup(self.context, successful)
            self.context.state.pop("complete_optimizer_step", None)
            self.context.state.pop("optimizer_step_hooks_handled", None)

    def _train_model(self, config, trainset, sampler, **kwargs):
        """The main training loop using strategies."""
        self.trainset = trainset
        self.sampler = sampler
        self.context.config = config
        self.context.current_round = self.current_round

        # Ensure training step strategy respects higher-order gradient settings
        if self.training_step_strategy is not None:
            if hasattr(self.training_step_strategy, "create_graph"):
                create_graph = config.get("create_graph")
                if isinstance(create_graph, bool):
                    setattr(self.training_step_strategy, "create_graph", create_graph)
            if hasattr(self.training_step_strategy, "retain_graph"):
                retain_graph = config.get("retain_graph")
                if retain_graph is None and config.get("create_graph"):
                    retain_graph = True
                if isinstance(retain_graph, bool):
                    setattr(self.training_step_strategy, "retain_graph", retain_graph)

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
        self.loss_strategy.on_train_start(self.context)

        # Create data loader using strategy
        batch_size = config["batch_size"]
        self.train_loader = self.data_loader_strategy.create_train_loader(
            trainset, sampler, batch_size, self.context
        )

        # Store train_loader in context for potential use by strategies
        self.context.state["train_loader"] = self.train_loader
        try:
            self.context.state["optimizer_updates_per_epoch"] = (
                self.training_step_strategy.optimizer_steps_per_epoch(
                    len(self.train_loader)
                )
            )
        except TypeError:
            self.context.state.pop("optimizer_updates_per_epoch", None)
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
        self.context.state["grad_accum_counter"] = 0
        self.context.state["grad_accum_loss_total"] = 0.0
        self.context.state["grad_accum_loss_count"] = 0

        # Create optimizer using strategy
        model = self._require_model()
        self.optimizer = self.optimizer_strategy.create_optimizer(model, self.context)
        self.context.state["optimizer"] = self.optimizer

        # Create LR scheduler using strategy
        self.lr_scheduler = self.lr_scheduler_strategy.create_scheduler(
            self.optimizer, self.context
        )

        # Move model to device
        model = self._require_model()
        model.to(self.device)
        model.train()

        # Training epochs
        total_epochs = config["epochs"]
        tic = time.perf_counter()
        training_stop_requested = False
        try:
            total_batches = len(self.train_loader)
        except (TypeError, AttributeError):
            total_batches = None

        for self.current_epoch in range(1, total_epochs + 1):
            self.context.current_epoch = self.current_epoch
            self._loss_tracker.reset()
            self.context.state["hf_optimizer_step_index"] = 0

            # Callbacks: epoch start
            self.callback_handler.call_event("on_train_epoch_start", self, config)

            # Training steps
            batches_seen = False
            last_batch_id = -1
            model = self._require_model()
            for batch_id, (examples, labels) in enumerate(self.train_loader):
                # Store current batch in context
                self.context.state["current_batch"] = batch_id
                batches_seen = True
                last_batch_id = batch_id
                is_last_batch = (
                    total_batches is not None and batch_id == total_batches - 1
                )
                self.context.state["is_last_batch"] = is_last_batch

                # Callbacks: step start
                self.callback_handler.call_event(
                    "on_train_step_start", self, config, batch=batch_id
                )

                # Strategy hook: before_step
                self.model_update_strategy.before_step(self.context)

                # Move data to device
                examples = examples.to(self.device)
                if labels is not None:
                    labels = labels.to(self.device)

                # Create loss criterion callable
                def compute_loss(outputs, labels_inner):
                    return self.loss_strategy.compute_loss(
                        outputs, labels_inner, self.context
                    )

                # Perform training step using strategy
                self.context.state.pop("optimizer_step_completed", None)
                self.context.state.pop("optimizer_step_hooks_handled", None)

                def complete_optimizer_step(step_loss):
                    # Dispatch at each actual update so schedulers can affect
                    # the next update of a multi-update training strategy.
                    self.context.state["optimizer_step_completed"] = True
                    self.context.state["optimizer_step_hooks_handled"] = True
                    self.context.state["last_loss"] = step_loss.item()
                    current_optimizer = self.optimizer
                    if current_optimizer is None:
                        raise RuntimeError("Optimizer is missing after a training update.")
                    self.optimizer_strategy.on_optimizer_step(current_optimizer, self.context)
                    self.model_update_strategy.after_step(self.context)
                    self.callback_handler.call_event(
                        "on_train_step_end", self, config, batch=batch_id, loss=step_loss
                    )

                self.context.state["complete_optimizer_step"] = complete_optimizer_step
                try:
                    loss = self.training_step_strategy.training_step(
                        model=model,
                        optimizer=self.optimizer,
                        examples=examples,
                        labels=labels,
                        loss_criterion=compute_loss,
                        context=self.context,
                    )
                finally:
                    self.context.state.pop("complete_optimizer_step", None)

                # Track loss
                self._loss_tracker.update(loss, labels.size(0))

                # Store last loss in context
                self.context.state["last_loss"] = loss.item()
                optimizer_step_done = bool(
                    self.context.state.get("optimizer_step_completed", True)
                )

                if optimizer_step_done:
                    if not self.context.state.pop("optimizer_step_hooks_handled", False):
                        complete_optimizer_step(loss)
                    self.context.state.pop("optimizer_step_completed", None)
                    self.context.state.pop("optimizer_step_hooks_handled", None)

                    control_actions = {}
                    if hasattr(self, "_consume_control_flags"):
                        control_actions = self._consume_control_flags()

                    if control_actions.get("save"):
                        self.save_model()

                    if control_actions.get("evaluate") and hasattr(
                        self, "_handle_control_evaluate"
                    ):
                        self._handle_control_evaluate()

                    if control_actions.get("log") and hasattr(
                        self, "_handle_control_log"
                    ):
                        self._handle_control_log()

                    if control_actions.get("stop_training"):
                        training_stop_requested = True
                        break

                    if control_actions.get("stop_epoch"):
                        break

            finalize_loss = None
            finalize_step_done = False
            finalize_callable = getattr(self.training_step_strategy, "finalize", None)
            if batches_seen and callable(finalize_callable):
                finalize_loss = finalize_callable(
                    model=model,
                    optimizer=self.optimizer,
                    context=self.context,
                )
                finalize_step_done = (
                    bool(self.context.state.get("optimizer_step_completed", False))
                    and finalize_loss is not None
                )
            if finalize_step_done:
                self.optimizer_strategy.on_optimizer_step(self.optimizer, self.context)
                self.model_update_strategy.after_step(self.context)
                self.callback_handler.call_event(
                    "on_train_step_end",
                    self,
                    config,
                    batch=last_batch_id,
                    loss=finalize_loss,
                )
                if finalize_loss is None:
                    last_loss_value = 0.0
                elif hasattr(finalize_loss, "item"):
                    last_loss_value = float(finalize_loss.item())
                else:
                    last_loss_value = float(finalize_loss)
                self.context.state["last_loss"] = last_loss_value
                self.context.state.pop("optimizer_step_completed", None)

                control_actions = {}
                if hasattr(self, "_consume_control_flags"):
                    control_actions = self._consume_control_flags()

                if control_actions.get("save"):
                    self.save_model()

                if control_actions.get("evaluate") and hasattr(
                    self, "_handle_control_evaluate"
                ):
                    self._handle_control_evaluate()

                if control_actions.get("log") and hasattr(self, "_handle_control_log"):
                    self._handle_control_log()

                if control_actions.get("stop_training"):
                    training_stop_requested = True

                if control_actions.get("stop_epoch"):
                    # No batches remain, but respect control flag.
                    pass

            self.context.state.pop("is_last_batch", None)
            self.context.state.pop("hf_optimizer_step_index", None)

            # LR scheduler step
            self.lr_scheduler_strategy.step(self.lr_scheduler, self.context)

            # Handle optimizer params state update if needed
            if hasattr(self.optimizer, "params_state_update"):
                update_fn = getattr(self.optimizer, "params_state_update")
                if callable(update_fn):
                    update_fn()

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
                model = self._require_model()
                model.cpu()
                training_time = time.perf_counter() - tic
                filename = checkpoint_name(
                    self.client_id, self.current_epoch, str(training_time),
                    suffix=".safetensors",
                )
                self.save_model(filename)
                model.to(self.device)

            # Update metrics
            self.run_history.update_metric("train_loss", self._loss_tracker.average)

            # Callbacks: epoch end
            self.callback_handler.call_event("on_train_epoch_end", self, config)

            if training_stop_requested:
                break

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
            self.model_update_strategy.on_train_cleanup(self.context, successful=False)
            config["worker_state_token"] = uuid.uuid4().hex
            state_path = self._training_state_path(config["run_id"])
            if self.model_update_strategy.requires_worker_state and os.path.exists(state_path):
                os.remove(state_path)

            if mp.get_start_method(allow_none=True) != "spawn":
                mp.set_start_method("spawn", force=True)

            if hasattr(torch.multiprocessing, "set_sharing_strategy"):
                try:
                    torch.multiprocessing.set_sharing_strategy("file_system")
                except (RuntimeError, ValueError):
                    logging.debug(
                        "Unable to set torch sharing strategy to file_system."
                    )

            train_proc = mp.Process(
                target=self.train_process,
                args=(config, trainset, sampler),
                kwargs=kwargs,
            )
            train_proc.start()
            train_proc.join()

            if train_proc.exitcode not in (0, None):
                raise ValueError(
                    f"Training worker for client {self.client_id} exited with code {train_proc.exitcode}."
                )

            model_name = Config().trainer.model_name
            filename = (
                checkpoint_name(
                    model_name, self.client_id, Config().params["run_id"],
                    suffix=".safetensors",
                )
            )

            try:
                worker_state = None
                if self.model_update_strategy.requires_worker_state:
                    with open(state_path, "rb") as state_file:
                        worker_state = pickle.load(state_file)
                    if (not isinstance(worker_state, dict)
                            or worker_state.get("token") != config["worker_state_token"]
                            or worker_state.get("client_id") != self.client_id):
                        raise ValueError("Training worker returned stale or mismatched strategy state.")
                self.load_model(filename)
                if worker_state is not None:
                    self.model_update_strategy.load_worker_state(
                        worker_state.get("state"), self.context
                    )
                self.model_update_strategy.on_train_result_accepted(self.context)
            except OSError as error:
                self.model_update_strategy.on_train_cleanup(self.context, successful=False)
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
                self.model_update_strategy.on_train_cleanup(self.context, successful=False)
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

        accuracy_filename = self._test_accuracy_filename(config["run_id"])
        self.save_accuracy(self.accuracy, accuracy_filename)
        self._save_test_state(self._test_state_filename(config["run_id"]))

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
            model = self._require_model()
            model.cpu()

            if mp.get_start_method(allow_none=True) != "spawn":
                mp.set_start_method("spawn", force=True)

            test_proc = mp.Process(
                target=self.test_process,
                args=(config, testset, sampler),
                kwargs=kwargs,
            )
            test_proc.start()
            test_proc.join()

            if test_proc.exitcode not in (0, None):
                raise ValueError(
                    f"Testing worker for client {self.client_id} exited with code {test_proc.exitcode}."
                )

            accuracy_filename = self._test_accuracy_filename(Config().params["run_id"])
            state_filename = self._test_state_filename(Config().params["run_id"])

            try:
                accuracy = self.load_accuracy(accuracy_filename)
            except OSError as error:
                raise ValueError(
                    f"Testing on client {self.client_id} failed."
                ) from error

            self._load_test_state(state_filename)
            self.accuracy = accuracy

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
        model = self._require_model()
        accuracy = self.testing_strategy.test_model(
            model, config, testset, sampler, self.context
        )

        # Store accuracy for compatibility with existing code
        self.accuracy = accuracy

        run_configured_evaluation(
            model=model,
            context=self.context,
            trainer=self,
            tokenizer=getattr(self, "tokenizer", None),
            config=config,
            testset=testset,
            sampler=sampler,
            local_metric=accuracy,
            evaluator_override=getattr(self, "_configured_evaluator_override", None),
        )

        return accuracy

    def obtain_model_update(self, config, trainset, sampler):
        """
        Obtain model updates from training.

        Returns model weights and any additional payload from strategies.
        """
        # Perform training
        self.train_model(config, trainset, sampler)

        # Get model weights
        model = self._require_model()
        model_update = copy.deepcopy(model.state_dict())

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
        Obtain the latest snapshot completed strictly before the requested time.

        This method is used for asynchronous training with wall-clock simulation.
        It searches through saved model checkpoints and returns the model from
        the latest completion time strictly below the cutoff (epoch breaks ties).

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
        root = Config().params["model_path"]
        candidates = []
        if os.path.isdir(root):
            for filename in os.listdir(root):
                details = snapshot_details(filename)
                if (
                    details is None or details[0] != client_id
                    or not filename.endswith(".safetensors")
                ):
                    continue
                _, epoch, finished = details
                if finished < requested_time:
                    candidates.append((finished, epoch, filename))
        if candidates:
            finished, epoch, filename = max(candidates)
            with open(checkpoint_path(root, filename), "rb") as checkpoint:
                pretrained = deserialize_tree(checkpoint.read())
            # Preserve injected architecture and current model/device/mode.
            model = copy.deepcopy(self._require_model())
            model.load_state_dict(pretrained, strict=True)
            logging.info(
                "[Client #%s] Responding with epoch %s, finished at time %s.",
                client_id, epoch, finished,
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
