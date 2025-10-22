"""
Composable trainer runtime primitives for Apple's MLX backend.

This module mirrors the PyTorch-oriented composable trainer architecture with
MLX-native implementations. The goal is to provide drop-in compatible trainer
and strategy primitives so algorithms can target either framework without
modifying higher-level orchestration.
"""

from __future__ import annotations

import logging
import os
import pickle
import random
import time
import types
from abc import ABC, abstractmethod
from collections.abc import Iterable as ABCIterable
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from plato.callbacks.handler import CallbackHandler
from plato.callbacks.trainer import LogProgressCallback
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.models import registry as models_registry
from plato.serialization.safetensor import deserialize_tree, serialize_tree
from plato.trainers import base, tracking

try:  # pragma: no cover - import guard for optional dependency
    import mlx.core as mx
    import mlx.nn as mx_nn
    import mlx.optimizers as mx_optim
    from mlx.nn import utils as nn_utils
except ImportError as err:  # pragma: no cover - handled lazily
    mx = None
    mx_nn = None
    mx_optim = None
    nn_utils = None
    _MLX_IMPORT_ERROR = err
else:  # pragma: no cover - executed only when MLX is available
    _MLX_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None


class MLXNotAvailableError(ImportError):
    """Raised when MLX-specific functionality is requested but MLX is missing."""


def _ensure_mlx_available() -> None:
    """Raise an informative error if MLX is not installed."""
    if mx is None:
        hint = (
            "MLX is not installed. Install it with `pip install mlx` on Apple "
            "Silicon devices. Original import error: "
        )
        raise MLXNotAvailableError(hint + str(_MLX_IMPORT_ERROR)) from _MLX_IMPORT_ERROR


def _tree_map(func: Callable[[Any], Any], tree: Any) -> Any:
    """Recursively apply func to leaves in nested structures."""
    if isinstance(tree, dict):
        return {key: _tree_map(func, value) for key, value in tree.items()}
    if isinstance(tree, (list, tuple)):
        mapped = [_tree_map(func, value) for value in tree]
        return type(tree)(mapped)
    return func(tree)


def _tree_leaves(tree: Any) -> Iterator[Any]:
    """Yield leaves from a nested structure."""
    if isinstance(tree, dict):
        for value in tree.values():
            yield from _tree_leaves(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from _tree_leaves(value)
    else:
        yield tree


def _to_host_array(value: Any) -> Any:
    """Convert MLX arrays to numpy arrays for serialization."""
    if value is None:
        return None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if mx is not None and isinstance(value, mx.array):
        if hasattr(mx, "to_numpy"):
            return mx.to_numpy(value)
        if hasattr(value, "to_numpy"):
            return value.to_numpy()
        if hasattr(value, "__array__"):
            return np.asarray(value)
        return np.array(value)
    if hasattr(value, "to_host"):
        return value.to_host()
    if hasattr(value, "to_numpy"):
        return value.to_numpy()
    if isinstance(value, types.GeneratorType):
        return [_to_host_array(item) for item in value]
    if isinstance(value, list):
        return [_to_host_array(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_host_array(item) for item in value)
    if isinstance(value, ABCIterable) and not isinstance(value, (str, bytes)):
        return [_to_host_array(item) for item in value]
    return value


def _to_mx_array(value: Any) -> Any:
    """Convert inputs from numpy/torch/python types into MLX arrays."""
    _ensure_mlx_available()

    if value is None:
        return None

    if isinstance(value, mx.array):
        return value
    if hasattr(value, "to_host"):
        return mx.array(value.to_host())
    if hasattr(value, "detach") and callable(value.detach):
        # Works for PyTorch tensors without importing torch explicitly
        if hasattr(value, "cpu"):
            tensor = value.detach().cpu().numpy()
        else:
            tensor = value.detach().numpy()
        return mx.array(tensor)
    if hasattr(value, "numpy"):
        return mx.array(value.numpy())
    if isinstance(value, np.ndarray):
        return mx.array(value)
    if isinstance(value, (list, tuple)):
        return mx.array(np.array(value))
    return mx.array(value)


def _ensure_nhwc_layout(array: Any) -> Any:
    """Convert channel-first batches to NHWC for MLX convolutions."""
    if mx is None or not isinstance(array, mx.array):
        return array
    if array.ndim == 4:
        channels_first = array.shape[1] <= 4 and array.shape[-1] > 4
        if channels_first:
            return mx.transpose(array, (0, 2, 3, 1))
    return array


def _resolve_device(device_hint: Optional[str]) -> Optional[Any]:
    """Resolve a device string from configuration into an MLX device."""
    if mx is None:
        return None

    if device_hint is None:
        return mx.default_device()

    hint = device_hint.lower()
    if hint.startswith("cuda") or hint == "mps" or hint.startswith("gpu"):
        try:
            return mx.gpu
        except AttributeError:
            return mx.default_device()

    return mx.cpu if hasattr(mx, "cpu") else mx.default_device()


@dataclass
class MLXTrainingContext:
    """
    Shared runtime context mirroring the PyTorch training context but for MLX.

    Attributes:
        model: The MLX nn.Module being optimized.
        device: The MLX device or identifier backing computations.
        client_id: ID of the current client (0 for server).
        current_epoch: 1-indexed epoch counter.
        current_round: Federated round counter.
        config: Training configuration dictionary.
        state: Mutable dictionary for strategy coordination.
    """

    model: Optional["mx_nn.Module"] = None
    device: Optional[Any] = None
    client_id: int = 0
    current_epoch: int = 0
    current_round: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"MLXTrainingContext(client_id={self.client_id}, "
            f"epoch={self.current_epoch}, round={self.current_round})"
        )


class MLXStrategy(ABC):
    """Base class for MLX strategy components."""

    def setup(self, context: MLXTrainingContext) -> None:
        """Optional initialization hook called during trainer setup."""

    def teardown(self, context: MLXTrainingContext) -> None:
        """Optional teardown hook executed when training is complete."""


class MLXLossCriterionStrategy(MLXStrategy):
    """Strategy interface for computing MLX-based loss."""

    @abstractmethod
    def compute_loss(
        self,
        outputs: "mx.array",
        labels: Optional["mx.array"],
        context: MLXTrainingContext,
    ) -> "mx.array":
        """Compute scalar loss tensor."""


class MLXOptimizerStrategy(MLXStrategy):
    """Strategy interface for creating MLX optimizers."""

    @abstractmethod
    def create_optimizer(
        self, model: "mx_nn.Module", context: MLXTrainingContext
    ) -> "mx_optim.Optimizer":
        """Instantiate an optimizer for the provided model."""

    def on_optimizer_step(
        self, optimizer: "mx_optim.Optimizer", context: MLXTrainingContext
    ) -> None:
        """Hook called after each optimizer step."""


class MLXTrainingStepStrategy(MLXStrategy):
    """Strategy interface controlling forward/backward passes."""

    @abstractmethod
    def training_step(
        self,
        model: "mx_nn.Module",
        optimizer: "mx_optim.Optimizer",
        examples: "mx.array",
        labels: Optional["mx.array"],
        loss_criterion: Callable[["mx.array", Optional["mx.array"]], "mx.array"],
        context: MLXTrainingContext,
    ) -> "mx.array":
        """Execute a single training step and return the loss."""

    def finalize(
        self,
        model: "mx_nn.Module",
        optimizer: "mx_optim.Optimizer",
        context: MLXTrainingContext,
    ) -> Optional["mx.array"]:
        """Optional hook to flush delayed optimizer updates."""
        return None


class MLXLRSchedulerStrategy(MLXStrategy):
    """Strategy interface for MLX learning rate schedulers."""

    def create_scheduler(
        self, optimizer: "mx_optim.Optimizer", context: MLXTrainingContext
    ) -> Optional[Any]:
        """Return an MLX-compatible LR scheduler if configured."""
        return None

    def step(self, scheduler: Optional[Any], context: MLXTrainingContext) -> None:
        """Advance the learning rate scheduler if one is active."""
        if scheduler is None:
            return
        if hasattr(scheduler, "step") and callable(scheduler.step):
            scheduler.step()
        elif callable(scheduler):
            scheduler()


class MLXModelUpdateStrategy(MLXStrategy):
    """Strategy for weight/state management around training."""

    def on_train_start(self, context: MLXTrainingContext) -> None:
        """Hook executed before the first batch of training."""

    def before_step(self, context: MLXTrainingContext) -> None:
        """Hook executed before each optimizer step."""

    def after_step(self, context: MLXTrainingContext) -> None:
        """Hook executed after each optimizer step."""

    def on_train_end(self, context: MLXTrainingContext) -> None:
        """Hook executed when training concludes."""

    def get_update_payload(self, context: MLXTrainingContext) -> Dict[str, Any]:
        """Return additional payload to attach to model updates."""
        return {}


class MLXDataLoaderStrategy(MLXStrategy):
    """Strategy for constructing data loaders without assuming PyTorch."""

    @abstractmethod
    def create_train_loader(
        self,
        trainset: Any,
        sampler: Any,
        batch_size: int,
        context: MLXTrainingContext,
    ) -> Iterable[Tuple[Any, Any]]:
        """Return an iterable over training batches."""

    def create_test_loader(
        self,
        testset: Any,
        sampler: Any,
        batch_size: int,
        context: MLXTrainingContext,
    ) -> Iterable[Tuple[Any, Any]]:
        """Return an iterable over evaluation batches."""
        return self.create_train_loader(testset, sampler, batch_size, context)


class MLXTestingStrategy(MLXStrategy):
    """Strategy interface for model evaluation."""

    @abstractmethod
    def test_model(
        self,
        model: "mx_nn.Module",
        config: Dict[str, Any],
        testset: Any,
        sampler: Any,
        context: MLXTrainingContext,
    ) -> float:
        """Run evaluation and return a scalar metric."""


class DefaultMLXLossStrategy(MLXLossCriterionStrategy):
    """Default cross-entropy loss reduced to a scalar."""

    def __init__(
        self,
        loss_fn: Optional[Callable[["mx.array", "mx.array"], "mx.array"]] = None,
        reduction: str = "mean",
    ):
        self.loss_fn = loss_fn
        self.reduction = reduction

    def setup(self, context: MLXTrainingContext) -> None:
        _ensure_mlx_available()
        if self.loss_fn is None:
            from mlx.nn import losses as mx_losses

            if hasattr(mx_losses, "cross_entropy"):
                self.loss_fn = mx_losses.cross_entropy
            elif hasattr(mx_losses, "softmax_cross_entropy"):
                self.loss_fn = mx_losses.softmax_cross_entropy
            else:
                raise MLXNotAvailableError(
                    "MLX installation does not provide a softmax/cross entropy loss."
                )

    def compute_loss(
        self,
        outputs: "mx.array",
        labels: Optional["mx.array"],
        context: MLXTrainingContext,
    ) -> "mx.array":
        _ensure_mlx_available()
        if labels is None:
            raise ValueError("Labels are required for the default MLX loss strategy.")

        losses = self.loss_fn(outputs, labels)
        if self.reduction == "mean":
            return mx.mean(losses)
        if self.reduction == "sum":
            return mx.sum(losses)
        return losses


class DefaultMLXOptimizerStrategy(MLXOptimizerStrategy):
    """Default optimizer strategy using MLX's optimizer registry."""

    _REGISTERED_OPTIMIZERS: Dict[str, Callable[..., "mx_optim.Optimizer"]] = {
        "adam": getattr(mx_optim, "Adam", None),
        "adamw": getattr(mx_optim, "AdamW", None),
        "sgd": getattr(mx_optim, "SGD", None),
        "momentum": getattr(mx_optim, "Momentum", None),
        "rmsprop": getattr(mx_optim, "RMSProp", None),
        "lion": getattr(mx_optim, "Lion", None),
    }

    def __init__(self, optimizer_name: Optional[str] = None, **optimizer_kwargs: Any):
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs

    def setup(self, context: MLXTrainingContext) -> None:
        if self.optimizer_name is None:
            self.optimizer_name = getattr(Config().trainer, "optimizer", "adam")

    def create_optimizer(
        self, model: "mx_nn.Module", context: MLXTrainingContext
    ) -> "mx_optim.Optimizer":
        _ensure_mlx_available()

        name = (self.optimizer_name or "adam").lower()
        if name not in self._REGISTERED_OPTIMIZERS:
            raise ValueError(f"Unsupported MLX optimizer '{self.optimizer_name}'.")

        optimizer_cls = self._REGISTERED_OPTIMIZERS[name]
        if optimizer_cls is None:
            raise MLXNotAvailableError(
                "MLX optimizer "
                f"'{self.optimizer_name}' is unavailable in this installation."
            )

        params = dict(self.optimizer_kwargs)
        if not params and hasattr(Config().parameters, "optimizer"):
            params = Config().parameters.optimizer._asdict()

        return optimizer_cls(**params)


class DefaultMLXTrainingStepStrategy(MLXTrainingStepStrategy):
    """Default SGD-style training step using value-and-grad."""

    def __init__(self, jit: bool = False, clip_grad_norm: Optional[float] = None):
        self.jit = jit
        self.clip_grad_norm = clip_grad_norm
        self._value_and_grad = None

    def setup(self, context: MLXTrainingContext) -> None:
        _ensure_mlx_available()
        self._value_and_grad = nn_utils.value_and_grad

    def training_step(
        self,
        model: "mx_nn.Module",
        optimizer: "mx_optim.Optimizer",
        examples: "mx.array",
        labels: Optional["mx.array"],
        loss_criterion: Callable[["mx.array", Optional["mx.array"]], "mx.array"],
        context: MLXTrainingContext,
    ) -> "mx.array":
        _ensure_mlx_available()

        if labels is None:

            def inner_loss(examples_inner):
                outputs = model(examples_inner)
                return loss_criterion(outputs, None)

            loss, grads = self._value_and_grad(model, inner_loss)(examples)
        else:

            def inner_loss(examples_inner, labels_inner):
                outputs = model(examples_inner)
                return loss_criterion(outputs, labels_inner)

            loss, grads = self._value_and_grad(model, inner_loss)(examples, labels)

        if self.clip_grad_norm is not None:
            logging.warning(
                "Gradient clipping is requested but not implemented for MLX yet."
            )
        optimizer.update(model, grads)
        if hasattr(mx, "eval"):
            state = getattr(optimizer, "state", None)
            try:
                if state is not None:
                    mx.eval(model.parameters(), state)
                else:
                    mx.eval(model.parameters())
            except TypeError:
                params_tuple = tuple(model.parameters())
                if state is not None:
                    mx.eval(params_tuple, state)
                else:
                    mx.eval(params_tuple)
        return loss


class DefaultMLXDataLoaderStrategy(MLXDataLoaderStrategy):
    """Simple numpy-backed data loader that avoids PyTorch dependencies."""

    def __init__(
        self,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
    ):
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers  # Reserved for future parallel loaders

    def _resolve_indices(self, dataset: Any, sampler: Any) -> Sequence[int]:
        if sampler is None:
            if hasattr(dataset, "__len__"):
                return list(range(len(dataset)))
            raise ValueError("Sampler is required when dataset length is undefined.")

        if isinstance(sampler, Sequence):
            return list(int(idx) for idx in sampler)
        if hasattr(sampler, "indices"):
            return list(int(idx) for idx in sampler.indices)
        if hasattr(sampler, "__iter__"):
            return [int(idx) for idx in sampler]
        if hasattr(sampler, "get"):
            resolved = sampler.get()
            return self._resolve_indices(dataset, resolved)
        raise TypeError(f"Unsupported sampler type: {type(sampler)}")

    def _make_loader(
        self,
        dataset: Any,
        indices: Sequence[int],
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
    ) -> "SimpleDataLoader":
        return SimpleDataLoader(
            dataset,
            indices,
            batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def create_train_loader(
        self,
        trainset: Any,
        sampler: Any,
        batch_size: int,
        context: MLXTrainingContext,
    ) -> Iterable[Tuple[Any, Any]]:
        indices = self._resolve_indices(trainset, sampler)
        shuffle = self.shuffle
        if sampler is not None and not isinstance(sampler, Sequence):
            shuffle = False
        return self._make_loader(trainset, indices, batch_size, shuffle, self.drop_last)

    def create_test_loader(
        self,
        testset: Any,
        sampler: Any,
        batch_size: int,
        context: MLXTrainingContext,
    ) -> Iterable[Tuple[Any, Any]]:
        if sampler is not None:
            indices = self._resolve_indices(testset, sampler)
        else:
            indices = list(range(len(testset)))
        return self._make_loader(
            testset,
            indices,
            batch_size,
            shuffle=False,
            drop_last=False,
        )


class DefaultMLXTestingStrategy(MLXTestingStrategy):
    """Compute classification accuracy using MLX ops."""

    def __init__(self, batch_size: Optional[int] = None):
        self.batch_size = batch_size
        self.data_loader_strategy = DefaultMLXDataLoaderStrategy(shuffle=False)

    def test_model(
        self,
        model: "mx_nn.Module",
        config: Dict[str, Any],
        testset: Any,
        sampler: Any,
        context: MLXTrainingContext,
    ) -> float:
        _ensure_mlx_available()

        batch_size = self.batch_size or config.get("batch_size", 32)
        loader = self.data_loader_strategy.create_test_loader(
            testset, sampler, batch_size, context
        )

        context.state.pop("eval_label_debug_logged", None)

        total_samples = 0
        correct_predictions = 0

        for examples, labels in loader:
            if labels is None:
                raise ValueError("Labels are required for evaluation.")

            examples = _tree_map(_to_mx_array, examples)
            examples = _tree_map(_ensure_nhwc_layout, examples)
            labels = _tree_map(_to_mx_array, labels)
            if logging.getLogger(__name__).isEnabledFor(
                logging.DEBUG
            ) and not context.state.get("eval_label_debug_logged", False):
                label_arr = (
                    labels.to_numpy()
                    if hasattr(labels, "to_numpy")
                    else labels.to_host()
                    if hasattr(labels, "to_host")
                    else np.asarray(labels)
                )
                logging.debug(
                    "[MLX Eval] Label dtype=%s shape=%s",
                    label_arr.dtype,
                    label_arr.shape,
                )
                context.state["eval_label_debug_logged"] = True

            logits = model(examples)
            predicted = mx.argmax(logits, axis=-1)
            matches = predicted == labels
            correct_predictions += int(mx.sum(matches).item())
            if hasattr(labels, "shape"):
                total_samples += int(labels.shape[0])
            else:
                total_samples += len(labels)

        if total_samples == 0:
            return 0.0

        return correct_predictions / total_samples


class SimpleDataLoader(Iterable[Tuple[Any, Any]]):
    """Minimal iterable data loader supporting indexable datasets."""

    def __init__(
        self,
        dataset: Any,
        indices: Sequence[int],
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.indices = list(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        indices = list(self.indices)
        if self.shuffle:
            random.shuffle(indices)

        batch_examples: List[Any] = []
        batch_labels: List[Any] = []

        for idx in indices:
            item = self.dataset[idx]
            if isinstance(item, tuple):
                if len(item) == 1:
                    example, label = item[0], None
                else:
                    example, label = item[0], item[1]
            else:
                example, label = item, None

            if torch is not None and isinstance(label, torch.Tensor):
                if label.ndim == 0:
                    label = int(label.item())
                else:
                    label = label.detach().cpu().numpy()

            batch_examples.append(example)
            batch_labels.append(label)

            if len(batch_examples) == self.batch_size:
                yield self._stack_batch(batch_examples), self._stack_batch(batch_labels)
                batch_examples = []
                batch_labels = []

        if batch_examples and not self.drop_last:
            yield self._stack_batch(batch_examples), self._stack_batch(batch_labels)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.indices) // self.batch_size
        return int(np.ceil(len(self.indices) / self.batch_size))

    @staticmethod
    def _stack_batch(items: Sequence[Any]) -> Any:
        if not items:
            return None
        if items[0] is None:
            return None
        if mx is not None and isinstance(items[0], mx.array):
            return mx.stack(items, axis=0)
        if torch is not None and isinstance(items[0], torch.Tensor):
            stacked = torch.stack(items)
            return stacked.detach().cpu().numpy()
        if hasattr(items[0], "stack") and callable(getattr(items[0], "stack")):
            return items[0].stack(items, axis=0)
        if isinstance(items[0], np.ndarray):
            return np.stack(items, axis=0)
        return np.array(items)


class ComposableMLXTrainer(base.Trainer):
    """Composable trainer implementation backed by MLX runtime primitives."""

    def __init__(
        self,
        model: Optional[Union["mx_nn.Module", Callable[[], "mx_nn.Module"]]] = None,
        callbacks: Optional[List[Any]] = None,
        loss_strategy: Optional[MLXLossCriterionStrategy] = None,
        optimizer_strategy: Optional[MLXOptimizerStrategy] = None,
        training_step_strategy: Optional[MLXTrainingStepStrategy] = None,
        lr_scheduler_strategy: Optional[MLXLRSchedulerStrategy] = None,
        model_update_strategy: Optional[MLXModelUpdateStrategy] = None,
        data_loader_strategy: Optional[MLXDataLoaderStrategy] = None,
        testing_strategy: Optional[MLXTestingStrategy] = None,
    ):
        _ensure_mlx_available()

        super().__init__()

        self.context = MLXTrainingContext()
        self.context.device = _resolve_device(Config().device())
        self.context.client_id = self.client_id

        if model is None:
            self.model = models_registry.get()
        elif isinstance(model, mx_nn.Module):
            self.model = model
        elif callable(model):
            self.model = model()
        else:
            raise TypeError("MLX trainers require an nn.Module or factory callable.")

        self.context.model = self.model

        self.loss_strategy = loss_strategy or DefaultMLXLossStrategy()
        self.optimizer_strategy = optimizer_strategy or DefaultMLXOptimizerStrategy()
        self.training_step_strategy = (
            training_step_strategy or DefaultMLXTrainingStepStrategy()
        )
        self.lr_scheduler_strategy = lr_scheduler_strategy or MLXLRSchedulerStrategy()
        self.model_update_strategy = model_update_strategy or MLXModelUpdateStrategy()
        self.data_loader_strategy = (
            data_loader_strategy or DefaultMLXDataLoaderStrategy()
        )
        self.testing_strategy = testing_strategy or DefaultMLXTestingStrategy()

        self._setup_strategies()

        self.callbacks = [LogProgressCallback]
        if callbacks is not None:
            self.callbacks.extend(callbacks)
        self.callback_handler = CallbackHandler(self.callbacks)

        self.run_history = tracking.RunHistory()
        self._loss_tracker = tracking.LossTracker()

        self.trainset = None
        self.train_loader = None
        self.sampler = None
        self.optimizer = None
        self.lr_scheduler = None
        self.current_epoch = 0
        self.current_round = 0
        self.training_start_time = time.time()
        self.model_state_tree = None

    def _setup_strategies(self) -> None:
        for strategy in [
            self.loss_strategy,
            self.optimizer_strategy,
            self.training_step_strategy,
            self.lr_scheduler_strategy,
            self.model_update_strategy,
            self.data_loader_strategy,
            self.testing_strategy,
        ]:
            if strategy is not None:
                strategy.setup(self.context)

    def _teardown_strategies(self) -> None:
        for strategy in [
            self.loss_strategy,
            self.optimizer_strategy,
            self.training_step_strategy,
            self.lr_scheduler_strategy,
            self.model_update_strategy,
            self.data_loader_strategy,
            self.testing_strategy,
        ]:
            if strategy is not None:
                strategy.teardown(self.context)

    def set_client_id(self, client_id: int) -> None:
        super().set_client_id(client_id)
        self.context.client_id = client_id

    def zeros(self, shape: Union[int, Sequence[int]]) -> "mx.array":
        _ensure_mlx_available()
        assert self.client_id == 0
        return mx.zeros(shape)

    # ---------------------------------------------------------------------
    # Model persistence
    # ---------------------------------------------------------------------

    def save_model(
        self,
        filename: Optional[str] = None,
        location: Optional[str] = None,
    ) -> None:
        model_path = Config().params["model_path"] if location is None else location
        model_name = Config().trainer.model_name

        os.makedirs(model_path, exist_ok=True)

        if filename is not None:
            model_path = f"{model_path}/{filename}"
        else:
            model_path = f"{model_path}/{model_name}.safetensors"

        state_tree = self._capture_model_state()
        serialized = serialize_tree(state_tree)
        with open(model_path, "wb") as model_file:
            model_file.write(serialized)

        with open(model_path + ".pkl", "wb") as history_file:
            pickle.dump(self.run_history, history_file)

        identity = "Server" if self.client_id == 0 else f"Client #{self.client_id}"
        logging.info("[%s] MLX model saved to %s.", identity, model_path)

    def load_model(
        self,
        filename: Optional[str] = None,
        location: Optional[str] = None,
    ) -> None:
        model_path = Config().params["model_path"] if location is None else location
        model_name = Config().trainer.model_name

        if filename is not None:
            model_path = f"{model_path}/{filename}"
        else:
            model_path = f"{model_path}/{model_name}.safetensors"

        if not os.path.exists(model_path):
            raise OSError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as model_file:
            serialized = model_file.read()

        state_tree = deserialize_tree(serialized)

        self._apply_model_state(state_tree)

        history_path = model_path + ".pkl"
        if os.path.exists(history_path):
            with open(history_path, "rb") as history_file:
                self.run_history = pickle.load(history_file)

        identity = "Server" if self.client_id == 0 else f"Client #{self.client_id}"
        logging.info("[%s] MLX model loaded from %s.", identity, model_path)

    def simulate_sleep_time(self) -> None:
        """Simulate slower clients by pausing execution."""
        if (
            hasattr(Config().clients, "sleep_simulation")
            and Config().clients.sleep_simulation
        ):
            sleep_seconds = Config().client_sleep_times[self.client_id - 1]
            sleep_seconds = max(0, sleep_seconds)

            if sleep_seconds > 0:
                logging.info(
                    "[Client #%d] Simulating stragglers by sleeping for %.2f seconds.",
                    self.client_id,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

    def _capture_model_state(self) -> Any:
        parameters = self.model.parameters()
        return _tree_map(_to_host_array, parameters)

    def _apply_model_state(self, state_tree: Any) -> None:
        restored = _tree_map(_to_mx_array, state_tree)
        if hasattr(self.model, "update"):
            self.model.update(restored)
        else:
            raise RuntimeError(
                "The configured MLX model does not support parameter updates."
            )
        if mx is not None:
            leaves = [
                leaf
                for leaf in _tree_leaves(self.model.parameters())
                if isinstance(leaf, mx.array)
            ]
            if leaves:
                mx.eval(*leaves)

    # ---------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------

    def train_model(self, config, trainset, sampler, **kwargs):
        batch_size = config["batch_size"]
        self.trainset = trainset
        self.sampler = sampler
        self.context.config = config
        self.context.current_round = self.current_round

        if trainset is None:
            logging.warning(
                "[Client #%d] No training dataset provided; reloading via registry.",
                self.client_id,
            )
            datasource = datasources_registry.get(client_id=self.client_id)
            trainset = datasource.get_train_set()
            self.trainset = trainset

        self.run_history.reset()
        self._loss_tracker.reset()

        self.callback_handler.call_event("on_train_run_start", self, config)
        self.model_update_strategy.on_train_start(self.context)

        self.train_loader = self.data_loader_strategy.create_train_loader(
            trainset, sampler, batch_size, self.context
        )
        self.context.state["train_loader"] = self.train_loader

        sampled_size = self._infer_sampled_size(trainset, sampler)
        self.context.state["num_samples"] = sampled_size

        self.context.state.pop("train_label_debug_logged", None)

        self.optimizer = self.optimizer_strategy.create_optimizer(
            self.model,
            self.context,
        )
        self.lr_scheduler = self.lr_scheduler_strategy.create_scheduler(
            self.optimizer,
            self.context,
        )

        total_epochs = config["epochs"]
        tic = time.perf_counter()
        training_stop_requested = False

        for self.current_epoch in range(1, total_epochs + 1):
            self.context.current_epoch = self.current_epoch
            self._loss_tracker.reset()
            self.context.state["grad_accum_counter"] = 0

            self.callback_handler.call_event("on_train_epoch_start", self, config)
            batches_seen = False
            last_batch_id = -1

            for batch_id, (examples, labels) in enumerate(self.train_loader):
                self.context.state["current_batch"] = batch_id
                batches_seen = True
                last_batch_id = batch_id

                self.callback_handler.call_event(
                    "on_train_step_start",
                    self,
                    config,
                    batch=batch_id,
                )
                self.model_update_strategy.before_step(self.context)

                examples = _tree_map(_to_mx_array, examples)
                examples = _tree_map(_ensure_nhwc_layout, examples)
                labels = _tree_map(_to_mx_array, labels)
                if logging.getLogger(__name__).isEnabledFor(
                    logging.DEBUG
                ) and not self.context.state.get("train_label_debug_logged", False):
                    label_arr = (
                        labels.to_numpy()
                        if hasattr(labels, "to_numpy")
                        else labels.to_host()
                        if hasattr(labels, "to_host")
                        else np.asarray(labels)
                    )
                    logging.debug(
                        "[MLX Train] Label dtype=%s shape=%s",
                        label_arr.dtype,
                        label_arr.shape,
                    )
                    self.context.state["train_label_debug_logged"] = True

                def compute_loss_fn(outputs, labels_inner):
                    return self.loss_strategy.compute_loss(
                        outputs,
                        labels_inner,
                        self.context,
                    )

                loss = self.training_step_strategy.training_step(
                    model=self.model,
                    optimizer=self.optimizer,
                    examples=examples,
                    labels=labels,
                    loss_criterion=compute_loss_fn,
                    context=self.context,
                )

                loss_value = float(loss.item() if hasattr(loss, "item") else loss)
                batch_size_effective = self._batch_size(labels, examples)
                self._loss_tracker.update(loss_value, batch_size_effective)
                self.context.state["last_loss"] = loss_value

                self.optimizer_strategy.on_optimizer_step(self.optimizer, self.context)
                self.model_update_strategy.after_step(self.context)

                self.callback_handler.call_event(
                    "on_train_step_end", self, config, batch=batch_id, loss=loss
                )

                if self.context.state.pop("stop_training", False):
                    training_stop_requested = True
                    break

            finalize_loss = None
            finalize_callable = getattr(self.training_step_strategy, "finalize", None)
            if batches_seen and callable(finalize_callable):
                finalize_loss = finalize_callable(
                    model=self.model,
                    optimizer=self.optimizer,
                    context=self.context,
                )
                if finalize_loss is not None:
                    self.optimizer_strategy.on_optimizer_step(
                        self.optimizer,
                        self.context,
                    )
                    self.model_update_strategy.after_step(self.context)
                    self.callback_handler.call_event(
                        "on_train_step_end",
                        self,
                        config,
                        batch=last_batch_id,
                        loss=finalize_loss,
                    )
                    self.context.state["last_loss"] = float(
                        finalize_loss.item()
                        if hasattr(finalize_loss, "item")
                        else finalize_loss
                    )

            self.lr_scheduler_strategy.step(self.lr_scheduler, self.context)

            if (
                self.client_id != 0
                and hasattr(Config().clients, "speed_simulation")
                and Config().clients.speed_simulation
            ):
                self.simulate_sleep_time()

            if (
                hasattr(Config().server, "request_update")
                and Config().server.request_update
            ):
                epoch_time = time.perf_counter() - tic
                filename = (
                    f"{self.client_id}_{self.current_epoch}_{epoch_time}.safetensors"
                )
                self.save_model(filename)

            self.run_history.update_metric("train_loss", self._loss_tracker.average)

            self.callback_handler.call_event("on_train_epoch_end", self, config)

            if training_stop_requested:
                break

        toc = time.perf_counter()
        training_time = toc - tic

        self.model_update_strategy.on_train_end(self.context)

        self.callback_handler.call_event(
            "on_train_run_end", self, config, training_time=training_time
        )

        self.run_history.update_metric("train_time", training_time)

    def _infer_sampled_size(self, dataset: Any, sampler: Any) -> int:
        if sampler is not None:
            if hasattr(sampler, "num_samples") and callable(sampler.num_samples):
                try:
                    return int(sampler.num_samples())
                except TypeError:
                    pass
            if isinstance(sampler, Sequence):
                return len(sampler)
        if hasattr(dataset, "__len__"):
            return len(dataset)
        return 0

    def test(self, testset, sampler=None, **kwargs) -> float:
        config = Config().trainer._asdict()
        config["run_id"] = Config().params["run_id"]

        accuracy = self.test_model(config, testset, sampler, **kwargs)

        model_name = Config().trainer.model_name
        filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
        self.save_accuracy(accuracy, filename)
        return accuracy

    def test_model(self, config, testset, sampler=None, **kwargs):
        accuracy = self.testing_strategy.test_model(
            self.model, config, testset, sampler, self.context
        )
        self.accuracy = accuracy
        return accuracy

    def obtain_model_update(self, config, trainset, sampler):
        self.train_model(config, trainset, sampler)
        model_update = self._capture_model_state()
        additional_payload = self.model_update_strategy.get_update_payload(self.context)
        if additional_payload:
            return {"model_update": model_update, **additional_payload}
        return model_update

    def pause_training(self):
        """Remove temporary MLX artifacts created during concurrent execution."""
        if hasattr(Config().trainer, "max_concurrency"):
            model_name = Config().trainer.model_name
            model_path = Config().params["model_path"]
            base_filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}"
            model_file = f"{model_path}/{base_filename}.safetensors"
            history_file = f"{model_file}.pkl"
            accuracy_file = f"{model_path}/{base_filename}.acc"

            if os.path.exists(model_file):
                os.remove(model_file)
            if os.path.exists(history_file):
                os.remove(history_file)
            if os.path.exists(accuracy_file):
                os.remove(accuracy_file)

    def obtain_model_at_time(self, client_id, requested_time):
        raise NotImplementedError(
            "Wall-clock model retrieval is not yet implemented for MLX."
        )

    def train(self, trainset, sampler, **kwargs) -> float:
        config = Config().trainer._asdict()
        config["run_id"] = Config().params["run_id"]

        self.training_start_time = time.time()

        tic = time.perf_counter()
        self.train_model(config, trainset, sampler, **kwargs)
        toc = time.perf_counter()

        model_name = Config().trainer.model_name
        filename = f"{model_name}_{self.client_id}_{config['run_id']}.safetensors"
        self.save_model(filename)

        training_time = toc - tic
        return training_time

    def _batch_size(self, labels: Any, examples: Any) -> int:
        if labels is not None:
            if hasattr(labels, "shape"):
                return int(labels.shape[0])
            if isinstance(labels, (list, tuple)):
                return len(labels)
        if hasattr(examples, "shape"):
            return int(examples.shape[0])
        if isinstance(examples, (list, tuple)):
            return len(examples)
        return 1
