"""
DiLoCo aggregation strategy.

The strategy consumes Plato-style client deltas (`client_after - global_before`),
converts them to DiLoCo outer gradients, and returns Plato-compatible server
deltas for `algorithm.update_weights()` to add to the global model.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import numbers
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from plato.servers.strategies.aggregation.fedavg import FedAvgAggregationStrategy
from plato.servers.strategies.base import ServerContext

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = cast(Any, None)


class DiLoCoAggregationStrategy(FedAvgAggregationStrategy):
    """Aggregate client deltas with a server-side DiLoCo outer optimizer."""

    _SUPPORTED_OPTIMIZERS = {"sgd", "sgdm", "nesterov"}
    _SUPPORTED_WEIGHTING_MODES = {"uniform", "num_samples"}
    _SUPPORTED_APPLY_POLICIES = {"parameters", "all_floating"}

    def __init__(
        self,
        outer_optimizer: str = "nesterov",
        outer_learning_rate: float = 0.7,
        outer_momentum: float = 0.9,
        aggregation_weighting: str = "uniform",
        apply_outer_optimizer_to: str = "parameters",
    ):
        super().__init__()
        self.outer_optimizer = self._validate_outer_optimizer(outer_optimizer)
        self.outer_learning_rate = self._validate_learning_rate(
            outer_learning_rate
        )
        self.outer_momentum = self._validate_momentum(outer_momentum)
        self.aggregation_weighting = self._validate_weighting_mode(
            aggregation_weighting
        )
        self.apply_outer_optimizer_to = self._validate_apply_policy(
            apply_outer_optimizer_to
        )
        self.momentum_state: dict[str, Any] = {}

    async def aggregate_deltas(
        self,
        updates: list[SimpleNamespace],
        deltas_received: list[dict],
        context: ServerContext,
    ) -> dict:
        """Aggregate deltas and apply the configured DiLoCo outer optimizer."""
        eligible = self._eligible_updates(updates, deltas_received)
        if not eligible:
            self._remove_stale_momentum(set())
            return self._empty_delta(context, self._first_delta(deltas_received))

        weights = self._aggregation_weights(eligible)
        if not weights:
            self._remove_stale_momentum(set())
            return self._empty_delta(context, eligible[0][1])

        avg_delta: Any = None
        for (_, delta, _), weight in zip(eligible, weights):
            avg_delta = self._accumulate_weighted(avg_delta, delta, weight, context)
            await asyncio.sleep(0)

        if avg_delta is None:
            self._remove_stale_momentum(set())
            return self._empty_delta(context, eligible[0][1])

        avg_delta = self._match_reference_structure(avg_delta, eligible[0][1])
        optimizer_paths = self._outer_optimizer_paths(avg_delta, context)
        server_delta, active_paths = self._apply_outer_optimizer(
            avg_delta, optimizer_paths
        )
        logging.info(
            "[Server] DiLoCo outer optimizer applied: optimizer=%s "
            "outer_lr=%g outer_momentum=%g weighting=%s apply_to=%s "
            "eligible_updates=%d optimized_tensors=%d.",
            self.outer_optimizer,
            self.outer_learning_rate,
            self.outer_momentum,
            self.aggregation_weighting,
            self.apply_outer_optimizer_to,
            len(eligible),
            len(optimizer_paths),
        )
        self._remove_stale_momentum(active_paths)

        return self._match_reference_structure(server_delta, eligible[0][1])

    @classmethod
    def _validate_outer_optimizer(cls, value: str) -> str:
        optimizer = str(value).lower()
        if optimizer not in cls._SUPPORTED_OPTIMIZERS:
            supported = ", ".join(sorted(cls._SUPPORTED_OPTIMIZERS))
            raise ValueError(
                f"Invalid outer_optimizer '{value}'. Supported values: {supported}."
            )
        return optimizer

    @staticmethod
    def _validate_learning_rate(value: float) -> float:
        learning_rate = float(value)
        if learning_rate < 0:
            raise ValueError("outer_learning_rate must be nonnegative.")
        return learning_rate

    @staticmethod
    def _validate_momentum(value: float) -> float:
        momentum = float(value)
        if not 0 <= momentum < 1:
            raise ValueError("outer_momentum must be in the range [0, 1).")
        return momentum

    @classmethod
    def _validate_weighting_mode(cls, value: str) -> str:
        weighting = str(value).lower()
        if weighting not in cls._SUPPORTED_WEIGHTING_MODES:
            supported = ", ".join(sorted(cls._SUPPORTED_WEIGHTING_MODES))
            raise ValueError(
                "Invalid aggregation_weighting "
                f"'{value}'. Supported values: {supported}."
            )
        return weighting

    @classmethod
    def _validate_apply_policy(cls, value: str) -> str:
        policy = str(value).lower()
        if policy not in cls._SUPPORTED_APPLY_POLICIES:
            supported = ", ".join(sorted(cls._SUPPORTED_APPLY_POLICIES))
            raise ValueError(
                "Invalid apply_outer_optimizer_to "
                f"'{value}'. Supported values: {supported}."
            )
        return policy

    def _eligible_updates(
        self,
        updates: list[SimpleNamespace],
        deltas_received: list[dict],
    ) -> list[tuple[SimpleNamespace, dict, float]]:
        eligible: list[tuple[SimpleNamespace, dict, float]] = []
        for update, delta in zip(updates, deltas_received):
            if getattr(update.report, "type", "weights") == "features":
                continue

            num_samples = self._num_samples(update)
            if num_samples <= 0:
                continue

            eligible.append((update, delta, num_samples))

        return eligible

    @staticmethod
    def _num_samples(update: SimpleNamespace) -> float:
        try:
            return float(update.report.num_samples)
        except (AttributeError, TypeError, ValueError):
            return 0.0

    def _aggregation_weights(
        self, eligible: list[tuple[SimpleNamespace, dict, float]]
    ) -> list[float]:
        if not eligible:
            return []

        if self.aggregation_weighting == "uniform":
            return [1.0 / len(eligible)] * len(eligible)

        total_samples = sum(num_samples for _, _, num_samples in eligible)
        if total_samples <= 0:
            return []

        return [num_samples / total_samples for _, _, num_samples in eligible]

    def _outer_optimizer_paths(
        self, avg_delta: Any, context: ServerContext
    ) -> set[str]:
        if self.apply_outer_optimizer_to == "all_floating":
            return self._floating_leaf_paths(avg_delta)

        floating_paths = self._floating_leaf_paths(avg_delta)
        trainable_parameter_names = self._trainable_parameter_names(
            context, floating_paths
        )
        return floating_paths.intersection(trainable_parameter_names)

    def _apply_outer_optimizer(
        self, avg_delta: Any, optimizer_paths: set[str]
    ) -> tuple[Any, set[str]]:
        active_paths: set[str] = set()

        server_delta = self._map_tree(
            avg_delta,
            lambda value, path: self._apply_outer_optimizer_leaf(
                value, path, optimizer_paths, active_paths
            ),
        )
        return server_delta, active_paths

    def _apply_outer_optimizer_leaf(
        self,
        avg_delta: Any,
        path: str,
        optimizer_paths: set[str],
        active_paths: set[str],
    ) -> Any:
        if path not in optimizer_paths:
            return avg_delta

        outer_gradient = self._scale_tree(avg_delta, -1.0)
        if self.outer_optimizer == "sgd":
            return self._scale_tree(outer_gradient, -self.outer_learning_rate)

        return self._apply_momentum_leaf(outer_gradient, path, active_paths)

    def _apply_momentum_leaf(
        self, outer_gradient: Any, path: str, active_paths: set[str]
    ) -> Any:
        active_paths.add(path)
        previous = self.momentum_state.get(path)
        if previous is not None and not self._is_compatible(previous, outer_gradient):
            previous = None

        if previous is None:
            momentum = self._clone_tree(outer_gradient)
        else:
            momentum = self._add_values(
                self._scale_tree(previous, self.outer_momentum),
                outer_gradient,
            )

        self.momentum_state[path] = self._clone_tree(momentum)

        if self.outer_optimizer == "nesterov":
            direction = self._add_values(
                outer_gradient,
                self._scale_tree(momentum, self.outer_momentum),
            )
        else:
            direction = momentum

        return self._scale_tree(direction, -self.outer_learning_rate)

    def _remove_stale_momentum(self, active_paths: set[str]) -> None:
        if self.outer_optimizer == "sgd":
            self.momentum_state.clear()
            return

        for path in list(self.momentum_state):
            if path not in active_paths:
                del self.momentum_state[path]

    def _trainable_parameter_names(
        self, context: ServerContext, payload_paths: set[str] | None = None
    ) -> set[str]:
        model = self._model_from_context(context)
        adapter_names = self._adapter_names(model)
        trainable_names: set[str] = set()

        for name, parameter in model.named_parameters():
            if getattr(parameter, "requires_grad", False) and self._is_floating_value(
                parameter
            ):
                trainable_names.update(
                    self._payload_name_candidates(name, adapter_names, payload_paths)
                )

        return trainable_names

    @staticmethod
    def _adapter_names(model: Any) -> set[str]:
        adapter_names = {"default"}

        peft_config = getattr(model, "peft_config", None)
        if isinstance(peft_config, Mapping):
            adapter_names.update(str(name) for name in peft_config)

        active_adapter = getattr(model, "active_adapter", None)
        if isinstance(active_adapter, str):
            adapter_names.add(active_adapter)

        active_adapters = getattr(model, "active_adapters", None)
        if callable(active_adapters):
            try:
                adapter_names.update(str(name) for name in active_adapters())
            except TypeError:
                pass
        elif isinstance(active_adapters, (list, tuple, set)):
            adapter_names.update(str(name) for name in active_adapters)

        return adapter_names

    @classmethod
    def _payload_name_candidates(
        cls,
        parameter_name: str,
        adapter_names: set[str],
        payload_paths: set[str] | None,
    ) -> set[str]:
        candidates = {parameter_name}
        if payload_paths is not None and parameter_name in payload_paths:
            return candidates

        parts = parameter_name.split(".")
        for index, part in enumerate(parts):
            if part not in adapter_names:
                continue

            candidate = ".".join(parts[:index] + parts[index + 1 :])
            if payload_paths is None or candidate in payload_paths:
                candidates.add(candidate)

        return candidates

    @staticmethod
    def _model_from_context(context: ServerContext) -> Any:
        trainer = getattr(context, "trainer", None)
        model = getattr(trainer, "model", None) if trainer is not None else None
        if model is None or not hasattr(model, "named_parameters"):
            raise AttributeError(
                "DiLoCo apply_outer_optimizer_to='parameters' requires "
                "context.trainer.model with named_parameters()."
            )
        return model

    def _floating_leaf_paths(self, value: Any) -> set[str]:
        return self._collect_leaf_paths(
            value, lambda leaf, _: self._is_floating_value(leaf)
        )

    def _collect_leaf_paths(
        self,
        value: Any,
        predicate: Callable[[Any, str], bool],
        path: str = "",
    ) -> set[str]:
        if isinstance(value, Mapping):
            paths: set[str] = set()
            for key, item in value.items():
                paths.update(
                    self._collect_leaf_paths(
                        item, predicate, self._join_path(path, key)
                    )
                )
            return paths

        if isinstance(value, list):
            paths = set()
            for index, item in enumerate(value):
                paths.update(
                    self._collect_leaf_paths(
                        item, predicate, self._join_path(path, index)
                    )
                )
            return paths

        if isinstance(value, tuple):
            paths = set()
            for index, item in enumerate(value):
                paths.update(
                    self._collect_leaf_paths(
                        item, predicate, self._join_path(path, index)
                    )
                )
            return paths

        return {path} if predicate(value, path) else set()

    @staticmethod
    def _is_floating_value(value: Any) -> bool:
        if torch is not None and isinstance(value, torch.Tensor):
            return torch.is_floating_point(value)

        if isinstance(value, np.ndarray):
            return np.issubdtype(value.dtype, np.floating)

        return isinstance(value, numbers.Real) and not isinstance(
            value, (numbers.Integral, bool)
        )

    def _empty_delta(self, context: ServerContext, reference_delta: Any | None) -> dict:
        zero_delta = self._zero_delta(context, reference_delta)
        if zero_delta is not None:
            return zero_delta

        if reference_delta is None:
            return {}

        return self._scale_tree(reference_delta, 0.0)

    @staticmethod
    def _first_delta(deltas_received: list[dict]) -> dict | None:
        return deltas_received[0] if deltas_received else None

    def _map_tree(self, value: Any, leaf_fn: Callable[[Any, str], Any], path="") -> Any:
        if isinstance(value, Mapping):
            return {
                key: self._map_tree(item, leaf_fn, self._join_path(path, key))
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                self._map_tree(item, leaf_fn, self._join_path(path, index))
                for index, item in enumerate(value)
            ]

        if isinstance(value, tuple):
            return tuple(
                self._map_tree(item, leaf_fn, self._join_path(path, index))
                for index, item in enumerate(value)
            )

        return leaf_fn(value, path)

    def _scale_tree(self, value: Any, scalar: float) -> Any:
        if isinstance(value, Mapping):
            return {
                key: self._scale_tree(item, scalar) for key, item in value.items()
            }

        if isinstance(value, list):
            return [self._scale_tree(item, scalar) for item in value]

        if isinstance(value, tuple):
            return tuple(self._scale_tree(item, scalar) for item in value)

        return value * scalar

    @staticmethod
    def _add_values(left: Any, right: Any) -> Any:
        return left + right

    def _clone_tree(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: self._clone_tree(item) for key, item in value.items()}

        if isinstance(value, list):
            return [self._clone_tree(item) for item in value]

        if isinstance(value, tuple):
            return tuple(self._clone_tree(item) for item in value)

        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().clone()

        if isinstance(value, np.ndarray):
            return value.copy()

        try:
            return copy.deepcopy(value)
        except TypeError:
            return value

    @staticmethod
    def _is_compatible(left: Any, right: Any) -> bool:
        if torch is not None and isinstance(left, torch.Tensor):
            return (
                isinstance(right, torch.Tensor)
                and left.shape == right.shape
                and left.dtype == right.dtype
            )

        if isinstance(left, np.ndarray):
            return (
                isinstance(right, np.ndarray)
                and left.shape == right.shape
                and left.dtype == right.dtype
            )

        left_shape = getattr(left, "shape", None)
        right_shape = getattr(right, "shape", None)
        if left_shape is not None or right_shape is not None:
            return (
                left_shape == right_shape
                and getattr(left, "dtype", None) == getattr(right, "dtype", None)
            )

        return isinstance(left, numbers.Number) and isinstance(right, numbers.Number)

    @staticmethod
    def _join_path(prefix: str, key: Any) -> str:
        key_text = str(key)
        return key_text if not prefix else f"{prefix}.{key_text}"
