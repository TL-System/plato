"""
FedAvg aggregation strategy.

Implements the standard weighted averaging used by most federated learning
algorithms.
"""

from __future__ import annotations

import asyncio
import copy
import numbers
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, cast

import numpy as np

from plato.servers.strategies.base import AggregationStrategy, ServerContext

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = cast(Any, None)


class FedAvgAggregationStrategy(AggregationStrategy):
    """
    Standard Federated Averaging aggregation.

    Performs weighted averaging of client deltas based on the number of samples
    each client trained on.
    """

    async def aggregate_deltas(
        self,
        updates: list[SimpleNamespace],
        deltas_received: list[dict],
        context: ServerContext,
    ) -> dict:
        """Aggregate using weighted average by sample count."""
        eligible = [
            (update, deltas_received[idx])
            for idx, update in enumerate(updates)
            if getattr(update.report, "type", "weights") != "features"
        ]
        if not eligible:
            zero_delta = self._zero_delta(context)
            if zero_delta is not None:
                return zero_delta
            return {}

        total_samples = sum(update.report.num_samples for update, _ in eligible)
        if total_samples == 0:
            zero_delta = self._zero_delta(context, eligible[0][1])
            if zero_delta is not None:
                return zero_delta
            return {}

        reference_update = eligible[0][1]
        avg_update: Any = None
        for update, delta in eligible:
            num_samples = update.report.num_samples
            weight = num_samples / total_samples if total_samples > 0 else 0.0

            avg_update = self._accumulate_weighted(avg_update, delta, weight, context)

            await asyncio.sleep(0)

        if avg_update is None:
            zero_delta = self._zero_delta(context, reference_update)
            if zero_delta is not None:
                return zero_delta
            return {}

        return self._match_reference_structure(avg_update, reference_update)

    async def aggregate_weights(
        self,
        updates: list[SimpleNamespace],
        baseline_weights: dict,
        weights_received: list[dict],
        context: ServerContext,
    ) -> dict | None:
        """Aggregate weights directly when possible."""
        if not weights_received:
            return copy.deepcopy(baseline_weights)

        eligible = [
            (update, weights_received[idx])
            for idx, update in enumerate(updates)
            if getattr(update.report, "type", "weights") != "features"
        ]
        if not eligible:
            return None

        total_samples = sum(update.report.num_samples for update, _ in eligible)
        if total_samples == 0:
            return None

        avg_weights: Any = None
        for update, weights in eligible:
            num_samples = update.report.num_samples
            weight = num_samples / total_samples if total_samples > 0 else 0.0
            avg_weights = self._accumulate_weighted(
                avg_weights, weights, weight, context
            )
            await asyncio.sleep(0)

        if avg_weights is None:
            return None

        return self._match_reference_structure(avg_weights, baseline_weights)

    def _zero_delta(
        self, context: ServerContext, reference_update: Any | None = None
    ) -> dict | None:
        """Construct a zero delta matching the global model structure."""
        algorithm = getattr(context, "algorithm", None)
        if algorithm is None or not hasattr(algorithm, "extract_weights"):
            return None

        baseline_weights = algorithm.extract_weights()
        zero_deltas = algorithm.compute_weight_deltas(
            baseline_weights, [baseline_weights]
        )
        if not zero_deltas:
            return None

        zero_delta = zero_deltas[0]
        if reference_update is not None:
            return self._match_reference_structure(zero_delta, reference_update)
        return zero_delta

    def _accumulate_weighted(
        self,
        target: Any,
        value: Any,
        weight: float,
        context: ServerContext,
    ) -> Any:
        """Accumulate weighted values into the target structure and return it."""
        trainer = getattr(context, "trainer", None)
        zeros_fn: Callable[[Any], Any] | None = (
            cast(Callable[[Any], Any], trainer.zeros)
            if trainer is not None and hasattr(trainer, "zeros")
            else None
        )

        if value is None or weight == 0.0:
            return target

        if isinstance(value, dict):
            base = target if isinstance(target, dict) and target is not None else {}
            for key, item in value.items():
                base[key] = self._accumulate_weighted(
                    base.get(key), item, weight, context
                )
            return base

        if isinstance(value, (list, tuple)):
            is_tuple = isinstance(value, tuple)
            length = len(value)
            if (
                target is not None
                and isinstance(target, (list, tuple))
                and len(target) == length
            ):
                base_seq = list(target)
            else:
                base_seq = [None] * length
            for idx, item in enumerate(value):
                base_seq[idx] = self._accumulate_weighted(
                    base_seq[idx], item, weight, context
                )
            return tuple(base_seq) if is_tuple else base_seq

        if isinstance(value, np.ndarray):
            base_is_array = isinstance(target, np.ndarray)
            value_is_floating = np.issubdtype(value.dtype, np.floating)
            if base_is_array:
                base = target
                if not np.issubdtype(base.dtype, np.floating):
                    base = base.astype(np.float64, copy=False)
            else:
                dtype = value.dtype if value_is_floating else np.float64
                base = np.zeros_like(value, dtype=dtype)

            scaled_value = value.astype(base.dtype, copy=False) * weight
            return base + scaled_value

        if torch is not None and isinstance(value, torch.Tensor):
            value_tensor = value
            if not torch.is_floating_point(value_tensor):
                value_tensor = value_tensor.to(torch.get_default_dtype())

            if isinstance(target, torch.Tensor):
                base_tensor = target
                if not torch.is_floating_point(base_tensor):
                    base_tensor = base_tensor.to(value_tensor.dtype)
                elif base_tensor.dtype != value_tensor.dtype:
                    base_tensor = base_tensor.to(value_tensor.dtype)
            else:
                base_tensor = torch.zeros_like(value_tensor, dtype=value_tensor.dtype)

            return base_tensor + value_tensor * weight

        if hasattr(value, "shape"):
            base = target
            if base is None:
                try:
                    if zeros_fn is not None:
                        base = zeros_fn(value.shape)
                    else:
                        raise AttributeError
                except (AttributeError, TypeError, ValueError):
                    base = np.zeros(value.shape, dtype=getattr(value, "dtype", None))
            if hasattr(base, "__iadd__"):
                base += value * weight
                return base
            return base + value * weight

        base = 0.0 if target is None else target
        return base + value * weight

    def _match_reference_structure(self, data: Any, reference: Any) -> Any:
        """Cast aggregated data to match reference dtypes."""
        if reference is None or data is None:
            return data

        if isinstance(reference, Mapping) and isinstance(data, Mapping):
            for key, item in list(data.items()):
                if key in reference:
                    data[key] = self._match_reference_structure(item, reference[key])
            return data

        if isinstance(reference, (list, tuple)) and isinstance(data, (list, tuple)):
            length = min(len(reference), len(data))
            aligned = [
                self._match_reference_structure(data[idx], reference[idx])
                for idx in range(length)
            ]
            if isinstance(data, tuple):
                return tuple(aligned + list(data[length:]))
            aligned.extend(data[length:])
            return aligned

        if torch is not None and isinstance(reference, torch.Tensor):
            if isinstance(data, torch.Tensor):
                return self._cast_tensor_like(data, reference)

        if isinstance(reference, np.ndarray) and isinstance(data, np.ndarray):
            return self._cast_ndarray_like(data, reference)

        if isinstance(reference, numbers.Integral):
            if isinstance(data, numbers.Real):
                return type(reference)(round(float(data)))

        if isinstance(reference, bool):
            if isinstance(data, numbers.Real):
                return bool(data >= 0.5)

        return data

    @staticmethod
    def _cast_tensor_like(tensor: torch.Tensor, reference: torch.Tensor):
        """Cast a tensor to match the dtype of the reference tensor."""
        if tensor.dtype == reference.dtype:
            return tensor

        if torch.is_floating_point(reference):
            return tensor.to(reference.dtype)

        if reference.dtype == torch.bool:
            if torch.is_floating_point(tensor):
                return tensor >= 0.5
            return tensor.ne(0)

        if torch.is_floating_point(tensor):
            return torch.round(tensor).to(reference.dtype)

        return tensor.to(reference.dtype)

    @staticmethod
    def _cast_ndarray_like(array: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Cast an ndarray to match the reference array's dtype."""
        if array.dtype == reference.dtype:
            return array

        if np.issubdtype(reference.dtype, np.bool_):
            if np.issubdtype(array.dtype, np.floating):
                return array >= 0.5
            return array.astype(np.bool_, copy=False)

        if np.issubdtype(reference.dtype, np.integer) and np.issubdtype(
            array.dtype, np.floating
        ):
            return np.rint(array).astype(reference.dtype, copy=False)

        return array.astype(reference.dtype, copy=False)
