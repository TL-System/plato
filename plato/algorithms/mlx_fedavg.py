"""
Federated averaging utilities for MLX-based models.

This mirrors the PyTorch implementation but operates on MLX parameter trees,
allowing MLX trainers to participate in the same algorithm orchestration.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

from plato.algorithms import base

try:  # pragma: no cover - optional dependency
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None

from plato.trainers import mlx as mlx_trainer


def _tree_binary_map(
    func: Callable[[Any, Any], Any],
    tree_a: Any,
    tree_b: Any,
) -> Any:
    """Apply a function to paired leaves of two trees."""
    if isinstance(tree_a, dict) and isinstance(tree_b, dict):
        return {key: _tree_binary_map(func, tree_a[key], tree_b[key]) for key in tree_a}
    if isinstance(tree_a, (list, tuple)) and isinstance(tree_b, (list, tuple)):
        mapped = [
            _tree_binary_map(func, item_a, item_b)
            for item_a, item_b in zip(tree_a, tree_b)
        ]
        return type(tree_a)(mapped)
    return func(tree_a, tree_b)


def _to_numpy(value: Any) -> np.ndarray:
    """Ensure leaves are numpy arrays for transport."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if mx is not None and isinstance(value, mx.array):
        return np.array(value)
    if hasattr(value, "__array__"):
        return np.asarray(value)
    return np.array(value)


class Algorithm(base.Algorithm):
    """Federated averaging helper for MLX parameter trees."""

    def compute_weight_deltas(
        self,
        baseline_weights: Any,
        weights_received: Iterable[Any],
    ):
        """
        Compute deltas between the baseline and received weights.

        Args:
            baseline_weights: Nested structure of numpy arrays capturing baseline.
            weights_received: Iterable of nested structures matching baseline.

        Returns:
            List of nested structures representing parameter updates.
        """

        def difference(current, baseline):
            current_np = _to_numpy(current)
            baseline_np = _to_numpy(baseline)
            if current_np is None:
                return None
            return current_np - baseline_np

        deltas = []
        for weights in weights_received:
            deltas.append(_tree_binary_map(difference, weights, baseline_weights))
        return deltas

    def update_weights(self, deltas: Any):
        """Apply parameter deltas to the current model weights."""
        baseline = self.extract_weights()

        def add_delta(baseline_leaf, delta_leaf):
            baseline_np = _to_numpy(baseline_leaf)
            delta_np = _to_numpy(delta_leaf)
            if baseline_np is None:
                return None
            return baseline_np + delta_np

        updated = _tree_binary_map(add_delta, baseline, deltas)
        return updated

    def extract_weights(self, model=None):
        """Extract MLX model parameters as numpy-backed trees."""
        if model is None:
            params = self.model.parameters()
        else:
            params = model.parameters()
        return mlx_trainer._tree_map(mlx_trainer._to_host_array, params)

    def load_weights(self, weights):
        """Load weights into the MLX model."""
        restored = mlx_trainer._tree_map(mlx_trainer._to_mx_array, weights)
        if hasattr(self.model, "update"):
            self.model.update(restored)
        else:
            raise RuntimeError("MLX model does not support parameter updates.")
        if mx is not None:
            leaves = [
                leaf
                for leaf in mlx_trainer._tree_leaves(self.model.parameters())
                if isinstance(leaf, mx.array)
            ]
            if leaves:
                mx.eval(*leaves)
