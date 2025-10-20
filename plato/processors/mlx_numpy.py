"""
Processor that converts payloads to NumPy arrays for MLX compatibility.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from plato.processors import base

try:  # pragma: no cover - optional dependency
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None

import types


def _to_numpy(value: Any) -> Any:
    """Recursively convert tensors/arrays to numpy arrays."""
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        return value

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    if mx is not None and isinstance(value, mx.array):
        return value.to_host()

    if isinstance(value, types.GeneratorType):
        return [_to_numpy(item) for item in value]

    if isinstance(value, dict):
        return {key: _to_numpy(val) for key, val in value.items()}

    if isinstance(value, (list, tuple)):
        converted = [_to_numpy(val) for val in value]
        return type(value)(converted)

    return np.array(value)


class Processor(base.Processor):
    """Processor that converts data payloads to numpy arrays."""

    def process(self, data: Any) -> Any:
        return _to_numpy(data)
