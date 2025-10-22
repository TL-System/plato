"""
LeNet-5 implementation using Apple's MLX framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

try:  # pragma: no cover - optional dependency
    import mlx.core as mx
    import mlx.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The MLX LeNet-5 model requires the `mlx` package. "
        "Install it with `uv pip install mlx` on Apple Silicon."
    ) from exc


def _ensure_batch(x: mx.array) -> mx.array:
    """Ensure the input has a batch dimension."""
    if x.ndim == 3:
        return mx.expand_dims(x, axis=0)
    return x


class LeNet5(nn.Module):
    """Classic LeNet-5 architecture implemented with MLX layers."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        x = _ensure_batch(x)
        x = self.conv1(x)
        x = nn.relu(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = self.conv2(x)
        x = nn.relu(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = mx.reshape(x, (x.shape[0], -1))
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.fc2(x)
        x = nn.relu(x)
        x = self.fc3(x)
        return x


def Model(num_classes: int = 10, **_) -> LeNet5:
    """
    Factory compatible with PyTorch registry signature.

    Args:
        num_classes: Number of output classes (defaults to 10).
        **_: Ignored additional parameters for compatibility.
    """
    return LeNet5(num_classes=num_classes)
