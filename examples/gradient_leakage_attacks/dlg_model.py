"""
Obtaining models adapted from existing work's implementations.

An extra return object named `feature` is added in each model's forward function,
which will be used in the defense Soteria.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch.nn as nn
from nn import lenet, resnet

from plato.config import Config


def get(model_name: str | None = None) -> Optional[Callable[[], nn.Module]]:
    """Get the model constructor with the provided name."""
    resolved_name = model_name or Config().trainer.model_name

    if resolved_name == "lenet":
        return lenet.Model

    if resolved_name.startswith("resnet_"):
        return resnet.get(model_name=resolved_name)

    # Set up model through plato's model library
    if Config().trainer.model_type == "vit":
        return None

    raise ValueError(f"No such model: {resolved_name}")
