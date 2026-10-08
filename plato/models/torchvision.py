"""Obtain models from the installed torchvision dependency."""

from typing import Any

from torch import nn
from torchvision import models


class Model:
    """Construct a torchvision model without loading remote Python code."""

    @staticmethod
    def get(model_name: str | None = None, **kwargs: Any) -> nn.Module:
        """Return a named model, preserving explicit weights precedence."""
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("A valid torchvision model name must be provided.")
        if "pretrained" in kwargs:
            pretrained = kwargs.pop("pretrained")
            kwargs.setdefault("weights", "DEFAULT" if pretrained else None)
        return models.get_model(model_name, **kwargs)
