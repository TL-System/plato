"""
Model factory for MOON experiments.

Selects a MOON-compatible model (with projection head) based on the configured
trainer.model_name. Supports LeNet-5 (EMNIST/FEMNIST), ResNet-18 (CIFAR-10),
and VGG-16 (CINIC-10) using the same unified settings as other runs.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from moon_model import Model as MoonLeNetModel

from plato.config import Config
from plato.models import resnet, vgg


def _resolve_model_name() -> str:
    trainer = getattr(Config(), "trainer", None)
    model_name = getattr(trainer, "model_name", None) if trainer else None
    if not isinstance(model_name, str):
        return "lenet5"
    normalized = model_name.lower().replace("-", "_")
    if normalized == "resnet18":
        return "resnet_18"
    if normalized == "vgg16":
        return "vgg_16"
    return normalized


def _resolve_num_classes(default: int = 10) -> int:
    parameters = getattr(Config(), "parameters", None)
    model = getattr(parameters, "model", None) if parameters else None
    num_classes = getattr(model, "num_classes", None) if model else None
    return int(num_classes) if isinstance(num_classes, int) else default


class MoonLeNetWithProjection(MoonLeNetModel):
    """LeNet-5 MOON model with config-driven class count."""

    def __init__(self, num_classes: int | None = None, projection_dim: int = 128, **_):
        if num_classes is None:
            num_classes = _resolve_num_classes(default=10)
        super().__init__(num_classes=num_classes, projection_dim=projection_dim)


class MoonResNetWithProjection(nn.Module):
    """ResNet-18 backbone with a MOON projection head."""

    def __init__(self, num_classes: int | None = None, projection_dim: int = 128, **_):
        super().__init__()
        if num_classes is None:
            num_classes = _resolve_num_classes(default=10)
        model_name = _resolve_model_name()
        if not model_name.startswith("resnet_"):
            model_name = "resnet_18"
        self.base = resnet.Model.get(model_name=model_name, num_classes=num_classes)
        self.projection_head = nn.Sequential(
            nn.Linear(512 * resnet.BasicBlock.expansion, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.base.bn1(self.base.conv1(x)))
        out = self.base.layer1(out)
        out = self.base.layer2(out)
        out = self.base.layer3(out)
        out = self.base.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._encode(x)
        logits = self.base.linear(features)
        return logits

    def forward_with_projection(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._encode(x)
        projection = self.projection_head(features)
        projection = F.normalize(projection, dim=1, eps=1e-12)
        logits = self.base.linear(features)
        return features, projection, logits


class MoonVGGWithProjection(nn.Module):
    """VGG-16 backbone with a MOON projection head."""

    def __init__(self, num_classes: int | None = None, projection_dim: int = 128, **_):
        super().__init__()
        if num_classes is None:
            num_classes = _resolve_num_classes(default=10)
        model_name = _resolve_model_name()
        if not model_name.startswith("vgg_"):
            model_name = "vgg_16"
        self.base = vgg.Model.get(model_name=model_name, num_classes=num_classes)
        self.projection_head = nn.Sequential(
            nn.Linear(512, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base.layers(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._encode(x)
        logits = self.base.fc(features)
        return logits

    def forward_with_projection(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._encode(x)
        projection = self.projection_head(features)
        projection = F.normalize(projection, dim=1, eps=1e-12)
        logits = self.base.fc(features)
        return features, projection, logits


def resolve_moon_model() -> Any:
    """Return the MOON-compatible model class for the configured trainer model."""
    model_name = _resolve_model_name()
    if model_name.startswith("resnet_"):
        return MoonResNetWithProjection
    if model_name.startswith("vgg_"):
        return MoonVGGWithProjection
    return MoonLeNetWithProjection
