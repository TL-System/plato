"""
MOON-specific model with projection head for contrastive learning.

This module implements a LeNet-5 style backbone augmented with a projection
head, returning both logits and projection vectors for the MOON objective.

Reference:
Qinbin Li, Bingsheng He, and Dawn Song.
"Model-Contrastive Federated Learning." CVPR 2021.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class Model(nn.Module):
    """
    LeNet-style network with an additional projection head.

    The standard forward pass returns classification logits so that evaluation
    pipelines remain unchanged. The ``forward_with_projection`` helper returns
    the intermediate activation, the projection vector, and the logits which
    the MOON trainer uses for the contrastive loss.
    """

    def __init__(
        self,
        num_classes: int = 10,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5, bias=True),
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten()
        self.hidden = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(84, num_classes)
        self.projection_head = nn.Sequential(
            nn.Linear(84, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for compatibility with default evaluation."""
        features = self._encode(x)
        logits = self.classifier(features)
        return logits

    def forward_with_projection(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the hidden representation, projection, and logits.

        The projection vectors are L2-normalised as required by MOON's
        cosine-similarity based loss.
        """
        features = self._encode(x)
        projection = self.projection_head(features)
        projection = F.normalize(projection, dim=1, eps=1e-12)
        logits = self.classifier(features)
        return features, projection, logits

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode inputs up to the shared hidden representation."""
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.hidden(x)
        return x
