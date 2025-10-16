"""
MOON trainer composed from custom training and loss strategies.

The trainer packages the contrastive objective inspired by:
Qinbin Li, Bingsheng He, and Dawn Song.
"Model-Contrastive Federated Learning." CVPR 2021.
"""

from __future__ import annotations

import copy
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from moon_model import Model as MoonModel

from plato.config import Config
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.base import TrainingContext, TrainingStepStrategy
from plato.trainers.strategies.loss_criterion import LossCriterionStrategy


class MoonLossStrategy(LossCriterionStrategy):
    """Compute the combined cross-entropy and contrastive loss used by MOON."""

    def __init__(self) -> None:
        self.classification_loss: Optional[nn.Module] = None
        self.contrastive_loss: Optional[nn.Module] = None
        self.mu: float = 1.0
        self.temperature: float = 0.5

    def setup(self, context: TrainingContext) -> None:
        """Initialise loss functions and hyper-parameters."""
        self.classification_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = nn.CrossEntropyLoss()

        algorithm_cfg = getattr(Config(), "algorithm", None)
        self.mu = getattr(algorithm_cfg, "mu", 5.0) if algorithm_cfg else 5.0
        self.temperature = (
            getattr(algorithm_cfg, "temperature", 0.5) if algorithm_cfg else 0.5
        )

    def compute_loss(
        self, outputs, labels: torch.Tensor, context: TrainingContext
    ) -> torch.Tensor:
        """
        Combine task and contrastive objectives.

        Args:
            outputs: Dictionary with keys
                - logits: classification logits from the local model.
                - local_projection: projection vectors from the local model.
                - global_projection: projection vectors from the frozen global model.
                - prev_projections: list of projection tensors from historical
                  client models (may be empty on early rounds).
            labels: Ground-truth class indices for supervised loss.
            context: Training context (unused but required by interface).
        """
        if self.classification_loss is None or self.contrastive_loss is None:
            raise RuntimeError("MoonLossStrategy must be set up before computing loss.")

        logits = outputs["logits"]
        local_projection = outputs["local_projection"]
        global_projection = outputs["global_projection"]
        prev_projections: List[torch.Tensor] = outputs.get("prev_projections", [])

        cls_loss = self.classification_loss(logits, labels)

        # Build contrastive logits: first column is similarity to the current
        # global model (positive), remaining columns are similarities to the
        # historic local models (negatives).
        cosine = F.cosine_similarity
        pos_scores = cosine(local_projection, global_projection)
        contrastive_logits = pos_scores.unsqueeze(1)

        for prev_proj in prev_projections:
            neg_scores = cosine(local_projection, prev_proj)
            contrastive_logits = torch.cat(
                (contrastive_logits, neg_scores.unsqueeze(1)), dim=1
            )

        contrastive_logits = contrastive_logits / self.temperature
        contrastive_labels = torch.zeros(
            contrastive_logits.size(0),
            dtype=torch.long,
            device=contrastive_logits.device,
        )
        contrastive_loss = self.contrastive_loss(contrastive_logits, contrastive_labels)

        return cls_loss + self.mu * contrastive_loss


class MoonTrainingStepStrategy(TrainingStepStrategy):
    """Perform a MOON training step with frozen global and historical models."""

    def training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        examples: torch.Tensor,
        labels: torch.Tensor,
        loss_criterion,
        context: TrainingContext,
    ) -> torch.Tensor:
        """Compute gradients for the combined MOON loss."""
        if not hasattr(model, "forward_with_projection"):
            raise AttributeError(
                "MOON expects the model to implement forward_with_projection()."
            )

        optimizer.zero_grad()

        # Local model forward pass (requires gradients)
        _, local_projection, logits = model.forward_with_projection(examples)

        outputs = {
            "logits": logits,
            "local_projection": local_projection,
        }

        # Frozen global model for contrastive positive pairs
        global_model = context.state.get("moon_global_model")
        if global_model is not None:
            global_model.eval()
            with torch.no_grad():
                _, global_projection, _ = global_model.forward_with_projection(examples)
            outputs["global_projection"] = global_projection
        else:
            outputs["global_projection"] = local_projection.detach()

        # Historic models provide negative pairs
        prev_models = context.state.get("moon_prev_models", [])
        prev_projections: List[torch.Tensor] = []
        for prev_model in prev_models:
            prev_model.eval()
            with torch.no_grad():
                _, prev_proj, _ = prev_model.forward_with_projection(examples)
            prev_projections.append(prev_proj)
        outputs["prev_projections"] = prev_projections

        loss = loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        return loss


class Trainer(ComposableTrainer):
    """Composable trainer wired with MOON-specific strategies."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model or MoonModel,
            callbacks=callbacks,
            loss_strategy=MoonLossStrategy(),
            training_step_strategy=MoonTrainingStepStrategy(),
        )

    def clone_model(self) -> nn.Module:
        """
        Helper to clone the current model with detached parameters.

        Returns:
            A deepcopy of the underlying model positioned on CPU with gradients
            disabled, suitable for reuse as a frozen encoder.
        """
        cloned = copy.deepcopy(self.model)
        cloned.to(torch.device("cpu"))
        for param in cloned.parameters():
            param.requires_grad_(False)
        cloned.eval()
        return cloned
