"""
pFedGraph strategy implementations.

This module provides the client-side loss regularization and state management
needed for pFedGraph (ICML 2023).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from plato.config import Config
from plato.trainers.strategies.base import (
    LossCriterionStrategy,
    ModelUpdateStrategy,
    TrainingContext,
)


def _flatten_trainable_parameters(model: nn.Module) -> torch.Tensor:
    """Flatten trainable parameters into a single vector."""
    params = [param.view(-1) for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("No trainable parameters available for pFedGraph.")
    return torch.cat(params)


class PFedGraphUpdateStrategy(ModelUpdateStrategy):
    """Capture the reference model vector at the start of each local round."""

    def __init__(self, reference_key: str = "pfedgraph_reference_vector"):
        self.reference_key = reference_key

    def on_train_start(self, context: TrainingContext) -> None:
        model = context.model
        if model is None:
            raise ValueError("Training context must provide a model for pFedGraph.")

        reference_vector = _flatten_trainable_parameters(model).detach().cpu()
        context.state[self.reference_key] = reference_vector


class PFedGraphLossStrategy(LossCriterionStrategy):
    """
    pFedGraph loss strategy with cosine similarity regularization.

    The objective follows Equation (3) in the paper by maximizing cosine
    similarity between the local model and the aggregated model.
    """

    def __init__(
        self,
        lambda_reg: float = 0.01,
        base_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
        reference_key: str = "pfedgraph_reference_vector",
        eps: float = 1e-12,
    ):
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative for pFedGraph.")

        self.lambda_reg = lambda_reg
        self.base_loss_fn = base_loss_fn
        self.reference_key = reference_key
        self.eps = eps
        self._criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = (
            None
        )

    def setup(self, context: TrainingContext) -> None:
        if self.base_loss_fn is None:
            self._criterion = nn.CrossEntropyLoss()
        else:
            self._criterion = self.base_loss_fn

    def compute_loss(
        self, outputs: torch.Tensor, labels: torch.Tensor, context: TrainingContext
    ) -> torch.Tensor:
        criterion = self._criterion
        if criterion is None:
            raise RuntimeError("pFedGraph loss criterion has not been initialised.")

        base_loss = criterion(outputs, labels)
        if self.lambda_reg == 0:
            return base_loss

        model = context.model
        if model is None:
            raise ValueError("Training context must provide a model for pFedGraph.")

        reference_vector = context.state.get(self.reference_key)
        if reference_vector is None:
            return base_loss

        current_vector = _flatten_trainable_parameters(model)
        reference_vector = reference_vector.to(current_vector.device)
        if reference_vector.numel() != current_vector.numel():
            return base_loss

        similarity = torch.nn.functional.cosine_similarity(
            reference_vector, current_vector, dim=0, eps=self.eps
        )
        # Maximize similarity by subtracting it from the objective.
        total_loss = base_loss - self.lambda_reg * similarity
        return total_loss


class PFedGraphLossStrategyFromConfig(PFedGraphLossStrategy):
    """Config-driven variant of the pFedGraph loss strategy."""

    def __init__(self):
        config = Config()
        lambda_reg = 0.01
        if hasattr(config, "algorithm"):
            if hasattr(config.algorithm, "pfedgraph_lambda"):
                lambda_reg = config.algorithm.pfedgraph_lambda
            elif hasattr(config.algorithm, "pfedgraph_reg_lambda"):
                lambda_reg = config.algorithm.pfedgraph_reg_lambda

        super().__init__(lambda_reg=lambda_reg)
