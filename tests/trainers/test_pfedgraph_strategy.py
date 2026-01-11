"""Tests for pFedGraph trainer strategies."""

import torch
import torch.nn as nn

from plato.trainers.strategies.algorithms.pfedgraph_strategy import (
    PFedGraphLossStrategy,
    PFedGraphUpdateStrategy,
)
from plato.trainers.strategies.base import TrainingContext


class SimpleModel(nn.Module):
    """Minimal model for strategy tests."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return self.fc(x)


def test_pfedgraph_loss_maximizes_similarity():
    model = SimpleModel()
    context = TrainingContext()
    context.model = model

    update_strategy = PFedGraphUpdateStrategy()
    update_strategy.on_train_start(context)

    loss_strategy = PFedGraphLossStrategy(lambda_reg=1.0)
    loss_strategy.setup(context)

    outputs = torch.zeros((1, 2))
    labels = torch.zeros(1, dtype=torch.long)

    loss = loss_strategy.compute_loss(outputs, labels, context)
    reference = context.state["pfedgraph_reference_vector"]
    current = torch.cat([p.view(-1) for p in model.parameters()])
    similarity = torch.nn.functional.cosine_similarity(
        reference.to(current.device), current, dim=0, eps=1e-12
    )
    base_loss = nn.CrossEntropyLoss()(outputs, labels)
    assert torch.isclose(loss, base_loss - similarity, atol=1e-6)
