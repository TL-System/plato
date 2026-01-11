"""
Composable trainer wiring for pFedGraph.
"""

from __future__ import annotations

from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms.pfedgraph_strategy import (
    PFedGraphLossStrategyFromConfig,
    PFedGraphUpdateStrategy,
)


class Trainer(ComposableTrainer):
    """Trainer pre-configured with pFedGraph strategies."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=PFedGraphLossStrategyFromConfig(),
            model_update_strategy=PFedGraphUpdateStrategy(),
        )
