"""
Client selection strategy for personalized federated learning.
"""

from __future__ import annotations

from typing import List, Optional

from plato.servers.strategies.base import ClientSelectionStrategy, ServerContext
from plato.servers.strategies.client_selection.random_selection import (
    RandomSelectionStrategy,
)


class PersonalizedRatioSelectionStrategy(ClientSelectionStrategy):
    """
    Select clients according to a participation ratio during regular rounds and
    include all clients during the personalization phase.

    Args:
        ratio: Fraction of total clients eligible during regular rounds.
        personalization_rounds: Number of regular rounds before personalization.
        base_strategy: Strategy used to perform the actual sampling.
    """

    def __init__(
        self,
        ratio: float,
        personalization_rounds: int,
        base_strategy: Optional[ClientSelectionStrategy] = None,
    ):
        self.ratio = max(0.0, min(1.0, ratio))
        self.personalization_rounds = personalization_rounds
        self.base_strategy = base_strategy or RandomSelectionStrategy()

    def setup(self, context: ServerContext) -> None:
        self.base_strategy.setup(context)

    def select_clients(
        self,
        clients_pool: List[int],
        clients_count: int,
        context: ServerContext,
    ) -> List[int]:
        candidates = clients_pool

        if context.current_round <= self.personalization_rounds:
            max_candidates = max(
                clients_count,
                int(round(context.total_clients * self.ratio)),
            )
            max_candidates = min(max_candidates, len(clients_pool))
            candidates = clients_pool[:max_candidates]

        return self.base_strategy.select_clients(candidates, clients_count, context)

    def on_clients_selected(
        self, selected_clients: List[int], context: ServerContext
    ) -> None:
        self.base_strategy.on_clients_selected(selected_clients, context)

    def on_reports_received(self, updates, context: ServerContext) -> None:
        self.base_strategy.on_reports_received(updates, context)
