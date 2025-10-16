"""
Random client selection strategy.
"""

from __future__ import annotations

import logging
import random
from typing import List

from plato.servers.strategies.base import ClientSelectionStrategy, ServerContext


class RandomSelectionStrategy(ClientSelectionStrategy):
    """Select clients uniformly at random from the available pool."""

    def select_clients(
        self,
        clients_pool: List[int],
        clients_count: int,
        context: ServerContext,
    ) -> List[int]:
        assert clients_count <= len(clients_pool), (
            f"Cannot select {clients_count} clients from pool of {len(clients_pool)}"
        )

        prng_state = context.state.get("prng_state")
        if prng_state:
            random.setstate(prng_state)

        selected_clients = random.sample(clients_pool, clients_count)
        context.state["prng_state"] = random.getstate()

        logging.info("[Server] Selected clients: %s", selected_clients)
        return selected_clients
