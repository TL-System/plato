"""
Client selection strategy for split learning.
"""

from __future__ import annotations

import logging
import random
from typing import List

from plato.servers.strategies.base import ClientSelectionStrategy, ServerContext


class SplitLearningSequentialSelectionStrategy(ClientSelectionStrategy):
    """
    Select clients sequentially for split learning.

    Split learning requires clients to interact with the server one at a time.
    This strategy shuffles the client pool once, then serves clients in that
    order until all have participated. The same client is re-selected while the
    server continues the gradient exchange phase, and the next client is only
    scheduled when the server signals it is ready.
    """

    STATE_KEY = "split_learning_selection"

    def setup(self, context: ServerContext) -> None:
        """Initialize split learning selection state."""
        state = context.state.setdefault(self.STATE_KEY, {})
        state.setdefault("client_queue", [])
        state.setdefault("current_client", None)

    def select_clients(
        self,
        clients_pool: List[int],
        clients_count: int,
        context: ServerContext,
    ) -> List[int]:
        assert clients_count == 1, (
            "Split learning supports only one client per round. "
            f"Requested clients_per_round={clients_count}"
        )

        server = context.server
        state = context.state.setdefault(self.STATE_KEY, {})
        client_queue = state.setdefault("client_queue", [])
        current_client = state.get("current_client")

        # Determine if a new client should be scheduled.
        request_next_client = True
        if server is not None:
            request_next_client = getattr(server, "next_client", True)

        if request_next_client or current_client is None:
            # Refresh queue if exhausted.
            if not client_queue:
                client_queue.extend(self._shuffle_clients(clients_pool, context))
                logging.warning("Client order: %s", client_queue)

            current_client = client_queue.pop(0)
            state["current_client"] = current_client

            if server is not None:
                server.next_client = False

        return [current_client]

    def on_reports_received(self, updates, context: ServerContext) -> None:
        """No-op hook; kept for future adaptive policies."""

    def release_current_client(self, context: ServerContext) -> None:
        """Release the currently scheduled client."""
        state = context.state.get(self.STATE_KEY)
        if state is not None:
            state["current_client"] = None

    @staticmethod
    def _shuffle_clients(clients_pool: List[int], context: ServerContext) -> List[int]:
        """Shuffle clients reproducibly using shared PRNG state."""
        prng_state = context.state.get("prng_state")
        if prng_state:
            random.setstate(prng_state)

        shuffled = list(clients_pool)
        random.shuffle(shuffled)
        context.state["prng_state"] = random.getstate()

        return shuffled
