"""Tests for the split learning client selection strategy."""

import random
from types import SimpleNamespace

import pytest

from plato.servers.strategies.base import ServerContext
from plato.servers.strategies.client_selection import (
    SplitLearningSequentialSelectionStrategy,
)


def _build_context(seed: int = 1234) -> ServerContext:
    """Helper to construct a server context with deterministic PRNG."""
    random.seed(seed)
    context = ServerContext()
    context.server = SimpleNamespace(next_client=True)
    context.state["prng_state"] = random.getstate()
    return context


def test_split_learning_reuses_client_until_release():
    """Ensure the same client is reused while gradients are exchanged."""
    context = _build_context()
    strategy = SplitLearningSequentialSelectionStrategy()
    strategy.setup(context)

    clients_pool = [1, 2, 3]

    first_selection = strategy.select_clients(clients_pool, 1, context)
    assert len(first_selection) == 1
    assert context.server.next_client is False

    second_selection = strategy.select_clients(clients_pool, 1, context)
    assert second_selection == first_selection
    assert context.server.next_client is False


def test_split_learning_cycles_through_clients():
    """Ensure clients are served sequentially before re-shuffling."""
    context = _build_context()
    strategy = SplitLearningSequentialSelectionStrategy()
    strategy.setup(context)

    clients_pool = [1, 2, 3]
    selected_clients = []

    # First client
    selection = strategy.select_clients(clients_pool, 1, context)
    selected_clients.append(selection[0])

    # Release twice to collect the remaining clients.
    for _ in range(2):
        context.server.next_client = True
        strategy.release_current_client(context)
        selection = strategy.select_clients(clients_pool, 1, context)
        selected_clients.append(selection[0])

    assert set(selected_clients) == set(clients_pool)
    assert len(selected_clients) == len(clients_pool)

    # After all clients have been used, releasing again should regenerate order.
    context.server.next_client = True
    strategy.release_current_client(context)
    new_selection = strategy.select_clients(clients_pool, 1, context)
    assert new_selection[0] in clients_pool
    assert new_selection[0] in selected_clients


def test_split_learning_rejects_multiple_clients():
    """Ensure the strategy enforces single-client-per-round semantics."""
    context = _build_context()
    strategy = SplitLearningSequentialSelectionStrategy()
    strategy.setup(context)

    with pytest.raises(AssertionError):
        strategy.select_clients([1, 2], 2, context)
