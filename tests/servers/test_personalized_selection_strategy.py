"""Tests for personalized client selection strategy."""

import random

from plato.servers.strategies.base import ServerContext
from plato.servers.strategies.client_selection import (
    PersonalizedRatioSelectionStrategy,
)


def _build_context(total_clients: int, clients_per_round: int) -> ServerContext:
    random.seed(1234)
    context = ServerContext()
    context.total_clients = total_clients
    context.clients_per_round = clients_per_round
    context.state["prng_state"] = random.getstate()
    return context


def test_personalized_selection_respects_ratio():
    clients_pool = list(range(10))
    context = _build_context(total_clients=10, clients_per_round=3)
    context.current_round = 1

    strategy = PersonalizedRatioSelectionStrategy(
        ratio=0.4,
        personalization_rounds=5,
    )
    strategy.setup(context)

    selected = strategy.select_clients(clients_pool, 3, context)
    assert len(selected) == 3
    assert all(client < 4 for client in selected)


def test_personalized_selection_ensures_minimum_candidates():
    clients_pool = list(range(10))
    context = _build_context(total_clients=10, clients_per_round=4)
    context.current_round = 2

    strategy = PersonalizedRatioSelectionStrategy(
        ratio=0.1,  # Would otherwise yield 1 candidate
        personalization_rounds=5,
    )
    strategy.setup(context)

    selected = strategy.select_clients(clients_pool, 4, context)
    assert len(selected) == 4


def test_personalized_selection_uses_all_clients_after_regular_rounds():
    clients_pool = list(range(6))
    context = _build_context(total_clients=6, clients_per_round=6)
    context.current_round = 6  # Exceeds personalization_rounds

    strategy = PersonalizedRatioSelectionStrategy(
        ratio=0.3,
        personalization_rounds=5,
    )
    strategy.setup(context)

    selected = strategy.select_clients(clients_pool, len(clients_pool), context)
    assert len(selected) == len(clients_pool)
    assert set(selected) == set(clients_pool)


def test_personalized_selection_clamps_ratio():
    clients_pool = list(range(5))
    context = _build_context(total_clients=5, clients_per_round=2)
    context.current_round = 1

    strategy = PersonalizedRatioSelectionStrategy(
        ratio=1.5,  # Should be clamped to 1.0
        personalization_rounds=10,
    )
    strategy.setup(context)

    selected = strategy.select_clients(clients_pool, 2, context)
    assert len(selected) == 2
