"""Unit tests for MPC coordination utilities."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import torch

from plato.mpc import RoundInfoStore
from plato.processors.mpc_model_encrypt_additive import Processor as AdditiveProcessor
from plato.processors.mpc_model_encrypt_shamir import Processor as ShamirProcessor
from plato.servers.strategies.mpc import (
    MPCAdditiveAggregationStrategy,
    MPCShamirAggregationStrategy,
)


class DummyTrainer:
    """Minimal trainer stub returning zero tensors."""

    @staticmethod
    def zeros(shape):
        return torch.zeros(shape)


def test_round_info_store_local(tmp_path):
    store = RoundInfoStore(storage_dir=str(tmp_path))

    state = store.initialise_round(round_number=1, selected_clients=[1, 2])
    assert state.selected_clients == [1, 2]

    store.record_client_samples(1, 10)
    store.record_client_samples(2, 20)

    store.append_additive_share(1, {"w": torch.tensor(1.0)})
    store.store_pairwise_share(1, 2, {"w": torch.tensor([[1.0, 1.0]])})

    loaded = store.load_state()
    assert loaded.client_samples[1] == 10
    assert torch.equal(loaded.additive_shares[1]["w"], torch.tensor(1.0))
    assert (1, 2) in loaded.pairwise_shares


def _run_additive_pipeline(tmp_path):
    store = RoundInfoStore(storage_dir=str(tmp_path))
    store.initialise_round(round_number=1, selected_clients=[1, 2])
    store.record_client_samples(1, 10)
    store.record_client_samples(2, 20)

    weights_client1 = {"w": torch.tensor(0.6)}
    weights_client2 = {"w": torch.tensor(0.8)}

    proc1 = AdditiveProcessor(client_id=1, round_store=store, debug_artifacts=False)
    proc2 = AdditiveProcessor(client_id=2, round_store=store, debug_artifacts=False)

    payload1 = proc1.process(weights_client1)
    payload2 = proc2.process(weights_client2)

    updates = [
        SimpleNamespace(client_id=1, report=SimpleNamespace(num_samples=10)),
        SimpleNamespace(client_id=2, report=SimpleNamespace(num_samples=20)),
    ]
    context = SimpleNamespace(trainer=DummyTrainer())

    strategy = MPCAdditiveAggregationStrategy(store)
    result = asyncio.run(
        strategy.aggregate_weights(
            updates,
            {"w": torch.tensor(0.0)},
            [payload1, payload2],
            context,
        )
    )

    return result


def test_additive_strategy_end_to_end(tmp_path):
    result = _run_additive_pipeline(tmp_path)
    expected = (10 * 0.6 + 20 * 0.8) / 30
    assert torch.isclose(result["w"], torch.tensor(expected))


def test_shamir_strategy_end_to_end(tmp_path):
    store = RoundInfoStore(storage_dir=str(tmp_path))
    store.initialise_round(round_number=1, selected_clients=[1, 2, 3])
    store.record_client_samples(1, 5)
    store.record_client_samples(2, 7)
    store.record_client_samples(3, 9)

    weights = [
        {"w": torch.tensor(0.2)},
        {"w": torch.tensor(0.4)},
        {"w": torch.tensor(0.6)},
    ]

    processors = [
        ShamirProcessor(client_id=idx + 1, round_store=store, debug_artifacts=False)
        for idx in range(3)
    ]

    payloads = [proc.process(weight) for proc, weight in zip(processors, weights)]

    updates = [
        SimpleNamespace(client_id=1, report=SimpleNamespace(num_samples=5)),
        SimpleNamespace(client_id=2, report=SimpleNamespace(num_samples=7)),
        SimpleNamespace(client_id=3, report=SimpleNamespace(num_samples=9)),
    ]
    context = SimpleNamespace(trainer=DummyTrainer())

    strategy = MPCShamirAggregationStrategy(store)
    result = asyncio.run(
        strategy.aggregate_weights(
            updates,
            {"w": torch.tensor(0.0)},
            payloads,
            context,
        )
    )

    total = 5 + 7 + 9
    expected = (5 * 0.2 + 7 * 0.4 + 9 * 0.6) / total
    assert torch.isclose(result["w"], torch.tensor(expected), atol=1e-4)
