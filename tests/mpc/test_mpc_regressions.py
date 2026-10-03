"""Numerical and negative-path checks of real MPC processors and strategies."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from plato.mpc import RoundInfoStore
from plato.processors.mpc_model_encrypt_additive import Processor as AdditiveProcessor
from plato.processors.mpc_model_encrypt_shamir import Processor as ShamirProcessor
from plato.servers.strategies.mpc import (
    MPCAdditiveAggregationStrategy,
    MPCShamirAggregationStrategy,
)


def round_pipeline(tmp_path, kind, clients, counts, threshold=None):
    store = RoundInfoStore(storage_dir=tmp_path)
    store.initialise_round(7, clients)
    for client_id, count in zip(clients, counts):
        store.record_client_samples(client_id, count)
    weights = [
        {"w": torch.tensor([0.125 + idx, -0.75 * idx], dtype=torch.float64)}
        for idx in range(len(clients))
    ]
    processor_cls = AdditiveProcessor if kind == "additive" else ShamirProcessor
    strategy_cls = (
        MPCAdditiveAggregationStrategy
        if kind == "additive"
        else MPCShamirAggregationStrategy
    )
    kwargs = {} if kind == "additive" else {"threshold": threshold}
    payloads = [
        processor_cls(
            client_id=client_id, round_store=store, debug_artifacts=False, **kwargs
        ).process(weight)
        for client_id, weight in zip(clients, weights)
    ]
    updates = [
        SimpleNamespace(client_id=client_id, report=SimpleNamespace(num_samples=count))
        for client_id, count in zip(clients, counts)
    ]
    context = SimpleNamespace(
        trainer=SimpleNamespace(zeros=torch.zeros), current_round=7
    )
    return store, strategy_cls(store, **kwargs), weights, payloads, updates, context


@pytest.mark.parametrize(
    "kind,clients,counts,threshold",
    [
        ("additive", [9], [3], None),
        ("additive", [9, 2, 5, 8], [2, 0, 6, 3], None),
        ("shamir", [9], [3], 1),
        ("shamir", [9, 2, 5, 8], [2, 4, 6, 3], 2),
        ("shamir", [9, 2, 5, 8], [2, 4, 6, 3], 4),
    ],
)
def test_weighted_round_matches_plaintext_reference(
    tmp_path, kind, clients, counts, threshold
):
    store, strategy, weights, payloads, updates, context = round_pipeline(
        tmp_path, kind, clients, counts, threshold
    )
    # An independent plaintext calculation, before any secret sharing.
    expected = sum(w["w"] * n for w, n in zip(weights, counts)) / sum(counts)
    result = asyncio.run(
        strategy.aggregate_weights(
            updates[::-1], {"w": torch.zeros(2)}, payloads[::-1], context
        )
    )
    torch.testing.assert_close(result["w"].double(), expected, atol=2e-6, rtol=0)
    store.initialise_round(8, [2])
    state = store.load_state()
    assert state.client_samples == {2: None}
    assert state.additive_shares == {2: None}
    assert state.pairwise_shares == {(2, 2): None}


@pytest.mark.parametrize("kind", ["additive", "shamir"])
@pytest.mark.parametrize(
    "damage", ["cardinality", "samples", "duplicate", "stale", "shape", "keys"]
)
def test_malformed_round_fails_without_mutating_baseline(tmp_path, kind, damage):
    store, strategy, _weights, payloads, updates, context = round_pipeline(
        tmp_path, kind, [9, 2, 5], [2, 4, 6]
    )
    if damage == "cardinality":
        payloads.pop()
    elif damage == "samples":
        updates[0].report.num_samples = 999
    elif damage == "duplicate":
        updates[1].client_id = updates[0].client_id
    elif damage == "stale":
        context.current_round = 8
    elif damage == "shape":
        payloads[0]["w"] = payloads[0]["w"][:1]
    elif damage == "keys":
        payloads[0]["extra"] = payloads[0]["w"]
    baseline = {"w": torch.tensor([20.0, 30.0])}
    with pytest.raises((ValueError, RuntimeError)):
        asyncio.run(strategy.aggregate_weights(updates, baseline, payloads, context))
    assert torch.equal(baseline["w"], torch.tensor([20.0, 30.0]))


def test_additive_missing_participant_cannot_bias_aggregate(tmp_path):
    _store, strategy, _weights, payloads, updates, context = round_pipeline(
        tmp_path, "additive", [9, 2, 5], [2, 4, 6]
    )
    with pytest.raises(ValueError, match="participants"):
        asyncio.run(
            strategy.aggregate_weights(
                updates[:2], {"w": torch.zeros(2)}, payloads[:2], context
            )
        )


@pytest.mark.parametrize("kind", ["additive", "shamir"])
def test_missing_sample_count_is_rejected_before_storing_shares(tmp_path, kind):
    store = RoundInfoStore(storage_dir=tmp_path)
    store.initialise_round(1, [1, 2])
    processor_cls = AdditiveProcessor if kind == "additive" else ShamirProcessor
    processor = processor_cls(client_id=1, round_store=store, debug_artifacts=False)
    with pytest.raises(ValueError, match="sample"):
        processor.process({"w": torch.ones(2)})
    assert store.load_state().additive_shares == {1: None, 2: None}
    assert all(share is None for share in store.load_state().pairwise_shares.values())


@pytest.mark.parametrize("count", [-1, float("nan"), float("inf")])
def test_invalid_sample_count_is_not_persisted(tmp_path, count):
    store = RoundInfoStore(storage_dir=tmp_path)
    store.initialise_round(1, [1])
    with pytest.raises(ValueError, match="sample"):
        store.record_client_samples(1, count)
    assert store.load_state().client_samples == {1: None}


def test_stale_round_writer_cannot_contaminate_next_round(tmp_path, monkeypatch):
    store = RoundInfoStore(storage_dir=tmp_path)
    store.initialise_round(1, [1, 2])
    store.record_client_samples(1, 2)
    original = AdditiveProcessor._split_tensor

    def advance_round(tensor, num_shares):
        store.initialise_round(2, [1, 2])
        return original(tensor, num_shares)

    monkeypatch.setattr(AdditiveProcessor, "_split_tensor", staticmethod(advance_round))
    processor = AdditiveProcessor(client_id=1, round_store=store, debug_artifacts=False)
    with pytest.raises(RuntimeError, match="round"):
        processor.process({"w": torch.ones(2)})
    assert store.load_state().additive_shares == {1: None, 2: None}


def test_failed_state_save_keeps_last_complete_round(tmp_path, monkeypatch):
    from plato.mpc import round_store

    store = RoundInfoStore(storage_dir=tmp_path)
    store.initialise_round(1, [1])

    def failed_dump(_state, handle):
        handle.write(b"partial")
        raise OSError("Simulated write failure")

    with monkeypatch.context() as patch:
        patch.setattr(round_store.pickle, "dump", failed_dump)
        with pytest.raises(OSError, match="write failure"):
            store.record_client_samples(1, 2)
    assert store.load_state().client_samples == {1: None}
    assert list(tmp_path.iterdir()) == [tmp_path / "round_info"]


def test_training_records_zero_samples_and_rejects_round_change(tmp_path, monkeypatch):
    from plato.clients.strategies.defaults import DefaultTrainingStrategy
    from plato.clients.strategies.mpc import MPCTrainingStrategy

    store = RoundInfoStore(storage_dir=tmp_path)
    store.initialise_round(1, [1])

    async def train_zero(_self, _context):
        return SimpleNamespace(num_samples=0), {}

    monkeypatch.setattr(DefaultTrainingStrategy, "train", train_zero)
    context = SimpleNamespace(client_id=1)
    asyncio.run(MPCTrainingStrategy(store).train(context))
    assert store.load_state().client_samples == {1: 0}

    async def delayed_train(_self, _context):
        store.initialise_round(2, [1])
        return SimpleNamespace(num_samples=7), {}

    monkeypatch.setattr(DefaultTrainingStrategy, "train", delayed_train)
    with pytest.raises(RuntimeError, match="round"):
        asyncio.run(MPCTrainingStrategy(store).train(context))
    assert store.load_state().client_samples == {1: None}
