"""Numerical and negative-path checks of real MPC processors and strategies."""

import asyncio
import random
import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace

import pytest
import torch

from plato.mpc import RoundInfoStore
from plato.processors.mpc_model_encrypt_additive import Processor as AdditiveProcessor
from plato.processors.mpc_model_encrypt_shamir import Processor as ShamirProcessor
from plato.servers.strategies.base import ServerContext
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
    strategy = (
        MPCAdditiveAggregationStrategy(store) if kind == "additive"
        else MPCShamirAggregationStrategy(store, threshold=threshold)
    )
    return store, strategy, weights, payloads, updates, context


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


@pytest.mark.parametrize(
    "configured_threshold,processor_override",
    [(1, None), (2, None), (None, None), (1, 4), (None, 1)],
)
def test_configured_shamir_client_lifecycle_matches_server_reconstruction(
    tmp_path, monkeypatch, configured_threshold, processor_override
):
    from tests.integration.utils import build_minimal_config, configure_environment

    config = build_minimal_config(clients_per_round=5, total_clients=5)
    config["clients"]["outbound_processors"] = [
        "mpc_model_encrypt_shamir", "safetensor_encode"
    ]
    if configured_threshold is not None:
        config["server"]["mpc_shamir_threshold"] = configured_threshold

    # Isolate only general server startup. Execute the actual MPC wrapper,
    # configured store, client lifecycle/registry and weighted server strategy.
    def parent_init(server, *_args, **_kwargs):
        server._mpc_round_lock = threading.Lock()

    with configure_environment(config, runtime_root=tmp_path):
        from plato.clients.strategies.mpc import MPCLifecycleStrategy
        from plato.servers import fedavg, fedavg_mpc_shamir

        monkeypatch.setattr(fedavg.Server, "__init__", parent_init)
        server = fedavg_mpc_shamir.Server()
        store = server.round_store
        clients, counts = [9, 2, 5, 8, 4], [0.5, 0, 2.5, 1, 3]
        store.initialise_round(9, clients)
        weights = [
            {"w": torch.tensor([0.125 * (idx + 1), -0.2 * idx], dtype=torch.float64)}
            for idx in range(len(clients))
        ]
        expected = sum(w["w"] * n for w, n in zip(weights, counts)) / sum(counts)
        payloads, updates, contexts = [], [], []
        random.seed(23)
        for client_id, count, weight in zip(clients, counts, weights):
            store.record_client_samples(client_id, count, round_number=9)
            kwargs = {"model_deepcopy": {"client_id": 99}}
            if processor_override is not None:
                kwargs["mpc_model_encrypt_shamir"] = {"threshold": processor_override}
            context = SimpleNamespace(
                client_id=client_id, round_store=store, debug_artifacts=False,
                processor_kwargs=kwargs, model=None, datasource=None,
                trainer=SimpleNamespace(set_client_id=lambda _id: None),
                algorithm=SimpleNamespace(set_client_id=lambda _id: None),
            )
            MPCLifecycleStrategy().configure(context)
            encoded = context.outbound_processor.process(weight)
            assert isinstance(encoded, bytes)
            payloads.append(context.inbound_processor.process(encoded))
            updates.append(SimpleNamespace(
                client_id=client_id, report=SimpleNamespace(num_samples=count)
            ))
            contexts.append(context)

        baseline = {"w": torch.zeros(2, dtype=torch.float64)}
        aggregation_context = ServerContext()
        aggregation_context.current_round = 9
        assert isinstance(server.aggregation_strategy, MPCShamirAggregationStrategy)
        actual = asyncio.run(server.aggregation_strategy.aggregate_weights(
            updates[::-1], baseline, payloads[::-1], aggregation_context
        ))
        torch.testing.assert_close(actual["w"], expected, atol=2e-6, rtol=0)
        # Protocol parameters from server config take precedence over local
        # processor kwargs; unrelated processor overrides remain intact.
        assert server.aggregation_strategy.threshold == configured_threshold
        for context in contexts:
            assert context.processor_kwargs["mpc_model_encrypt_shamir"].get(
                "threshold"
            ) == configured_threshold
            assert context.processor_kwargs["model_deepcopy"] == {"client_id": 99}


def test_impossible_shamir_coefficient_pool_is_bounded_before_persistence(tmp_path):
    code = textwrap.dedent("""
        import pathlib
        import random
        import sys
        import time
        import torch
        from plato.mpc import RoundInfoStore
        from plato.processors.mpc_model_encrypt_shamir import Processor

        root = pathlib.Path(sys.argv[1])
        store = RoundInfoStore(storage_dir=root)
        store.initialise_round(1, range(1000))
        store.record_client_samples(0, 1, round_number=1)
        before = (root / "round_info").read_bytes()
        processor = Processor(client_id=0, round_store=store, threshold=1000,
                              debug_artifacts=False)
        random.seed(23)
        print("Entering Shamir process", flush=True)
        started = time.monotonic()
        try:
            processor.process({"w": torch.tensor(0.000010, dtype=torch.float64)})
        except ValueError as error:
            assert "coefficient pool" in str(error), str(error)
        else:
            raise AssertionError("Impossible coefficient request succeeded")
        elapsed = time.monotonic() - started
        assert elapsed < 2, elapsed
        assert (root / "round_info").read_bytes() == before
        assert sorted(path.name for path in root.iterdir()) == ["round_info"]
        print(f"Rejected before persistence in {elapsed:.3f}s", flush=True)
    """)
    # subprocess.run contains the historical infinite loop without leaking
    # CPU workers into the rest of the suite. The public path uses a real store.
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path)],
            capture_output=True, text=True, timeout=12, check=False,
        )
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b""
        if isinstance(output, bytes):
            output = output.decode()
        assert "Entering Shamir process" in output, output
        pytest.fail("Public Shamir process entered but exceeded 12-second deadline")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Rejected before persistence" in result.stdout
