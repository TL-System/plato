"""
Lifecycle-oriented tests covering base, composable, split-learning, and MPC clients.
"""

from __future__ import annotations

import asyncio
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from plato.callbacks.client import LogProgressCallback
from plato.clients import base, mpc, split_learning
from plato.clients.strategies.defaults import DefaultTrainingStrategy
from plato.clients.strategies.mpc import MPCTrainingStrategy
from plato.config import Config
from tests.test_utils.fakes import (
    IdentityLifecycleStrategy,
    InMemoryReportingStrategy,
    NoOpCommunicationStrategy,
    RecordingPayloadStrategy,
    StaticTrainingStrategy,
)


class DummyClient(base.Client):
    """Concrete client wiring fake strategies for deterministic unit tests."""

    def __init__(
        self,
        *,
        model_factory,
        lifecycle_strategy,
        payload_strategy,
        training_strategy,
        reporting_strategy,
        communication_strategy,
    ):
        self.custom_model_factory = model_factory
        self._custom_datasource_factory = getattr(
            lifecycle_strategy, "_datasource_factory", None
        )

        super().__init__(callbacks=None)

        self.custom_model = (
            model_factory() if callable(model_factory) else model_factory
        )
        self.model = self.custom_model
        self.custom_datasource = self._custom_datasource_factory
        self.custom_trainer = None
        self.custom_algorithm = None
        self.trainer_callbacks = None

        self._context.custom_model = self.custom_model
        self._context.custom_datasource = self.custom_datasource
        self._context.custom_trainer = self.custom_trainer
        self._context.custom_algorithm = self.custom_algorithm
        self._context.trainer_callbacks = None
        self._context.report_customizer = lambda report: report
        self._context.state.setdefault("event_log", [])

        self._configure_composable(
            lifecycle_strategy=lifecycle_strategy,
            payload_strategy=payload_strategy,
            training_strategy=training_strategy,
            reporting_strategy=reporting_strategy,
            communication_strategy=communication_strategy,
        )

    def configure(self) -> None:
        self.lifecycle_strategy.configure(self._context)
        self._sync_from_context(
            (
                "model",
                "trainer",
                "algorithm",
                "outbound_processor",
                "inbound_processor",
                "sampler",
                "testset_sampler",
            )
        )

    def _load_data(self) -> None:
        self.lifecycle_strategy.load_data(self._context)
        self._sync_from_context(("datasource",))

    def _allocate_data(self) -> None:
        self.lifecycle_strategy.allocate_data(self._context)
        self._sync_from_context(("trainset", "testset"))

    def _load_payload(self, server_payload) -> None:
        self.training_strategy.load_payload(self._context, server_payload)

    async def _train(self):
        report, payload = await self.training_strategy.train(self._context)
        report = self.reporting_strategy.build_report(self._context, report)
        self._context.latest_report = report
        self.report = report
        return report, payload

    async def _obtain_model_at_time(self, client_id, requested_time):
        return await self.reporting_strategy.obtain_model_at_time(
            self._context, client_id, requested_time
        )


class RecordingClientCallback(LogProgressCallback):
    """Callback capturing outbound readiness events for ordering assertions."""

    def __init__(self):
        self.events: list[str] = []

    def on_outbound_ready(self, client, report, outbound_processor):
        client._context.state.setdefault("event_log", []).append("callback_outbound")
        self.events.append("callback_outbound")


@pytest.fixture
def dummy_client(
    temp_config,
    fake_model_cls,
    fake_datasource_cls,
    fake_reporting_strategy,
    fake_communication_strategy,
):
    """Build a dummy client with reusable fake strategies."""
    lifecycle = IdentityLifecycleStrategy(datasource_factory=fake_datasource_cls)
    payload_strategy = RecordingPayloadStrategy()
    training_strategy = StaticTrainingStrategy()
    client = DummyClient(
        model_factory=fake_model_cls,
        lifecycle_strategy=lifecycle,
        payload_strategy=payload_strategy,
        training_strategy=training_strategy,
        reporting_strategy=fake_reporting_strategy,
        communication_strategy=fake_communication_strategy,
    )
    return client, payload_strategy, training_strategy


def test_base_client_lifecycle_sets_attributes(
    dummy_client,
):
    client, payload_strategy, _ = dummy_client

    client._load_data()
    client.configure()
    client._allocate_data()

    assert client.datasource is not None
    assert client.trainset is not None
    assert client.datasource.num_train_examples() == len(client.trainset)

    inbound_payload = {"server": torch.ones(1)}
    client._load_payload(inbound_payload)
    assert client._context.state["loaded_payload"] == inbound_payload

    report, payload = asyncio.run(client._train())
    assert report.client_id == client.client_id
    assert report.num_samples == len(client.trainset)
    assert isinstance(payload, dict)
    assert set(payload.keys()) == {"weights"}
    assert client.report is client._context.latest_report
    assert payload_strategy.events == []  # no payload handling path yet


def test_composable_client_handles_payload_roundtrip(
    temp_config,
    tmp_path,
    fake_model_cls,
    fake_datasource_cls,
    fake_reporting_strategy,
    fake_communication_strategy,
):
    payload_strategy = RecordingPayloadStrategy()
    client = DummyClient(
        model_factory=fake_model_cls,
        lifecycle_strategy=IdentityLifecycleStrategy(
            datasource_factory=fake_datasource_cls
        ),
        payload_strategy=payload_strategy,
        training_strategy=StaticTrainingStrategy(),
        reporting_strategy=fake_reporting_strategy,
        communication_strategy=fake_communication_strategy,
    )

    client._load_data()
    client.configure()
    client._allocate_data()

    payload_data = {"weights": torch.tensor([3.0])}
    payload_path = tmp_path / "payload.pkl"
    with payload_path.open("wb") as handle:
        pickle.dump(payload_data, handle)

    response = {
        "current_round": 3,
        "id": client.client_id,
        "payload_filename": str(payload_path),
    }

    asyncio.run(client._payload_to_arrive(response))

    assert client.current_round == 3
    assert client.server_payload == payload_data
    assert "handle_done" in payload_strategy.events
    assert client._context.state["sent_reports"]
    assert client._context.state["sent_payloads"]


def test_base_client_handles_missing_datasource(
    temp_config,
    fake_model_cls,
    fake_reporting_strategy,
    fake_communication_strategy,
):
    payload_strategy = RecordingPayloadStrategy()
    client = DummyClient(
        model_factory=fake_model_cls,
        lifecycle_strategy=IdentityLifecycleStrategy(datasource_factory=None),
        payload_strategy=payload_strategy,
        training_strategy=StaticTrainingStrategy(),
        reporting_strategy=fake_reporting_strategy,
        communication_strategy=fake_communication_strategy,
    )

    client._load_data()
    client._allocate_data()
    client._context.state["event_log"] = []

    client._load_payload({"dummy": 1})
    report, _ = asyncio.run(client._train())

    assert report.num_samples == 0
    assert client.datasource is None
    assert client.trainset is None


def test_composable_client_request_update_orders_callbacks(dummy_client):
    client, payload_strategy, _ = dummy_client

    client._load_data()
    client.configure()
    client._allocate_data()
    client._load_payload({"initial": 1})
    asyncio.run(client._train())

    recording_callback = RecordingClientCallback
    client.add_callbacks([recording_callback])
    callback_instance = next(
        cb
        for cb in client.callback_handler.callbacks
        if isinstance(cb, RecordingClientCallback)
    )

    client._context.state["event_log"] = []
    payload_strategy.events.clear()

    asyncio.run(client._request_update({"client_id": client.client_id, "time": 0.5}))

    assert client._context.state["event_log"] == [
        "callback_outbound",
        "payload_outbound",
    ]
    assert callback_instance.events == ["callback_outbound"]
    assert payload_strategy.events.count("outbound_ready") == 1
    assert client._context.state["sent_reports"]
    assert client._context.state["sent_payloads"]


def test_composable_client_chunk_flow_supports_retries(dummy_client):
    client, payload_strategy, _ = dummy_client

    client.comm_simulation = False
    client._context.comm_simulation = False

    payload_bytes = pickle.dumps({"weights": torch.tensor([1.0])})

    asyncio.run(client._chunk_arrived(payload_bytes))
    asyncio.run(client._payload_arrived(client.client_id))
    asyncio.run(client._payload_done(client.client_id))

    assert "chunk" in payload_strategy.events
    assert "commit" in payload_strategy.events
    assert "finalise" in payload_strategy.events
    assert "handle_done" in payload_strategy.events
    assert client._context.state["sent_reports"]
    assert client._context.state["sent_payloads"]

    # Second run should not raise and should reuse buffers cleanly.
    payload_strategy.events.clear()
    asyncio.run(client._chunk_arrived(payload_bytes))
    asyncio.run(client._payload_arrived(client.client_id))
    asyncio.run(client._payload_done(client.client_id))
    assert "handle_done" in payload_strategy.events


def test_split_learning_client_state_management(temp_config):
    original_clients = Config.clients
    clients_dict = dict(original_clients._asdict())
    clients_dict["iteration"] = 5
    clients_dict["do_test"] = False
    new_clients = Config.namedtuple_from_dict(clients_dict)
    Config.clients = new_clients
    if getattr(Config, "_instance", None) is not None:
        Config._instance.clients = new_clients

    try:
        client = split_learning.Client()
        state = client._context.state["split_learning"]

        assert state["iterations"] == 5
        assert client.iterations == 5
        assert client.iter_left == 5

        client.iter_left = 3
        assert client.iter_left == 3

        client.contexts = {"foo": "bar"}
        assert client.contexts == {"foo": "bar"}

        client.original_weights = {"w": 1}
        assert client.original_weights == {"w": 1}

        client.static_sampler = SimpleNamespace(name="sampler")
        assert client.static_sampler.name == "sampler"
    finally:
        Config.clients = original_clients
        if getattr(Config, "_instance", None) is not None:
            Config._instance.clients = original_clients


def test_mpc_client_round_store_configuration(temp_config, tmp_path):
    original_clients = Config.clients
    clients_dict = dict(original_clients._asdict())
    clients_dict["outbound_processors"] = ["mpc_model_encrypt_aes"]
    new_clients = Config.namedtuple_from_dict(clients_dict)
    Config.clients = new_clients
    if getattr(Config, "_instance", None) is not None:
        Config._instance.clients = new_clients

    original_mpc_path = Config.params.get("mpc_data_path")
    Config.params["mpc_data_path"] = str(tmp_path / "mpc_data")

    try:
        client = mpc.Client(debug_artifacts=True)
        assert client._context.round_store is client.round_store
        assert client._context.debug_artifacts is True

        processor_kwargs = client.lifecycle_strategy._build_processor_kwargs(
            client._context
        )
        encrypt_settings = processor_kwargs["mpc_model_encrypt_aes"]
        assert encrypt_settings["client_id"] == client.client_id
        assert encrypt_settings["round_store"] is client.round_store
        assert encrypt_settings["debug_artifacts"] is True
    finally:
        Config.clients = original_clients
        if getattr(Config, "_instance", None) is not None:
            Config._instance.clients = original_clients
        if original_mpc_path is None:
            Config.params.pop("mpc_data_path", None)
        else:
            Config.params["mpc_data_path"] = original_mpc_path


def test_mpc_training_strategy_records_samples(temp_config, monkeypatch):
    class DummyRoundStore:
        def __init__(self):
            self.calls = []

        def record_client_samples(self, client_id, num_samples):
            self.calls.append((client_id, num_samples))

    round_store = DummyRoundStore()
    strategy = MPCTrainingStrategy(round_store)
    context = SimpleNamespace(client_id=7)

    async_mock = AsyncMock(
        return_value=(SimpleNamespace(num_samples=11), {"weights": torch.ones(1)})
    )
    monkeypatch.setattr(DefaultTrainingStrategy, "train", async_mock)

    report, payload = asyncio.run(strategy.train(context))
    assert report.num_samples == 11
    assert round_store.calls == [(7, 11)]
    assert payload == {"weights": torch.ones(1)}
