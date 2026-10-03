"""Actual accepted runtime checkpoint writers and trainer readers agree."""

import asyncio
import copy
import pickle
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch

from plato.algorithms import fedavg, fedavg_personalized
from plato.clients.base import Client
from plato.clients.strategies.base import ClientContext
from plato.clients.strategies.defaults import DefaultCommunicationStrategy
from plato.clients.strategies.fedavg_personalized import (
    FedAvgPersonalizedPayloadStrategy,
)
from plato.config import Config
from plato.trainers.composable import ComposableTrainer
from plato.utils.checkpoint_paths import checkpoint_name
from tests.integration.utils import build_minimal_config, configure_environment
from tests.test_utils.fakes import (
    IdentityLifecycleStrategy,
    InMemoryReportingStrategy,
    NoOpCommunicationStrategy,
    RecordingPayloadStrategy,
    StaticTrainingStrategy,
)


def make_server():
    from plato.servers.fedavg import Server

    trainer = ComposableTrainer(model=torch.nn.Linear(2, 2))
    trainer.device = trainer.context.device = torch.device("cpu")
    server = Server()
    server.trainer = trainer
    server.algorithm = fedavg.Algorithm(trainer)
    server.clients = {100: {"sid": "worker", "client_id": 1}}
    server.sio = SimpleNamespace(emit=AsyncMock())
    return server


@pytest.mark.parametrize(
    "name", ["lenet5", "org/model", "org_model", "../model", "x" * 240]
)
def test_server_save_resume_restores_round_rng_weights_and_history(tmp_path, name):
    config = build_minimal_config(model_name=name, total_clients=10)
    config["clients"]["per_round"] = 2
    with configure_environment(config, runtime_root=tmp_path):
        server = make_server()
        server.current_round = 10
        expected = copy.deepcopy(server.trainer.model.state_dict())
        server.trainer.run_history.update_metric("round_marker", 10)
        selector = random.Random(9)
        server.prng_state = selector.getstate()
        pool = list(range(1, 11))
        first = server.choose_clients(pool, 2)
        assert first == selector.sample(pool, 2)
        expected_selections = [selector.sample(pool, 2) for _ in range(5)]
        # Global draws after construction are unrelated to the selector stream.
        random.seed(71)
        for _ in range(7):
            random.random()
        np.random.seed(19)
        server.save_to_checkpoint()
        expected_numpy = np.random.RandomState(19).random_sample()
        server.trainer.model.weight.data.zero_()
        server.trainer.run_history.reset()
        server.current_round = 0
        random.seed(20)
        np.random.seed(30)
        server._resume_from_checkpoint()
        assert server.current_round == 10 and server.resumed_session
        assert np.random.random() == expected_numpy
        actual_selections = []
        for _ in range(5):
            random.random()
            actual_selections.append(server.choose_clients(pool, 2))
        assert actual_selections == expected_selections
        for key, value in server.trainer.model.state_dict().items():
            torch.testing.assert_close(value, expected[key])
        assert server.trainer.run_history.get_metric_values("round_marker") == [10]
        root = Path(Config.params["checkpoint_path"])
        assert (
            root / checkpoint_name("checkpoint", name, 10, suffix=".safetensors")
        ).is_file()
        assert all(file.parent == root for file in root.iterdir())


def test_server_simulated_writer_to_actual_supplied_filename_reader(tmp_path):
    config = build_minimal_config(model_name="org/model", total_clients=1)
    with configure_environment(config, runtime_root=tmp_path):
        server = make_server()
        asyncio.run(server._select_clients())
        response = server.sio.emit.call_args_list[0].args[1]["response"]
        filename = Path(response["payload_filename"])
        assert filename.parent == Path(Config.params["checkpoint_path"])
        expected = copy.deepcopy(server.trainer.model.state_dict())
        client = Client()
        client._configure_composable(
            lifecycle_strategy=IdentityLifecycleStrategy(),
            payload_strategy=RecordingPayloadStrategy(),
            training_strategy=StaticTrainingStrategy(),
            reporting_strategy=InMemoryReportingStrategy(),
            communication_strategy=NoOpCommunicationStrategy(),
        )
        asyncio.run(client._payload_to_arrive(response))
        for key, value in client.server_payload.items():
            torch.testing.assert_close(value, expected[key])
        assert client.client_id == 1


@pytest.mark.parametrize("legacy_sender", [False, True])
def test_client_simulated_writers_and_server_reader_do_not_alias_names(
    tmp_path, legacy_sender
):
    config = build_minimal_config(model_name="org/model", total_clients=1)
    with configure_environment(config, runtime_root=tmp_path):
        paths = []
        for name, number in [("org/model", 7), ("org_model", 11)]:
            Config().trainer.model_name = name
            payload = {"weight": torch.tensor([number])}
            context = ClientContext()
            # Preserve B's urgent report identity even when the physical worker
            # has already been reassigned to another logical client.
            context.client_id = 2
            context.state["outbound_client_id"] = 1
            if legacy_sender:
                client = Client()
                client.client_id = 1
                asyncio.run(client._send(payload))
            else:
                asyncio.run(
                    DefaultCommunicationStrategy().send_payload(context, payload)
                )
            root = Path(Config.params["checkpoint_path"])
            paths.append(root / checkpoint_name(name, "client", 1, suffix=".pkl"))
            server = make_server()
            server.clients[100]["client_id"] = 1
            server.training_clients[1] = {"id": 1, "starting_round": 1, "start_time": 0}
            server.training_sids = ["worker"]
            server._assign_client("worker", 1)
            server.process_client_info = AsyncMock()
            report = SimpleNamespace(client_id=1, num_samples=2)
            asyncio.run(
                server._client_report_arrived("worker", 1, pickle.dumps(report))
            )
            assert server.client_payload["worker"]["weight"].item() == number
            server.process_client_info.assert_awaited_once()
        assert paths[0] != paths[1]
        assert all(path.is_file() for path in paths)
        with paths[0].open("rb") as file:
            assert pickle.load(file)["weight"].item() == 7


@pytest.mark.parametrize("trailing_separator", [False, True])
def test_actual_personalized_outbound_hook_to_local_layer_reader(
    tmp_path, trailing_separator
):
    config = build_minimal_config(model_name="org/model")
    config["algorithm"]["local_layer_names"] = ["bias"]
    with configure_environment(config, runtime_root=tmp_path):
        if trailing_separator:
            Config.params["model_path"] += "/"
        trainer = ComposableTrainer(model=torch.nn.Linear(2, 2))
        trainer.set_client_id(7)
        algorithm = fedavg_personalized.Algorithm(trainer)
        algorithm.set_client_id(7)
        trainer.model.bias.data.fill_(4)
        context = ClientContext()
        context.client_id = 7
        context.owner = SimpleNamespace(outbound_ready=lambda *_: None)
        context.algorithm = algorithm
        FedAvgPersonalizedPayloadStrategy().outbound_ready(context, None, None)
        root = Path(Config.params["model_path"])
        assert (
            root
            / checkpoint_name("org/model", 7, "local_layers", suffix=".safetensors")
        ).is_file()
        incoming = {
            name: torch.zeros_like(value)
            for name, value in trainer.model.state_dict().items()
        }
        algorithm.load_weights(incoming)
        assert torch.all(trainer.model.bias == 4)
        assert torch.all(trainer.model.weight == 0)
        for filename in (
            "../outside.safetensors",
            str(tmp_path / "outside.safetensors"),
        ):
            with pytest.raises(ValueError, match="within"):
                algorithm.save_local_layers({"bias": trainer.model.bias}, filename)


def test_urgent_snapshot_numeric_selection_and_owned_cleanup_agree(tmp_path):
    with configure_environment(build_minimal_config(), runtime_root=tmp_path):
        trainer = ComposableTrainer(model=torch.nn.Linear(1, 1))
        trainer.set_client_id(7)
        root = Path(Config.params["model_path"])
        for epoch, time, weight in [(9, "9e-06", 9), (10, "1e-05", 10)]:
            trainer.model.weight.data.fill_(weight)
            trainer.save_model(checkpoint_name(7, epoch, time, suffix=".safetensors"))
        assert trainer.obtain_model_at_time(7, 11e-06).weight.item() == 10
        legacy = root / "7_11_0.25.pth"
        torch.save(trainer.model.state_dict(), legacy)
        (root / (legacy.name + ".pkl")).write_bytes(b"history")
        other = root / "8_10_1e-05.safetensors"
        other.write_bytes(b"other client")
        client = Client()
        client.client_id = 7
        client._clear_checkpoint_files()
        assert list(root.iterdir()) == [other]
