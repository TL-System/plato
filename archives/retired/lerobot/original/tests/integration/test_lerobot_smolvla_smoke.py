"""Integration smoke for LeRobot datasource + SmolVLA model + LeRobot trainer."""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import pytest

from plato.config import Config
from tests.integration.utils import (
    async_run,
    build_minimal_config,
    configure_environment,
)
from tests.test_utils.lerobot_stubs import (
    FakeLeRobotDataset,
    FakeLeRobotDatasetMetadata,
    FakeSmolVLAPolicy,
    identity_pre_post_processors,
)

pytestmark = pytest.mark.integration


def test_lerobot_smolvla_end_to_end_smoke(monkeypatch):
    """Smoke test startup + one short local train + server report processing."""
    config = build_minimal_config(
        rounds=1,
        clients_per_round=1,
        total_clients=1,
        model_name="smolvla",
        trainer_type="lerobot",
    )
    config["server"]["do_test"] = False
    config["data"] = {
        "datasource": "LeRobot",
        "partition_size": 2,
        "sampler": "iid",
        "random_seed": 1,
    }
    config["trainer"].update(
        {
            "type": "lerobot",
            "model_type": "smolvla",
            "model_name": "smolvla",
            "epochs": 1,
            "batch_size": 2,
            "optimizer": "SGD",
        }
    )
    config["parameters"] = {
        "policy": {
            "path": "stub/smolvla",
            "finetune_mode": "adapter",
            "adapter_parameter_patterns": ["adapter"],
        },
        "dataset": {
            "repo_id": "stub/lerobot",
            "split_seed": 4,
            "train_split": 0.5,
            "task_aware_split": True,
            "task_aware_partition": True,
        },
        "optimizer": {
            "lr": 0.05,
            "momentum": 0.0,
            "weight_decay": 0.0,
        },
    }

    with configure_environment(config):
        Config.args.id = 1
        FakeLeRobotDataset.reset_calls()
        FakeSmolVLAPolicy.reset_calls()

        lerobot_datasource = import_module("plato.datasources.lerobot")
        smolvla_model = import_module("plato.models.smolvla")
        lerobot_trainer = import_module("plato.trainers.lerobot")
        processor_registry = import_module("plato.processors.registry")
        client_mod = import_module("plato.clients.simple")
        server_mod = import_module("plato.servers.fedavg")

        monkeypatch.setattr(
            lerobot_datasource,
            "_import_lerobot",
            lambda: (FakeLeRobotDataset, FakeLeRobotDatasetMetadata),
        )
        monkeypatch.setattr(
            lerobot_datasource,
            "_build_image_transforms",
            lambda _cfg: None,
        )
        monkeypatch.setattr(
            smolvla_model,
            "_import_smolvla_policy",
            lambda: FakeSmolVLAPolicy,
        )
        monkeypatch.setattr(
            lerobot_trainer,
            "_import_make_pre_post_processors",
            lambda: identity_pre_post_processors,
        )
        monkeypatch.setattr(
            processor_registry,
            "get",
            lambda *args, **kwargs: (None, None),
        )

        client = client_mod.Client()
        client._load_data()
        client.configure()
        client._allocate_data()

        report, payload = async_run(client._train())
        report.processing_time = 0.0

        assert report.num_samples > 0
        assert list(payload.keys()) == ["adapter.weight"]

        server = server_mod.Server()
        server.configure()
        server.updates = [
            SimpleNamespace(
                client_id=1,
                report=report,
                payload=payload,
            )
        ]
        server.current_round = 0
        server.context.current_round = 0

        async_run(server._process_reports())

        assert server.accuracy >= 0
