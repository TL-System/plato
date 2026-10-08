"""Disconnect cleanup must preserve checkpoints owned by other clients."""

import asyncio

from plato.clients.composable import ComposableClientEvents
from plato.config import Config
from tests.test_utils.fakes import (
    BoundaryClient,
    IdentityLifecycleStrategy,
    InMemoryReportingStrategy,
    NoOpCommunicationStrategy,
    RecordingPayloadStrategy,
    StaticTrainingStrategy,
)


def test_disconnect_removes_only_owned_temporary_checkpoints(temp_config, tmp_path):
    model_path = tmp_path / "models"
    model_path.mkdir()
    Config.params["model_path"] = str(model_path)
    Config().clients.shutdown_delay = 0
    client = BoundaryClient()
    client.client_id = 1
    owned = ["1_2_0.25.safetensors", "1_2_0.25.safetensors.pkl", "1_3_1.5.pth"]
    preserved = [
        "2_2_0.25.safetensors",
        "2_2_0.25.safetensors.pkl",
        "checkpoint_lenet5_2.safetensors",
        "current_round.pkl",
        "random_state_2.pkl",
        "1_2_0x25.pth",
        "1_2_0.25.pth.backup",
        "notes.txt",
    ]
    for name in owned + preserved:
        (model_path / name).write_text(name)
    client._configure_composable(
        lifecycle_strategy=IdentityLifecycleStrategy(),
        payload_strategy=RecordingPayloadStrategy(),
        training_strategy=StaticTrainingStrategy(),
        reporting_strategy=InMemoryReportingStrategy(),
        communication_strategy=NoOpCommunicationStrategy(),
    )
    core = client._require_composable()
    events = ComposableClientEvents("/", core)
    asyncio.run(events.on_disconnect())
    assert {p.name for p in model_path.iterdir()} == set(preserved)
