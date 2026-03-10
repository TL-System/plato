"""Tests for LeRobot trainer training-step behavior with synthetic data."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from plato.config import Config
from plato.trainers import lerobot as lerobot_trainer


class _SyntheticLeRobotDataset(torch.utils.data.Dataset):
    """Tiny deterministic dataset exposing LeRobot-like dict samples."""

    def __init__(self):
        self.meta = SimpleNamespace(stats={"action": {"mean": [0.0], "std": [1.0]}})
        self.samples = [
            {
                "observation.image": torch.tensor([0.0, 1.0], dtype=torch.float32),
                "action": torch.tensor([0.0, 2.0], dtype=torch.float32),
                "episode_index": 0,
            },
            {
                "observation.image": torch.tensor([1.0, 2.0], dtype=torch.float32),
                "action": torch.tensor([1.0, 3.0], dtype=torch.float32),
                "episode_index": 0,
            },
            {
                "observation.image": torch.tensor([2.0, 3.0], dtype=torch.float32),
                "action": torch.tensor([2.0, 4.0], dtype=torch.float32),
                "episode_index": 1,
            },
            {
                "observation.image": torch.tensor([3.0, 4.0], dtype=torch.float32),
                "action": torch.tensor([3.0, 5.0], dtype=torch.float32),
                "episode_index": 1,
            },
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


class _TinyLeRobotPolicy(nn.Module):
    """Tiny policy returning tuple output variants consumed by LeRobot trainer."""

    def __init__(self):
        super().__init__()
        self.adapter_scale = nn.Parameter(torch.tensor(1.0))
        self.config = {"policy": "tiny"}
        self.plato_policy_path = "stub/smolvla"

    def forward(self, batch: dict[str, Any], reduction: str = "mean"):
        action = batch["action"].float()
        prediction = self.adapter_scale * torch.ones_like(action)
        loss = torch.mean((prediction - action) ** 2)
        # Return mapping + tensor tuple to exercise normalization branch.
        return {"loss_component": loss.detach()}, loss


def test_lerobot_trainer_train_model_runs_on_tiny_synthetic_batch(
    temp_config,
    monkeypatch,
):
    """Trainer should complete one short run and update model parameters."""
    config = Config()
    config.trainer = config.trainer._replace(
        type="lerobot",
        model_type="smolvla",
        model_name="smolvla_unit",
        batch_size=2,
        epochs=1,
        optimizer="SGD",
    )
    config.parameters = Config.node_from_dict(
        {
            "optimizer": {
                "lr": 0.1,
                "momentum": 0.0,
                "weight_decay": 0.0,
            },
            "policy": {"path": "stub/smolvla"},
        }
    )

    factory_calls: dict[str, Any] = {}

    def _fake_pre_post_factory(policy_config, **kwargs):
        factory_calls["policy_config"] = policy_config
        factory_calls["kwargs"] = kwargs

        def _pre(batch):
            return batch

        def _post(outputs):
            return outputs

        return _pre, _post

    monkeypatch.setattr(
        lerobot_trainer,
        "_import_make_pre_post_processors",
        lambda: _fake_pre_post_factory,
    )

    model = _TinyLeRobotPolicy()
    trainer = lerobot_trainer.Trainer(model=model)
    trainset = _SyntheticLeRobotDataset()

    start_value = float(model.adapter_scale.detach().item())
    trainer.train_model(
        {
            "batch_size": 2,
            "epochs": 1,
            "run_id": "lerobot-unit",
        },
        trainset,
        sampler=list(range(len(trainset))),
    )
    end_value = float(model.adapter_scale.detach().item())

    assert end_value != start_value
    assert callable(trainer.context.state["lerobot_preprocessor"])
    assert trainer.context.state["lerobot_loss_dict"]["loss_component"] >= 0.0
    assert trainer.run_history.get_metric_values("train_loss")
    assert factory_calls["kwargs"]["pretrained_path"] == "stub/smolvla"
    assert factory_calls["kwargs"]["dataset_stats"] == trainset.meta.stats


def test_lerobot_trainer_consumes_policy_precision_and_device(
    temp_config,
    monkeypatch,
):
    """Trainer should apply policy precision/device runtime settings."""
    config = Config()
    config.trainer = config.trainer._replace(
        type="lerobot",
        model_type="smolvla",
        model_name="smolvla_unit",
        batch_size=2,
        epochs=1,
        optimizer="SGD",
    )
    config.parameters = Config.node_from_dict(
        {
            "optimizer": {
                "lr": 0.05,
                "momentum": 0.0,
                "weight_decay": 0.0,
            },
            "policy": {
                "path": "stub/smolvla",
                "precision": "bf16",
                "device": "cpu",
            },
        }
    )

    monkeypatch.setattr(
        lerobot_trainer,
        "_import_make_pre_post_processors",
        lambda: (lambda *_args, **_kwargs: (lambda batch: batch, lambda out: out)),
    )

    trainer = lerobot_trainer.Trainer(model=_TinyLeRobotPolicy())
    trainset = _SyntheticLeRobotDataset()

    assert trainer.device == "cpu"
    assert trainer.context.device == torch.device("cpu")
    assert trainer.context.state["lerobot_precision"] == "bf16"

    trainer.train_model(
        {"batch_size": 2, "epochs": 1, "run_id": "lerobot-precision"},
        trainset,
        sampler=list(range(len(trainset))),
    )
    assert trainer.context.state["lerobot_precision"] == "bf16"
    assert isinstance(trainer.context.state["lerobot_autocast_enabled"], bool)


def test_lerobot_trainer_rejects_unavailable_cuda_device(
    temp_config,
    monkeypatch,
):
    """Policy device should be validated against runtime accelerator availability."""
    config = Config()
    config.trainer = config.trainer._replace(
        type="lerobot",
        model_type="smolvla",
        model_name="smolvla_unit",
    )
    config.parameters = Config.node_from_dict(
        {
            "policy": {
                "path": "stub/smolvla",
                "device": "cuda",
            },
        }
    )

    monkeypatch.setattr(lerobot_trainer.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        lerobot_trainer.Trainer(model=_TinyLeRobotPolicy())
