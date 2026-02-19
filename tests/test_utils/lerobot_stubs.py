"""Deterministic LeRobot/SmolVLA stubs for offline integration tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn


class FakeLeRobotDatasetMetadata:
    """Minimal metadata surface consumed by the LeRobot datasource adapter."""

    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.total_episodes = 6
        self.tasks = ["pick", "place"]
        self.episodes = [
            {
                "episode_index": episode,
                "task_index": episode % 2,
                "task": self.tasks[episode % 2],
            }
            for episode in range(self.total_episodes)
        ]


class FakeLeRobotDataset:
    """Small dict-style dataset that mimics LeRobot sample payloads."""

    constructor_calls: list[dict[str, Any]] = []

    def __init__(
        self,
        repo_id: str,
        episodes: list[int] | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        image_transforms: Any = None,
        **kwargs: Any,
    ):
        self.repo_id = repo_id
        self.episodes = [int(episode) for episode in (episodes or [])]
        self.delta_timestamps = delta_timestamps
        self.image_transforms = image_transforms
        self.extra_kwargs = dict(kwargs)
        self.meta = SimpleNamespace(
            stats={
                "action": {
                    "mean": [0.0, 0.0],
                    "std": [1.0, 1.0],
                }
            }
        )

        self.samples: list[dict[str, Any]] = []
        for step, episode in enumerate(self.episodes):
            observation = torch.tensor(
                [float(episode), float(step)],
                dtype=torch.float32,
            )
            action = torch.tensor(
                [float(episode) + 0.25, float(step) + 0.5],
                dtype=torch.float32,
            )
            if callable(image_transforms):
                observation = image_transforms(observation)

            self.samples.append(
                {
                    "observation.image": observation,
                    "action": action,
                    "episode_index": episode,
                    "step_index": step,
                    "task": "pick" if episode % 2 == 0 else "place",
                }
            )

        self.targets = [sample["task"] for sample in self.samples]
        self.classes = ("pick", "place")

        type(self).constructor_calls.append(
            {
                "repo_id": repo_id,
                "episodes": list(self.episodes),
                "delta_timestamps": delta_timestamps,
                "image_transforms": image_transforms,
                "extra_kwargs": dict(kwargs),
            }
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]

    @classmethod
    def reset_calls(cls) -> None:
        """Clear constructor call history for test isolation."""
        cls.constructor_calls = []


class FakeSmolVLAPolicy(nn.Module):
    """Tiny policy compatible with SmolVLA wrapper expectations."""

    load_calls: list[dict[str, Any]] = []

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2, bias=False)
        self.adapter = nn.Linear(2, 2, bias=False)
        self.config = {"policy": "fake-smolvla"}

        with torch.no_grad():
            self.backbone.weight.copy_(torch.eye(2))
            self.adapter.weight.copy_(0.5 * torch.eye(2))

    @classmethod
    def from_pretrained(
        cls,
        policy_path: str,
        token: str | None = None,
        strict: bool = True,
    ) -> "FakeSmolVLAPolicy":
        """Match the LeRobot loader interface used by the wrapper."""
        cls.load_calls.append(
            {
                "path": policy_path,
                "token": token,
                "strict": strict,
            }
        )
        return cls()

    def forward(self, batch: dict[str, Any], reduction: str = "mean"):
        """Return tensor loss + dict payload like SmolVLA policies do."""
        action = batch["action"].float()
        prediction = self.adapter(self.backbone(action))
        per_sample = (prediction - action) ** 2

        if reduction == "sum":
            loss = per_sample.sum()
        else:
            loss = per_sample.mean()

        return loss, {"mse": loss.detach()}

    def save_pretrained(self, *args: Any, **kwargs: Any) -> None:
        """Compatibility no-op for checkpoint contract checks."""

    @classmethod
    def reset_calls(cls) -> None:
        """Clear loader call history for test isolation."""
        cls.load_calls = []


def identity_pre_post_processors(policy_config: Any, **kwargs: Any):
    """Return identity processors while recording constructor args."""

    def preprocessor(batch):
        return batch

    def postprocessor(outputs):
        return outputs

    setattr(preprocessor, "policy_config", policy_config)
    setattr(preprocessor, "kwargs", kwargs)

    return preprocessor, postprocessor
