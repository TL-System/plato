"""Tests for LeRobot datasource registry wiring and constructor behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from plato.datasources import lerobot as lerobot_datasource
from plato.datasources import registry as datasources_registry
from tests.test_utils.lerobot_stubs import (
    FakeLeRobotDataset,
    FakeLeRobotDatasetMetadata,
)


@pytest.fixture
def patched_lerobot_backend(monkeypatch):
    """Patch LeRobot imports with deterministic local stubs."""
    FakeLeRobotDataset.reset_calls()
    fake_transforms = object()

    monkeypatch.setattr(
        lerobot_datasource,
        "_import_lerobot",
        lambda: (FakeLeRobotDataset, FakeLeRobotDatasetMetadata),
    )
    monkeypatch.setattr(
        lerobot_datasource,
        "_build_image_transforms",
        lambda _cfg: fake_transforms,
    )

    return SimpleNamespace(fake_transforms=fake_transforms)


def test_lerobot_is_registered_as_partitioned_datasource():
    """LeRobot datasource should be wired through the partitioned registry."""
    assert "LeRobot" in datasources_registry.registered_partitioned_datasources


def test_registry_constructs_lerobot_datasource_for_client(
    temp_config,
    patched_lerobot_backend,
):
    """Registry should build LeRobot datasource and pass client-aware options."""
    datasource = datasources_registry.get(
        datasource_name="LeRobot",
        client_id=1,
        repo_id="stub/lerobot",
        split_seed=7,
        train_split=0.5,
        delta_timestamps={"observation.image": [-0.1, 0.0]},
        dataset_kwargs={"streaming": True},
        task_aware_split=False,
        task_aware_partition=False,
    )

    assert isinstance(datasource, lerobot_datasource.DataSource)
    assert datasource.client_id == 1
    assert datasource.repo_id == "stub/lerobot"
    assert len(datasource.train_episodes) > 0

    train_call = FakeLeRobotDataset.constructor_calls[0]
    assert train_call["episodes"] == datasource.train_episodes
    assert train_call["delta_timestamps"] == {"observation.image": [-0.1, 0.0]}
    assert train_call["extra_kwargs"]["streaming"] is True
    assert train_call["image_transforms"] is patched_lerobot_backend.fake_transforms


def test_lerobot_constructor_is_deterministic_and_maps_samples(
    temp_config,
    patched_lerobot_backend,
):
    """Constructor should produce stable splits and mapped Plato sample keys."""
    first = lerobot_datasource.DataSource(
        client_id=2,
        repo_id="stub/lerobot",
        split_seed=11,
        train_split=0.5,
        task_aware_split=True,
        task_aware_partition=True,
    )
    second = lerobot_datasource.DataSource(
        client_id=2,
        repo_id="stub/lerobot",
        split_seed=11,
        train_split=0.5,
        task_aware_split=True,
        task_aware_partition=True,
    )

    assert first.train_episodes == second.train_episodes
    assert first.test_episodes == second.test_episodes
    assert first.num_train_examples() == len(first.get_train_set())
    assert first.num_test_examples() == len(first.get_test_set())

    sample = first.get_train_set()[0]
    assert "plato_inputs" in sample
    assert "plato_targets" in sample
    assert "plato_metadata" in sample
    assert "observation.image" in sample["plato_inputs"]
    assert torch.equal(sample["plato_targets"], sample["action"])
    assert sample["plato_metadata"]["episode_index"] in first.train_episodes
