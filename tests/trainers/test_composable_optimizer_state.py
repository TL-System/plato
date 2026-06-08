"""Tests for in-process optimizer state preservation in ComposableTrainer."""

import copy
import os
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from plato.config import Config
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies import (
    AdamWOptimizerStrategy,
    CrossEntropyLossStrategy,
    DefaultTrainingStepStrategy,
    SGDOptimizerStrategy,
    StepLRSchedulerStrategy,
)
from plato.trainers.strategies.base import OptimizerStrategy, TrainingContext

LOCAL_STATE_PAYLOAD_KEYS = {
    "optimizer_state",
    "scheduler_state",
    "trainer_state",
    "local_metadata",
    "metadata",
    "global_step",
    "local_optimizer_steps",
    "_optimizer_state_input_filename",
    "_optimizer_state_output_filename",
}


@pytest.fixture
def tiny_dataset():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.5],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    return TensorDataset(features, labels)


@pytest.fixture
def one_step_config():
    return {
        "batch_size": 4,
        "epochs": 1,
        "lr": 0.01,
        "run_id": "optimizer-state-test",
    }


class CapturingTrainingStep(DefaultTrainingStepStrategy):
    """Record optimizer state before each local optimizer step."""

    def __init__(self):
        super().__init__()
        self.pre_step_states = []
        self.pre_step_lrs = []

    def training_step(
        self,
        model,
        optimizer,
        examples,
        labels,
        loss_criterion,
        context,
    ):
        optimizer_state = optimizer.state_dict()
        self.pre_step_states.append(copy.deepcopy(optimizer_state["state"]))
        self.pre_step_lrs.append(
            [group["lr"] for group in optimizer_state["param_groups"]]
        )
        return super().training_step(
            model=model,
            optimizer=optimizer,
            examples=examples,
            labels=labels,
            loss_criterion=loss_criterion,
            context=context,
        )


def _linear_model():
    return nn.Sequential(OrderedDict([("linear", nn.Linear(2, 2))]))


class DeviceTrackingModel(nn.Module):
    """Model that records whether it has been moved to a trainer device."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.moved_to_trainer_device = False

    def forward(self, features):
        return self.linear(features)

    def to(self, *args, **kwargs):
        self.moved_to_trainer_device = True
        return super().to(*args, **kwargs)


class RestoreOrderOptimizer(torch.optim.SGD):
    """Optimizer that records whether state restore happens after model.to()."""

    def __init__(self, params, model: DeviceTrackingModel):
        self.model = model
        self.loaded_after_model_to = None
        super().__init__(params, lr=0.01, momentum=0.9)

    def load_state_dict(self, state_dict):
        self.loaded_after_model_to = self.model.moved_to_trainer_device
        if not self.loaded_after_model_to:
            raise AssertionError("optimizer state restored before model.to()")
        return super().load_state_dict(state_dict)


class RestoreOrderOptimizerStrategy(OptimizerStrategy):
    """Create restore-order-aware optimizers for regression tests."""

    def __init__(self):
        self.optimizers = []

    def create_optimizer(
        self, model: DeviceTrackingModel, context: TrainingContext
    ) -> torch.optim.Optimizer:
        optimizer = RestoreOrderOptimizer(model.parameters(), model)
        self.optimizers.append(optimizer)
        return optimizer


def _two_layer_model(first_name="first", second_name="second"):
    return nn.Sequential(
        OrderedDict(
            [
                (first_name, nn.Linear(2, 2, bias=False)),
                (second_name, nn.Linear(2, 2, bias=False)),
            ]
        )
    )


def _first_param_state(optimizer_state):
    return next(iter(optimizer_state.values()))


def _state_step(param_state):
    step = param_state["step"]
    if isinstance(step, torch.Tensor):
        return int(step.item())
    return int(step)


def _configure_subprocess_training(
    monkeypatch,
    tmp_path,
    *,
    preserve_optimizer_state,
):
    """Configure parent and spawned child processes to share local artifacts."""
    model_path = Path(tmp_path) / "models" / "pretrained"
    model_path.mkdir(parents=True, exist_ok=True)
    Config.params["model_path"] = str(model_path)
    Config.params["checkpoint_path"] = str(Path(tmp_path) / "checkpoints")
    Config.params["run_id"] = "subprocess-optimizer-state"
    os.makedirs(Config.params["checkpoint_path"], exist_ok=True)
    monkeypatch.setattr(sys, "argv", [sys.argv[0], "-b", str(tmp_path)])
    Config().trainer = Config().trainer._replace(
        max_concurrency=1,
        preserve_optimizer_state=preserve_optimizer_state,
        batch_size=4,
        epochs=1,
    )


def _cached_optimizer_step(trainer):
    payload = trainer._preserved_optimizer_states[trainer.client_id]
    return _state_step(_first_param_state(payload["optimizer_state"]["state"]))


def _cached_scheduler_last_epoch(trainer):
    payload = trainer._preserved_optimizer_states[trainer.client_id]
    return payload["scheduler_state"]["last_epoch"]


def _assert_model_update_contains_only_model_weights(update, model):
    model_state = model.state_dict()

    assert set(update) == set(model_state)
    assert LOCAL_STATE_PAYLOAD_KEYS.isdisjoint(update)
    assert all(torch.is_tensor(value) for value in update.values())


def test_adamw_moment_buffers_persist_between_rounds_for_same_client(
    temp_config, tiny_dataset, one_step_config
):
    step_strategy = CapturingTrainingStep()
    trainer = ComposableTrainer(
        model=_linear_model,
        loss_strategy=CrossEntropyLossStrategy(),
        optimizer_strategy=AdamWOptimizerStrategy(lr=0.01),
        training_step_strategy=step_strategy,
    )
    trainer.set_client_id(7)
    config = {**one_step_config, "preserve_optimizer_state": True}

    trainer.train_model(config, tiny_dataset, list(range(len(tiny_dataset))))
    round1_state = copy.deepcopy(trainer.optimizer.state_dict()["state"])
    trainer.train_model(config, tiny_dataset, list(range(len(tiny_dataset))))

    assert step_strategy.pre_step_states[0] == {}
    restored_state = _first_param_state(step_strategy.pre_step_states[1])
    saved_state = _first_param_state(round1_state)
    assert torch.allclose(restored_state["exp_avg"], saved_state["exp_avg"])
    assert torch.allclose(restored_state["exp_avg_sq"], saved_state["exp_avg_sq"])
    final_param_state = _first_param_state(trainer.optimizer.state_dict()["state"])
    assert _state_step(final_param_state) == 2


def test_preserved_optimizer_state_restores_after_model_moves_to_device(
    temp_config, tiny_dataset, one_step_config
):
    config = {**one_step_config, "preserve_optimizer_state": True}
    source_trainer = ComposableTrainer(
        model=DeviceTrackingModel,
        loss_strategy=CrossEntropyLossStrategy(),
        optimizer_strategy=RestoreOrderOptimizerStrategy(),
    )
    source_trainer.set_client_id(11)
    source_trainer.train_model(config, tiny_dataset, list(range(len(tiny_dataset))))

    restore_strategy = RestoreOrderOptimizerStrategy()
    trainer = ComposableTrainer(
        model=DeviceTrackingModel,
        loss_strategy=CrossEntropyLossStrategy(),
        optimizer_strategy=restore_strategy,
    )
    trainer.set_client_id(11)
    trainer._preserved_optimizer_states[11] = copy.deepcopy(
        source_trainer._preserved_optimizer_states[11]
    )

    trainer.train_model(config, tiny_dataset, list(range(len(tiny_dataset))))

    assert restore_strategy.optimizers[0].loaded_after_model_to is True
    restored_state = _first_param_state(
        trainer._preserved_optimizer_states[11]["optimizer_state"]["state"]
    )
    assert "momentum_buffer" in restored_state


def test_scheduler_state_and_lr_progress_persist_between_rounds(
    temp_config, tiny_dataset, one_step_config
):
    step_strategy = CapturingTrainingStep()
    trainer = ComposableTrainer(
        model=_linear_model,
        loss_strategy=CrossEntropyLossStrategy(),
        optimizer_strategy=SGDOptimizerStrategy(lr=0.2),
        training_step_strategy=step_strategy,
        lr_scheduler_strategy=StepLRSchedulerStrategy(step_size=1, gamma=0.5),
    )
    trainer.set_client_id(3)
    config = {**one_step_config, "preserve_optimizer_state": True}

    trainer.train_model(config, tiny_dataset, list(range(len(tiny_dataset))))
    trainer.train_model(config, tiny_dataset, list(range(len(tiny_dataset))))

    assert step_strategy.pre_step_lrs == [[0.2], [0.1]]
    assert trainer.lr_scheduler.last_epoch == 2
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_subprocess_optimizer_state_parent_reloads_after_child(
    temp_config, monkeypatch, tmp_path, tiny_dataset
):
    _configure_subprocess_training(monkeypatch, tmp_path, preserve_optimizer_state=True)
    trainer = ComposableTrainer(
        model=_linear_model,
        loss_strategy=CrossEntropyLossStrategy(),
        optimizer_strategy=AdamWOptimizerStrategy(lr=0.01),
    )
    trainer.set_client_id(7)

    trainer.train(tiny_dataset, list(range(len(tiny_dataset))))

    assert trainer.client_id in trainer._preserved_optimizer_states
    assert _cached_optimizer_step(trainer) == 1
    state_path = Path(Config.params["model_path"]) / trainer._optimizer_state_filename(
        Config.params["run_id"]
    )
    assert state_path.exists()
    assert "optimizer_state" not in trainer.obtain_model_update(
        {
            "batch_size": 4,
            "epochs": 1,
            "lr": 0.01,
            "run_id": "payload-check",
            "preserve_optimizer_state": True,
        },
        tiny_dataset,
        list(range(len(tiny_dataset))),
    )


def test_subprocess_optimizer_state_persists_across_rounds_for_same_client(
    temp_config, monkeypatch, tmp_path, tiny_dataset
):
    _configure_subprocess_training(monkeypatch, tmp_path, preserve_optimizer_state=True)
    trainer = ComposableTrainer(
        model=_linear_model,
        loss_strategy=CrossEntropyLossStrategy(),
        optimizer_strategy=AdamWOptimizerStrategy(lr=0.01),
    )
    trainer.set_client_id(7)

    trainer.train(tiny_dataset, list(range(len(tiny_dataset))))
    trainer.train(tiny_dataset, list(range(len(tiny_dataset))))

    assert _cached_optimizer_step(trainer) == 2


def test_subprocess_scheduler_state_persists_across_rounds(
    temp_config, monkeypatch, tmp_path, tiny_dataset
):
    _configure_subprocess_training(monkeypatch, tmp_path, preserve_optimizer_state=True)
    trainer = ComposableTrainer(
        model=_linear_model,
        loss_strategy=CrossEntropyLossStrategy(),
        optimizer_strategy=SGDOptimizerStrategy(lr=0.2),
        lr_scheduler_strategy=StepLRSchedulerStrategy(step_size=1, gamma=0.5),
    )
    trainer.set_client_id(3)

    trainer.train(tiny_dataset, list(range(len(tiny_dataset))))
    trainer.train(tiny_dataset, list(range(len(tiny_dataset))))

    payload = trainer._preserved_optimizer_states[trainer.client_id]
    assert _cached_scheduler_last_epoch(trainer) == 2
    assert payload["optimizer_state"]["param_groups"][0]["lr"] == pytest.approx(0.05)


def test_subprocess_missing_sidecar_clears_inherited_parent_cache(
    temp_config, monkeypatch, tmp_path, tiny_dataset, one_step_config
):
    _configure_subprocess_training(monkeypatch, tmp_path, preserve_optimizer_state=True)
    source_trainer = ComposableTrainer(
        model=_linear_model,
        loss_strategy=CrossEntropyLossStrategy(),
        optimizer_strategy=AdamWOptimizerStrategy(lr=0.01),
    )
    source_trainer.set_client_id(7)
    config = {
        **one_step_config,
        "run_id": Config.params["run_id"],
        "preserve_optimizer_state": True,
    }
    source_trainer.train_model(config, tiny_dataset, list(range(len(tiny_dataset))))
    assert _cached_optimizer_step(source_trainer) == 1

    trainer = ComposableTrainer(
        model=_linear_model,
        loss_strategy=CrossEntropyLossStrategy(),
        optimizer_strategy=AdamWOptimizerStrategy(lr=0.01),
    )
    trainer.set_client_id(7)
    trainer._preserved_optimizer_states[7] = copy.deepcopy(
        source_trainer._preserved_optimizer_states[7]
    )

    state_path = Path(Config.params["model_path"]) / trainer._optimizer_state_filename(
        Config.params["run_id"]
    )
    state_path.unlink(missing_ok=True)

    trainer.train(tiny_dataset, list(range(len(tiny_dataset))))

    assert _cached_optimizer_step(trainer) == 1


def test_missing_subprocess_output_removes_stale_input_sidecar(
    temp_config, monkeypatch, tmp_path, tiny_dataset, one_step_config
):
    _configure_subprocess_training(monkeypatch, tmp_path, preserve_optimizer_state=True)
    trainer = ComposableTrainer(
        model=_linear_model,
        loss_strategy=CrossEntropyLossStrategy(),
        optimizer_strategy=AdamWOptimizerStrategy(lr=0.01),
    )
    trainer.set_client_id(7)
    config = {
        **one_step_config,
        "run_id": Config.params["run_id"],
        "preserve_optimizer_state": True,
    }
    trainer.train_model(config, tiny_dataset, list(range(len(tiny_dataset))))

    input_filename = trainer._optimizer_state_filename(Config.params["run_id"])
    missing_output_filename = trainer._optimizer_state_output_filename(
        Config.params["run_id"]
    )
    assert trainer._save_preserved_optimizer_state_file(input_filename)
    input_path = Path(Config.params["model_path"]) / input_filename
    assert input_path.exists()

    trainer._finish_subprocess_optimizer_state(input_filename, missing_output_filename)

    assert trainer.client_id not in trainer._preserved_optimizer_states
    assert not input_path.exists()


def test_subprocess_invalid_optimizer_state_resets_safely(
    temp_config, monkeypatch, tmp_path, tiny_dataset
):
    _configure_subprocess_training(monkeypatch, tmp_path, preserve_optimizer_state=True)
    trainer = ComposableTrainer(
        model=_linear_model,
        loss_strategy=CrossEntropyLossStrategy(),
        optimizer_strategy=AdamWOptimizerStrategy(lr=0.01),
    )
    trainer.set_client_id(7)
    state_path = Path(Config.params["model_path"]) / trainer._optimizer_state_filename(
        Config.params["run_id"]
    )
    with open(state_path, "wb") as state_file:
        pickle.dump({"optimizer_type": torch.optim.SGD}, state_file)

    trainer.train(tiny_dataset, list(range(len(tiny_dataset))))

    payload = trainer._preserved_optimizer_states[trainer.client_id]
    assert payload["optimizer_type"] is torch.optim.AdamW
    assert _cached_optimizer_step(trainer) == 1


def test_subprocess_optimizer_state_is_not_persisted_when_disabled(
    temp_config, monkeypatch, tmp_path, tiny_dataset
):
    _configure_subprocess_training(
        monkeypatch, tmp_path, preserve_optimizer_state=False
    )
    trainer = ComposableTrainer(
        model=_linear_model,
        loss_strategy=CrossEntropyLossStrategy(),
        optimizer_strategy=AdamWOptimizerStrategy(lr=0.01),
    )
    trainer.set_client_id(7)

    trainer.train(tiny_dataset, list(range(len(tiny_dataset))))
    trainer.train(tiny_dataset, list(range(len(tiny_dataset))))

    assert trainer._preserved_optimizer_states == {}
    state_path = Path(Config.params["model_path"]) / trainer._optimizer_state_filename(
        Config.params["run_id"]
    )
    assert not state_path.exists()


def test_preserved_optimizer_state_is_local_to_logical_client(
    temp_config, tiny_dataset, one_step_config
):
    step_strategy = CapturingTrainingStep()
    trainer = ComposableTrainer(
        model=_linear_model,
        loss_strategy=CrossEntropyLossStrategy(),
        optimizer_strategy=AdamWOptimizerStrategy(lr=0.01),
        training_step_strategy=step_strategy,
    )
    config = {**one_step_config, "preserve_optimizer_state": True}

    trainer.set_client_id(1)
    trainer.train_model(config, tiny_dataset, list(range(len(tiny_dataset))))
    client1_state = copy.deepcopy(trainer.optimizer.state_dict()["state"])

    trainer.set_client_id(2)
    trainer.train_model(config, tiny_dataset, list(range(len(tiny_dataset))))

    trainer.set_client_id(1)
    trainer.train_model(config, tiny_dataset, list(range(len(tiny_dataset))))

    assert step_strategy.pre_step_states[0] == {}
    assert step_strategy.pre_step_states[1] == {}
    restored_state = _first_param_state(step_strategy.pre_step_states[2])
    saved_state = _first_param_state(client1_state)
    assert torch.allclose(restored_state["exp_avg"], saved_state["exp_avg"])


def test_preserved_state_stays_out_of_model_update_payload(
    temp_config, tiny_dataset, one_step_config
):
    trainer = ComposableTrainer(
        model=_linear_model,
        loss_strategy=CrossEntropyLossStrategy(),
        optimizer_strategy=AdamWOptimizerStrategy(lr=0.01),
        lr_scheduler_strategy=StepLRSchedulerStrategy(step_size=1, gamma=0.5),
    )
    trainer.set_client_id(5)
    config = {**one_step_config, "preserve_optimizer_state": True}

    update = trainer.obtain_model_update(
        config, tiny_dataset, list(range(len(tiny_dataset)))
    )
    preserved_state = trainer._preserved_optimizer_states[trainer.client_id]

    assert preserved_state["optimizer_state"]["state"]
    assert preserved_state["scheduler_state"]["last_epoch"] >= 1
    _assert_model_update_contains_only_model_weights(update, trainer.model)


def test_preserved_state_invalidates_when_parameter_order_changes(
    temp_config, tiny_dataset, one_step_config
):
    step_strategy = CapturingTrainingStep()
    trainer = ComposableTrainer(
        model=lambda: _two_layer_model("first", "second"),
        loss_strategy=CrossEntropyLossStrategy(),
        optimizer_strategy=AdamWOptimizerStrategy(lr=0.01),
        training_step_strategy=step_strategy,
    )
    config = {**one_step_config, "preserve_optimizer_state": True}

    trainer.train_model(config, tiny_dataset, list(range(len(tiny_dataset))))
    trainer.model = _two_layer_model("second", "first")
    trainer.context.model = trainer.model
    trainer.train_model(config, tiny_dataset, list(range(len(tiny_dataset))))

    assert step_strategy.pre_step_states[1] == {}


def test_preserved_state_invalidates_when_optimizer_type_changes(
    temp_config, tiny_dataset, one_step_config
):
    step_strategy = CapturingTrainingStep()
    trainer = ComposableTrainer(
        model=_linear_model,
        loss_strategy=CrossEntropyLossStrategy(),
        optimizer_strategy=AdamWOptimizerStrategy(lr=0.01),
        training_step_strategy=step_strategy,
    )
    config = {**one_step_config, "preserve_optimizer_state": True}

    trainer.train_model(config, tiny_dataset, list(range(len(tiny_dataset))))
    trainer.optimizer_strategy = SGDOptimizerStrategy(lr=0.1)
    trainer.train_model(config, tiny_dataset, list(range(len(tiny_dataset))))

    assert step_strategy.pre_step_states[1] == {}
    assert isinstance(trainer.optimizer, torch.optim.SGD)


def test_preserved_state_compatibility_rejects_shape_dtype_and_scheduler_changes(
    temp_config, tiny_dataset, one_step_config
):
    trainer = ComposableTrainer(
        model=_linear_model,
        loss_strategy=CrossEntropyLossStrategy(),
        optimizer_strategy=AdamWOptimizerStrategy(lr=0.01),
    )
    trainer.set_client_id(4)
    config = {**one_step_config, "preserve_optimizer_state": True}

    trainer.train_model(config, tiny_dataset, list(range(len(tiny_dataset))))
    payload = copy.deepcopy(trainer._preserved_optimizer_states[4])

    current_model = trainer.model
    current_optimizer = trainer.optimizer_strategy.create_optimizer(
        current_model, trainer.context
    )
    changed_scheduler = StepLRSchedulerStrategy(
        step_size=1, gamma=0.5
    ).create_scheduler(current_optimizer, trainer.context)
    assert not trainer._preserved_state_is_compatible(
        payload, current_model, current_optimizer, changed_scheduler
    )

    changed_shape_model = nn.Sequential(OrderedDict([("linear", nn.Linear(2, 3))]))
    changed_shape_optimizer = trainer.optimizer_strategy.create_optimizer(
        changed_shape_model, trainer.context
    )
    assert not trainer._preserved_state_is_compatible(
        payload, changed_shape_model, changed_shape_optimizer, None
    )

    changed_dtype_model = _linear_model().to(torch.float64)
    changed_dtype_optimizer = trainer.optimizer_strategy.create_optimizer(
        changed_dtype_model, trainer.context
    )
    assert not trainer._preserved_state_is_compatible(
        payload, changed_dtype_model, changed_dtype_optimizer, None
    )


@pytest.mark.parametrize("preserve_value", [None, False])
def test_optimizer_state_is_not_restored_when_disabled_or_unset(
    temp_config, tiny_dataset, one_step_config, preserve_value
):
    step_strategy = CapturingTrainingStep()
    trainer = ComposableTrainer(
        model=_linear_model,
        loss_strategy=CrossEntropyLossStrategy(),
        optimizer_strategy=AdamWOptimizerStrategy(lr=0.01),
        training_step_strategy=step_strategy,
    )
    config = dict(one_step_config)
    if preserve_value is not None:
        config["preserve_optimizer_state"] = preserve_value

    trainer.train_model(config, tiny_dataset, list(range(len(tiny_dataset))))
    trainer.train_model(config, tiny_dataset, list(range(len(tiny_dataset))))

    assert step_strategy.pre_step_states == [{}, {}]
    final_param_state = _first_param_state(trainer.optimizer.state_dict()["state"])
    assert _state_step(final_param_state) == 1
