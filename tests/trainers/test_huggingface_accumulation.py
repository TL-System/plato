"""Numerical regression coverage for full and partial HF accumulation windows."""

from types import SimpleNamespace

import pytest
import torch

from plato.trainers.huggingface import HuggingFaceTrainingStepStrategy
from plato.trainers.strategies.base import TrainingContext


class RegressionModel(torch.nn.Module):
    """A differentiable HF-style loss whose update has an independent oracle."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))

    def forward(self, input_ids, labels, **kwargs):
        return SimpleNamespace(loss=((self.weight * input_ids - labels) ** 2).mean())


@pytest.mark.parametrize("tail_at_last_batch", [True, False])
@pytest.mark.parametrize("batches", [2, 3, 5])
def test_accumulation_matches_mean_loss_windows(batches, tail_at_last_batch):
    model = RegressionModel()
    oracle = RegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.4)
    oracle_optimizer = torch.optim.SGD(oracle.parameters(), lr=0.03, momentum=0.4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    oracle_scheduler = torch.optim.lr_scheduler.StepLR(oracle_optimizer, 1, gamma=0.9)
    events = []

    class Callbacks:
        def _hf_on_pre_optimizer_step(self):
            events.append("pre")

        def _hf_on_optimizer_step(self):
            events.append("post")

    context = TrainingContext()
    context.device = torch.device("cpu")
    context.state["hf_trainer"] = Callbacks()
    strategy = HuggingFaceTrainingStepStrategy(2)
    strategy.setup(context)
    oracle_losses = []
    actual_updates = 0
    for index in range(batches):
        inputs = torch.tensor([index + 1.0], dtype=torch.float64)
        labels = torch.tensor([0.5 - index], dtype=torch.float64)
        context.state["is_last_batch"] = tail_at_last_batch and index == batches - 1
        loss = strategy.training_step(
            model, optimizer, {"input_ids": inputs}, labels, None, context
        )
        assert torch.isfinite(loss)
        oracle_losses.append(oracle(inputs, labels).loss)
        expected_update = (index + 1) % 2 == 0 or context.state["is_last_batch"]
        assert context.state["optimizer_step_completed"] is expected_update
        if expected_update:
            torch.stack(oracle_losses).mean().backward()
            oracle_optimizer.step()
            oracle_optimizer.zero_grad()
            oracle_losses = []
            scheduler.step()
            oracle_scheduler.step()
            actual_updates += 1
            torch.testing.assert_close(model.weight, oracle.weight)

    finalized = strategy.finalize(model, optimizer, context)
    if oracle_losses:
        assert finalized is not None
        assert context.state["optimizer_step_completed"] is True
        torch.stack(oracle_losses).mean().backward()
        oracle_optimizer.step()
        oracle_scheduler.step()
        scheduler.step()
        actual_updates += 1
    else:
        assert finalized is None
    torch.testing.assert_close(model.weight, oracle.weight)
    assert events == [event for _ in range(actual_updates) for event in ("pre", "post")]
    assert context.state["hf_optimizer_step_index"] == actual_updates
    assert scheduler.last_epoch == oracle_scheduler.last_epoch == actual_updates
    assert context.state["grad_accum_counter"] == 0
    assert strategy.finalize(model, optimizer, context) is None
    assert context.state["optimizer_step_completed"] is False
    assert strategy.optimizer_steps_per_epoch(batches) == (batches + 1) // 2


@pytest.mark.parametrize("stop_after_tail", [False, True])
def test_real_hf_trainer_tail_callbacks_scheduler_and_control_flags(
    tmp_path,
    stop_after_tail,
):
    from transformers import TrainerCallback

    from plato.trainers.huggingface import Trainer
    from plato.trainers.strategies.lr_scheduler import DefaultLRSchedulerStrategy
    from tests.integration.utils import configure_environment
    from tests.test_utils.qwen3 import create_tiny_qwen3, reference_config

    directory = create_tiny_qwen3(tmp_path / "model")
    config = reference_config(directory)
    config["trainer"].update(
        gradient_accumulation_steps=2,
        optimizer="SGD",
        epochs=2 if stop_after_tail else 1,
    )
    config["parameters"]["optimizer"] = {"lr": 0.03, "momentum": 0.4}
    dataset = [
        {
            "input_ids": [1, index + 2],
            "attention_mask": [1, 1],
            "labels": [0, index + 1],
        }
        for index in range(5)
    ]
    events = []

    class TailControl(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            events.append("pre")

        def on_optimizer_step(self, args, state, control, **kwargs):
            events.append("post")

        def on_step_end(self, args, state, control, **kwargs):
            events.append("end")
            control.should_log = True
            if stop_after_tail and state.global_step == 3:
                control.should_training_stop = True
            return control

        def on_log(self, args, state, control, **kwargs):
            events.append("log")

    with configure_environment(config):
        model, oracle = RegressionModel(), RegressionModel()
        trainer = Trainer(model=model, callbacks=[TailControl()])
        trainer.lr_scheduler_strategy = DefaultLRSchedulerStrategy(
            scheduler_fn=lambda optimizer: torch.optim.lr_scheduler.StepLR(
                optimizer,
                1,
                gamma=0.9,
            )
        )
        oracle_optimizer = torch.optim.SGD(oracle.parameters(), lr=0.03, momentum=0.4)
        for offset in range(0, len(dataset), 2):
            losses = []
            for row in dataset[offset : offset + 2]:
                losses.append(
                    oracle(
                        torch.tensor(row["input_ids"], dtype=torch.float64),
                        torch.tensor(row["labels"], dtype=torch.float64),
                    ).loss
                )
            torch.stack(losses).mean().backward()
            oracle_optimizer.step()
            oracle_optimizer.zero_grad()
        trainer.train_model(config["trainer"], dataset, None)
        torch.testing.assert_close(model.weight, oracle.weight)
        assert events == [
            event for _ in range(3) for event in ("pre", "post", "end", "log")
        ]
        assert trainer._hf_state.global_step == 3
        assert trainer.context.state["optimizer_updates_per_epoch"] == 3
        assert trainer.lr_scheduler is not None
        assert trainer.optimizer is not None
        assert trainer.lr_scheduler.last_epoch == 1
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.027)
        assert trainer.current_epoch == 1
        assert not any(trainer._consume_control_flags().values())
