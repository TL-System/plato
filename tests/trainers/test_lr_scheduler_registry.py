"""Tests for the learning rate scheduler registry utilities."""

import pytest
import torch

from plato.trainers import lr_schedulers


def _build_optimizer(lr: float = 0.1):
    model = torch.nn.Linear(2, 2)
    return torch.optim.SGD(model.parameters(), lr=lr)


def _current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def test_step_lr_scheduler_decays_at_step():
    """StepLR should decay after the configured number of steps."""
    optimizer = _build_optimizer(lr=0.1)
    scheduler = lr_schedulers.get(
        optimizer,
        iterations_per_epoch=10,
        lr_scheduler="StepLR",
        lr_params={"step_size": 5, "gamma": 0.5},
    )

    for _ in range(5):
        scheduler.step()

    assert _current_lr(optimizer) == pytest.approx(0.05)

    for _ in range(5):
        scheduler.step()

    assert _current_lr(optimizer) == pytest.approx(0.025)


def test_multi_step_scheduler_applies_milestones():
    """MultiStepLR should drop at each milestone."""
    optimizer = _build_optimizer(lr=0.2)
    scheduler = lr_schedulers.get(
        optimizer,
        iterations_per_epoch=5,
        lr_scheduler="MultiStepLR",
        lr_params={"gamma": 0.1, "milestone_steps": "2ep,4ep"},
    )

    for step in range(10):
        scheduler.step()
        if step == 1:
            assert _current_lr(optimizer) == pytest.approx(0.02)
        if step == 3:
            assert _current_lr(optimizer) == pytest.approx(0.002)


def test_lambda_scheduler_supports_linear_warmup():
    """LambdaLR should ramp up during warmup iterations."""
    optimizer = _build_optimizer(lr=0.08)
    scheduler = lr_schedulers.get(
        optimizer,
        iterations_per_epoch=10,
        lr_scheduler="LambdaLR",
        lr_params={"gamma": 1.0, "warmup_steps": "3it"},
    )

    # The warmup schedule should scale lr linearly until reaching base lr.
    scheduler.step()
    assert _current_lr(optimizer) == pytest.approx(0.08 / 3)

    scheduler.step()
    assert _current_lr(optimizer) == pytest.approx(2 * 0.08 / 3)

    scheduler.step()
    assert _current_lr(optimizer) == pytest.approx(0.08)

    # Subsequent steps should keep the base lr (gamma=1.0).
    for _ in range(5):
        scheduler.step()
    assert _current_lr(optimizer) == pytest.approx(0.08)
