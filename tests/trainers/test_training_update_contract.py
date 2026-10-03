"""Numerical regressions for local optimizer updates and round state."""

import copy
import math

import pytest
import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.data import TensorDataset

from plato.callbacks.trainer import TrainerCallback
from plato.trainers.basic import TrainerWithTimmScheduler
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms.fedprox_strategy import FedProxLossStrategy
from plato.trainers.strategies.base import TrainingContext
from plato.trainers.strategies.loss_criterion import (
    CompositeLossStrategy,
    MSELossStrategy,
)
from plato.trainers.strategies.optimizer import (
    GradientClippingOptimizerStrategy,
    SGDOptimizerStrategy,
)
from plato.trainers.strategies.training_step import (
    GradientAccumulationStepStrategy,
    ValidateBeforeStepStrategy,
)
from tests.integration.utils import build_minimal_config, configure_environment


@pytest.mark.parametrize("sizes,window", [([2, 2, 1], 2), ([2, 1], 4), ([2, 2], 2)])
def test_accumulation_matches_mean_of_microbatch_losses(sizes, window):
    """Retain equal microbatch weighting, including a smaller final batch."""
    torch.manual_seed(12)
    model = torch.nn.Linear(2, 1).double()
    reference = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    context = TrainingContext()
    context.model = model
    strategy = GradientAccumulationStepStrategy(window)
    strategy.setup(context)
    batches = [
        (torch.randn(size, 2).double(), torch.randn(size, 1).double()) for size in sizes
    ]
    for start in range(0, len(batches), window):
        chunk = batches[start : start + window]
        reference_optimizer.zero_grad()
        reference_loss = sum(
            torch.nn.functional.mse_loss(reference(x), y) for x, y in chunk
        ) / len(chunk)
        reference_loss.backward()
        reference_optimizer.step()
    flags = []
    for x, y in batches:
        strategy.training_step(
            model, optimizer, x, y, torch.nn.functional.mse_loss, context
        )
        flags.append(context.state.get("optimizer_step_completed"))
    assert flags == [(i + 1) % window == 0 for i in range(len(batches))]
    strategy.finalize(model, optimizer, context)
    assert context.state["optimizer_step_completed"] == (len(batches) % window != 0)
    strategy.finalize(model, optimizer, context)
    assert context.state["optimizer_step_completed"] is False
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        assert actual.grad is None or torch.count_nonzero(actual.grad) == 0


class UpdateRecorder(TrainerCallback):
    def __init__(self):
        self.updates = []

    def on_train_step_end(self, trainer, config, batch, loss, **kwargs):
        self.updates.append((trainer.current_epoch, batch))


def test_accumulation_run_epoch_and_exception_cleanup(tmp_path):
    config = build_minimal_config()
    config["trainer"].update(batch_size=2, epochs=2)
    with configure_environment(config, runtime_root=tmp_path):
        recorder = UpdateRecorder()
        strategy = GradientAccumulationStepStrategy(2)
        trainer = ComposableTrainer(
            model=torch.nn.Linear(2, 1),
            callbacks=[recorder],
            training_step_strategy=strategy,
            loss_strategy=MSELossStrategy(),
            optimizer_strategy=SGDOptimizerStrategy(lr=0.01),
        )
        trainer.device = trainer.context.device = torch.device("cpu")
        dataset = TensorDataset(torch.ones(5, 2), torch.zeros(5, 1))
        run = {**config["trainer"], "run_id": "updates"}
        for _ in range(2):
            trainer.train_model(run, dataset, list(range(5)))
        assert recorder.updates == [(1, 1), (1, 2), (2, 1), (2, 2)] * 2
        original = trainer.loss_strategy.compute_loss
        calls = 0

        def interrupt(outputs, labels, context):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("interrupted accumulation")
            return original(outputs, labels, context)

        trainer.loss_strategy.compute_loss = interrupt
        with pytest.raises(RuntimeError, match="interrupted accumulation"):
            trainer.train_model(run, dataset, list(range(5)))
        assert strategy.current_step == 0
        assert all(
            p.grad is None or torch.count_nonzero(p.grad) == 0
            for p in trainer.model.parameters()
        )
        trainer.loss_strategy.compute_loss = original
        trainer.train_model(run, dataset, list(range(5)))
        assert recorder.updates[-4:] == [(1, 1), (1, 2), (2, 1), (2, 2)]


@pytest.mark.parametrize("mu,norm", [(0.0, "l2"), (0.1, "l2"), (0.1, "l1")])
def test_fedprox_origin_and_nonzero_analytical_gradient(mu, norm):
    model = torch.nn.Linear(2, 1, bias=False).double()
    context = TrainingContext()
    context.model = model
    strategy = FedProxLossStrategy(
        mu, base_loss_fn=torch.nn.functional.mse_loss, norm_type=norm
    )
    strategy.setup(context)
    x, y = torch.tensor([[1.0, 2.0]], dtype=torch.double), torch.zeros(1, 1).double()
    base = torch.nn.functional.mse_loss(model(x), y)
    base_grad = torch.autograd.grad(base, model.weight)[0]
    total = strategy.compute_loss(model(x), y, context)
    origin_grad = torch.autograd.grad(total, model.weight)[0]
    torch.testing.assert_close(origin_grad, base_grad)
    with torch.no_grad():
        model.weight.add_(torch.tensor([[3.0, 4.0]], dtype=torch.double))
    base = torch.nn.functional.mse_loss(model(x), y)
    base_grad = torch.autograd.grad(base, model.weight)[0]
    total = strategy.compute_loss(model(x), y, context)
    delta = torch.tensor([[3.0, 4.0]], dtype=torch.double)
    expected_penalty = mu * 2.5 if norm == "l2" else mu * 7
    expected_grad = mu / 2 * delta / 5 if norm == "l2" else mu * delta.sign()
    assert (total - base).item() == pytest.approx(expected_penalty)
    torch.testing.assert_close(
        torch.autograd.grad(total, model.weight)[0], base_grad + expected_grad
    )


@pytest.mark.parametrize("mu", [-0.1, float("nan"), float("inf")])
def test_fedprox_rejects_invalid_penalty(mu):
    with pytest.raises(ValueError):
        FedProxLossStrategy(mu)


@pytest.mark.parametrize("composed", [False, True])
def test_fedprox_refreshes_received_weights_each_run(tmp_path, composed):
    class Observe(FedProxLossStrategy):
        def __init__(self):
            super().__init__(0.1)
            self.penalties = []

        def compute_loss(self, outputs, labels, context):
            total = super().compute_loss(outputs, labels, context)
            self.penalties.append(
                (total - torch.nn.functional.cross_entropy(outputs, labels)).item()
            )
            return total

    config = build_minimal_config()
    config["trainer"].update(batch_size=2, epochs=1)
    with configure_environment(config, runtime_root=tmp_path):
        strategy = Observe()
        trainer = ComposableTrainer(
            model=torch.nn.Linear(2, 2),
            loss_strategy=(CompositeLossStrategy([strategy]) if composed else strategy),
        )
        trainer.device = trainer.context.device = torch.device("cpu")
        data = TensorDataset(torch.eye(2).repeat(2, 1), torch.tensor([0, 1, 0, 1]))
        for value in (2.0, 3.0):
            with torch.no_grad():
                trainer.model.weight.fill_(value)
            trainer.train_model(
                {**config["trainer"], "run_id": "prox"}, data, list(range(4))
            )
            assert strategy.penalties[-2] == 0
            assert strategy.penalties[-1] > 0
            assert torch.isfinite(trainer.model.weight).all()


@pytest.mark.parametrize("window,batches", [(1, 10), (3, 5)])
@pytest.mark.parametrize("global_schedule", [False, True])
def test_timm_real_lr_sequence_epochs_rounds_and_accumulation(
    tmp_path, monkeypatch, window, batches, global_schedule
):
    from plato.trainers import lr_schedulers

    observations = []
    epoch_steps = []

    class RecordingCosine(CosineLRScheduler):
        def step_update(self, num_updates, metric=None):
            super().step_update(num_updates, metric)
            observations.append((num_updates, self.optimizer.param_groups[0]["lr"]))

        def step(self, epoch, metric=None):
            epoch_steps.append(epoch)
            super().step(epoch, metric)

    def factory(optimizer, iterations_per_epoch):
        return RecordingCosine(optimizer, t_initial=100, lr_min=0.0, t_in_epochs=False)

    monkeypatch.setattr(lr_schedulers, "get", factory)
    config = build_minimal_config()
    config["trainer"].update(
        batch_size=2, epochs=2, lr_scheduler="timm", global_lr_scheduler=global_schedule
    )
    updates = math.ceil(batches / window)
    with configure_environment(config, runtime_root=tmp_path):
        trainer = TrainerWithTimmScheduler(model=torch.nn.Linear(2, 2))
        trainer.training_step_strategy = GradientAccumulationStepStrategy(window)
        trainer.device = trainer.context.device = torch.device("cpu")
        data = TensorDataset(torch.ones(batches * 2, 2), torch.arange(batches * 2) % 2)
        for round_id in (1, 3):
            trainer.current_round = round_id
            observations.clear()
            epoch_steps.clear()
            trainer.train_model(
                {**config["trainer"], "run_id": "timm"}, data, list(range(len(data)))
            )
            offset = (round_id - 1) * 2 * updates if global_schedule else 0
            expected = list(range(offset + 1, offset + 2 * updates + 1))
            actual = [(n, lr) for n, lr in observations if n != offset]
            assert [n for n, _ in actual] == expected
            reference_lrs = [
                0.5 * 0.01 * (1 + math.cos(math.pi * n / 100)) for n in expected
            ]
            assert [lr for _, lr in actual] == pytest.approx(reference_lrs)
            assert epoch_steps[-2:] == [
                ((round_id - 1) * 2 if global_schedule else 0) + epoch
                for epoch in (1, 2)
            ]


def test_real_fedprox_reference_config_activates_proximal_training(
    tmp_path, monkeypatch
):
    import sys
    from pathlib import Path

    from plato.config import Config
    from plato.trainers import registry
    from tests.integration.utils import isolated_config_state

    config_path = (
        Path(__file__).resolve().parents[2] / "configs/MNIST/fedprox_lenet5.toml"
    )
    with isolated_config_state():
        monkeypatch.setenv("config_file", str(config_path))
        monkeypatch.setattr(sys, "argv", ["pytest", "-b", str(tmp_path), "--cpu"])
        config = Config()
        trainer = registry.get()
        assert isinstance(trainer.loss_strategy, FedProxLossStrategy)
        assert trainer.loss_strategy.mu == 0.1
        assert config.trainer.optimizer == "SGD"
        trainer.set_client_id(1)
        before = copy.deepcopy(trainer.model.state_dict())
        torch.manual_seed(19)
        data = TensorDataset(torch.randn(4, 1, 28, 28), torch.tensor([0, 1, 2, 3]))
        trainer.train_model(
            {**config.trainer._asdict(), "run_id": "reference"}, data, list(range(4))
        )
        assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
        assert any(
            not torch.equal(before[k], v) for k, v in trainer.model.state_dict().items()
        )


def test_validation_wrapper_retains_accumulation_and_checks_tail_gradients():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    context = TrainingContext()
    context.model = model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    actual_updates = []
    optimizer.register_step_post_hook(lambda *args: actual_updates.append(1))
    strategy = ValidateBeforeStepStrategy(
        base_strategy=GradientAccumulationStepStrategy(2), raise_on_error=True
    )
    strategy.setup(context)
    for _ in range(3):
        strategy.training_step(
            model,
            optimizer,
            torch.ones(1, 1),
            torch.zeros(1, 1),
            torch.nn.functional.mse_loss,
            context,
        )
    assert len(actual_updates) == 1
    assert model.weight.item() == pytest.approx(0.8)
    strategy.finalize(model, optimizer, context)
    assert len(actual_updates) == 2
    assert model.weight.item() == pytest.approx(0.64)
    strategy.training_step(
        model,
        optimizer,
        torch.ones(1, 1),
        torch.zeros(1, 1),
        torch.nn.functional.mse_loss,
        context,
    )
    model.weight.grad.fill_(float("nan"))
    with pytest.raises(ValueError, match="gradient"):
        strategy.finalize(model, optimizer, context)
    assert len(actual_updates) == 2
    assert model.weight.item() == pytest.approx(0.64)
    strategy.on_train_end(context)
    assert model.weight.grad is None


def test_fedmos_distinct_same_shape_parameters_keep_corresponding_global_reference():
    from plato.trainers.strategies.algorithms.fedmos_strategy import FedMosOptimizer

    model = torch.nn.ParameterList(
        [
            torch.nn.Parameter(torch.tensor([2.0], dtype=torch.double)),
            torch.nn.Parameter(torch.tensor([3.0], dtype=torch.double)),
        ]
    )
    global_model = torch.nn.ParameterList(
        [
            torch.nn.Parameter(torch.tensor([10.0], dtype=torch.double)),
            torch.nn.Parameter(torch.tensor([20.0], dtype=torch.double)),
        ]
    )
    optimizer = FedMosOptimizer(model.parameters(), lr=0.1, a=0.9, mu=0.5)
    for parameter in model:
        parameter.grad = torch.ones_like(parameter)
    optimizer.update_momentum()
    optimizer.step(global_model_params=global_model)
    # Preserve the existing sequential local/global update equation.
    assert [parameter.item() for parameter in model] == pytest.approx([5.95, 11.45])


def test_gradient_clipping_limits_real_updates_including_accumulation_tail(tmp_path):
    from plato.trainers.strategies.loss_criterion import DefaultLossCriterionStrategy

    config = build_minimal_config()
    config["trainer"].update(batch_size=1, epochs=1)
    with configure_environment(config, runtime_root=tmp_path):
        model = torch.nn.Linear(1, 1, bias=False).double()
        model.weight.data.fill_(1.0)
        recorder = UpdateRecorder()
        trainer = ComposableTrainer(
            model=model,
            callbacks=[recorder],
            training_step_strategy=GradientAccumulationStepStrategy(2),
            loss_strategy=DefaultLossCriterionStrategy(
                lambda outputs, labels: 10 * outputs.mean()
            ),
            optimizer_strategy=GradientClippingOptimizerStrategy(
                SGDOptimizerStrategy(lr=0.1), max_norm=1.0
            ),
        )
        data = TensorDataset(torch.ones(3, 1).double(), torch.zeros(3, 1).double())
        trainer.train_model({**config["trainer"], "run_id": "clip"}, data, [0, 1, 2])
        # Independent scalar clipping reference: gradient=10 on each real step,
        # torch's clipping coefficient is max_norm/(norm+1e-6).
        expected = 1 - 2 * 0.1 * 10 / (10 + 1e-6)
        assert model.weight.item() == pytest.approx(expected, abs=1e-12)
        assert len(recorder.updates) == 2


def test_loss_metrics_preserve_sample_weighting_without_retaining_batch_graphs():
    from plato.trainers.tracking import LossTracker

    tracker = LossTracker()
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    for target, size in [(0.0, 2), (1.0, 3), (4.0, 1)]:
        loss = (parameter - target).square()
        tracker.update(loss, size)
        loss.backward()
        assert tracker.total_loss.grad_fn is None
        assert tracker.loss_value.grad_fn is None
    assert tracker.average == pytest.approx((4 * 2 + 1 * 3 + 4 * 1) / 6)
    assert parameter.grad.item() == pytest.approx(2.0)


@pytest.mark.parametrize("family", ["fedper", "fedrep", "lgfedavg"])
def test_personalization_freezes_restore_model_ownership_after_failure(
    tmp_path, family
):
    from plato.trainers.strategies.algorithms.lgfedavg_strategy import (
        LGFedAvgStepStrategy,
    )
    from plato.trainers.strategies.algorithms.personalized_fl_strategy import (
        FedPerUpdateStrategy,
        FedRepUpdateStrategy,
    )

    config = build_minimal_config()
    config["trainer"].update(batch_size=2, epochs=2)
    with configure_environment(config, runtime_root=tmp_path):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        model[0].bias.requires_grad_(False)
        before = {
            name: parameter.requires_grad
            for name, parameter in model.named_parameters()
        }
        extra = {}
        if family == "fedper":
            extra["model_update_strategy"] = FedPerUpdateStrategy(["0."])
        elif family == "fedrep":
            extra["model_update_strategy"] = FedRepUpdateStrategy(
                ["0."], ["1."], local_epochs=1
            )
        else:
            extra["training_step_strategy"] = LGFedAvgStepStrategy(["0."], ["1."])
        trainer = ComposableTrainer(
            model=model, loss_strategy=MSELossStrategy(), **extra
        )
        trainer.current_round = 2 if family == "fedper" else 1
        original = trainer.loss_strategy.compute_loss

        def interrupt(*args):
            raise RuntimeError("interrupted freeze")

        trainer.loss_strategy.compute_loss = interrupt
        data = TensorDataset(torch.ones(2, 2), torch.zeros(2, 1))
        run = {**config["trainer"], "run_id": "freeze"}
        with pytest.raises(RuntimeError, match="interrupted freeze"):
            trainer.train_model(run, data, [0, 1])
        assert {
            name: parameter.requires_grad
            for name, parameter in model.named_parameters()
        } == before
        trainer.loss_strategy.compute_loss = original
        weights = copy.deepcopy(model.state_dict())
        trainer.train_model(run, data, [0, 1])
        assert {
            name: parameter.requires_grad
            for name, parameter in model.named_parameters()
        } == before
        torch.testing.assert_close(model[0].bias, weights["0.bias"])
        assert not torch.equal(model[1].weight, weights["1.weight"])


def test_lgfedavg_two_real_updates_dispatch_timm_between_passes(tmp_path, monkeypatch):
    from plato.trainers import lr_schedulers
    from plato.trainers.strategies.algorithms.lgfedavg_strategy import (
        LGFedAvgStepStrategy,
    )

    observed = []

    class RecordCosine(CosineLRScheduler):
        def step_update(self, num_updates, metric=None):
            super().step_update(num_updates, metric)
            observed.append(num_updates)

    monkeypatch.setattr(
        lr_schedulers,
        "get",
        lambda optimizer, iterations_per_epoch: RecordCosine(
            optimizer, t_initial=100, t_in_epochs=False
        ),
    )
    config = build_minimal_config()
    config["trainer"].update(batch_size=1, epochs=1, lr_scheduler="timm")
    with configure_environment(config, runtime_root=tmp_path):
        torch.manual_seed(9)
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        ).double()
        reference = copy.deepcopy(model)
        recorder = UpdateRecorder()
        trainer = TrainerWithTimmScheduler(model=model, callbacks=[recorder])
        trainer.training_step_strategy = LGFedAvgStepStrategy(["0."], ["1."])
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        reference_scheduler = CosineLRScheduler(
            reference_optimizer, t_initial=100, t_in_epochs=False
        )
        data = TensorDataset(torch.eye(2).double(), torch.tensor([0, 1]))
        updates = 0
        for x, label in data:
            for train_layer in (1, 0):
                for index, layer in enumerate(reference):
                    for parameter in layer.parameters():
                        parameter.requires_grad_(index == train_layer)
                reference_optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(
                    reference(x.unsqueeze(0)), label.unsqueeze(0)
                )
                loss.backward()
                reference_optimizer.step()
                updates += 1
                reference_scheduler.step_update(updates)
        trainer.train_model(
            {**config["trainer"], "run_id": "dual"},
            data,
            torch.utils.data.SequentialSampler(data),
        )
        assert [count for count in observed if count > 0] == [1, 2, 3, 4]
        assert len(recorder.updates) == 4
        assert trainer.context.state["optimizer_updates_per_epoch"] == 4
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        assert "complete_optimizer_step" not in trainer.context.state


def test_real_grad_scaler_overflow_does_not_dispatch_optimizer_update(tmp_path):
    """Qualify real CPU scaler completion; CUDA execution needs GPU hardware."""
    from plato.trainers.strategies.training_step import MixedPrecisionStepStrategy

    config = build_minimal_config()
    config["trainer"].update(batch_size=1, epochs=1)
    with configure_environment(config, runtime_root=tmp_path):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(1)
        recorder = UpdateRecorder()
        strategy = MixedPrecisionStepStrategy(enabled=False)
        trainer = ComposableTrainer(
            model=model, training_step_strategy=strategy, callbacks=[recorder],
        )
        # Inject the supported CPU scaler to exercise real overflow decisions
        # without claiming CUDA mixed-precision hardware qualification.
        strategy.enabled = True
        strategy.scaler = torch.amp.GradScaler("cpu")
        factor = [float("inf")]
        trainer.loss_strategy.compute_loss = (
            lambda output, labels, context:
            torch.nn.functional.mse_loss(output, labels) * factor[0]
        )
        data = TensorDataset(torch.ones(1, 1), torch.zeros(1, 1))
        run = {**config["trainer"], "run_id": "amp"}
        initial_scale = strategy.scaler.get_scale()
        trainer.train_model(run, data, [0])
        assert model.weight.item() == 1
        assert recorder.updates == []
        assert strategy.scaler.get_scale() < initial_scale
        factor[0] = 1.0
        trainer.train_model(run, data, [0])
        assert len(recorder.updates) == 1
        assert model.weight.item() == pytest.approx(0.98)
        assert not model.weight.grad.isnan().any()


def test_feddyn_all_zero_labels_preserve_zero_weight_fallback(tmp_path):
    """Keep the legacy objective; zero label sums must not make its guard NaN."""
    from plato.trainers.strategies.algorithms.feddyn_strategy import FedDynLossStrategy

    with configure_environment(build_minimal_config(), runtime_root=tmp_path):
        model = torch.nn.Linear(1, 1, bias=False).double()
        model.weight.data.fill_(2)
        context = TrainingContext()
        context.model = model
        strategy = FedDynLossStrategy(
            alpha=0.1, base_loss_fn=lambda output, labels: output.sum() * 0,
        )
        strategy.setup(context)
        labels = torch.zeros(2, dtype=torch.int64)
        loss = strategy.compute_loss(model(torch.ones(2, 1).double()), labels, context)
        loss.backward()
        # The preserved shifted-quadratic/linear legacy equation gives -.2
        # here. This test makes no paper-equivalence claim.
        assert torch.isfinite(loss)
        assert model.weight.grad.item() == pytest.approx(-0.2)
        coefficient = strategy._get_alpha_coefficient(torch.tensor([0, 1]), context)
        assert coefficient.item() == pytest.approx(0.075)
