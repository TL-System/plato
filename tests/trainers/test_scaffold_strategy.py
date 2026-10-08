"""Independent numerical SCAFFOLD equations and ownership boundaries (A01-A05/A09)."""

import copy
import logging
import pickle
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from plato.callbacks.trainer import TrainerCallback
from plato.config import Config
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms.scaffold_strategy import (
    SCAFFOLDUpdateStrategy,
    SCAFFOLDUpdateStrategyV2,
)
from plato.trainers.strategies.base import TrainingContext
from plato.trainers.strategies.loss_criterion import MSELossStrategy
from plato.trainers.strategies.optimizer import DefaultOptimizerStrategy
from plato.trainers.strategies.training_step import GradientAccumulationStepStrategy
from tests.integration.utils import build_minimal_config, configure_environment


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.double))

    def forward(self, examples):
        return self.theta.expand_as(examples)


def quadratic_loss(outputs, labels):
    return 0.5 * (outputs - labels).square().mean()


def scalar_controls(value):
    return OrderedDict(theta=torch.tensor([value], dtype=torch.double))


def prepare(context, strategy, optimizer, ci=1.0, c=2.0):
    strategy.setup(context)
    strategy.client_control_variate = scalar_controls(ci)
    context.state["server_control_variate"] = scalar_controls(c)
    strategy.on_train_start(context)
    context.state["optimizer"] = optimizer
    strategy.before_step(context)


@pytest.mark.parametrize(
    "strategy_type", [SCAFFOLDUpdateStrategy, SCAFFOLDUpdateStrategyV2]
)
@pytest.mark.parametrize(
    "gradients,expected_y,expected_ci", [([0.0], 0.9, 0.0), ([3.0, 1.0], 0.4, 2.0)]
)
def test_independent_main_equations(
    tmp_path, strategy_type, gradients, expected_y, expected_ci
):
    with configure_environment(build_minimal_config(), runtime_root=tmp_path):
        model = ScalarModel()
        context = TrainingContext()
        context.model, context.client_id = model, 1
        strategy = strategy_type()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        prepare(context, strategy, optimizer)
        inbound = copy.deepcopy(context.state["server_control_variate"])
        for gradient in gradients:
            model.theta.grad = torch.full_like(model.theta, gradient)
            optimizer.step()
            context.state["optimizer_step_completed"] = True
            strategy.after_step(context)
        strategy.on_train_end(context)
        strategy.on_train_result_accepted(context)
        assert model.theta.item() == pytest.approx(expected_y, abs=1e-12)
        assert strategy.client_control_variate["theta"].item() == pytest.approx(
            expected_ci, abs=1e-12
        )
        assert expected_ci == sum(gradients) / len(gradients)
        assert strategy.get_update_payload(context)["control_variate_delta"][
            "theta"
        ].item() == pytest.approx(expected_ci - 1, abs=1e-12)
        torch.testing.assert_close(
            context.state["server_control_variate"]["theta"], inbound["theta"]
        )
        strategy.on_train_cleanup(context, successful=True)
        assert not optimizer._optimizer_step_pre_hooks


@pytest.mark.parametrize("microbatches", [1, 3, 4])
@pytest.mark.parametrize(
    "strategy_type", [SCAFFOLDUpdateStrategy, SCAFFOLDUpdateStrategyV2]
)
def test_real_accumulation_updates_tail_and_empty_round(
    tmp_path, strategy_type, microbatches
):
    config = build_minimal_config()
    config["trainer"].update(batch_size=2, epochs=1)
    config["parameters"]["optimizer"]["lr"] = 0.1
    with configure_environment(config, runtime_root=tmp_path):
        strategy = strategy_type()
        trainer = ComposableTrainer(
            model=ScalarModel(),
            model_update_strategy=strategy,
            training_step_strategy=GradientAccumulationStepStrategy(2),
            loss_strategy=MSELossStrategy(),
        )
        assert isinstance(trainer.loss_strategy, MSELossStrategy)
        trainer.loss_strategy._criterion = quadratic_loss
        trainer.set_client_id(1)
        strategy.client_control_variate = scalar_controls(1.0)
        trainer.context.state["server_control_variate"] = scalar_controls(2.0)
        # Smaller last physical batch exercises equal microbatch weighting.
        size = microbatches * 2 - 1
        labels = torch.arange(size, dtype=torch.double).view(-1, 1) / 3
        data = TensorDataset(torch.ones_like(labels), labels)
        expected, gradients = 1.0, []
        for start in range(0, microbatches, 2):
            means = [
                labels[index * 2 : (index + 1) * 2].mean().item()
                for index in range(start, min(start + 2, microbatches))
            ]
            gradient = sum(expected - mean for mean in means) / len(means)
            gradients.append(gradient)
            expected -= 0.1 * (gradient - 1 + 2)
        run = {**config["trainer"], "run_id": "accum"}
        trainer.train_model(run, data, torch.utils.data.SequentialSampler(data))
        assert strategy.local_steps == (microbatches + 1) // 2
        assert trainer.model is not None
        assert trainer.model.theta.item() == pytest.approx(expected, abs=1e-12)
        assert strategy.client_control_variate["theta"].item() == pytest.approx(
            sum(gradients) / len(gradients), abs=1e-12
        )
        saved_ci = strategy.client_control_variate["theta"].clone()
        empty = TensorDataset(torch.empty(0, 1).double(), torch.empty(0, 1).double())
        trainer.train_model(run, empty, [])
        assert strategy.local_steps == 0
        torch.testing.assert_close(strategy.client_control_variate["theta"], saved_ci)
        assert (
            strategy.get_update_payload(trainer.context)["control_variate_delta"][
                "theta"
            ].item()
            == 0.0
        )


def test_skipped_update_is_not_corrected(tmp_path):
    with configure_environment(build_minimal_config(), runtime_root=tmp_path):
        context = TrainingContext()
        context.model, context.client_id = ScalarModel(), 1
        strategy = SCAFFOLDUpdateStrategy()
        optimizer = torch.optim.SGD(context.model.parameters(), lr=0.1)
        prepare(context, strategy, optimizer)
        context.state["optimizer_step_completed"] = False
        strategy.after_step(context)
        strategy.on_train_end(context)
        strategy.on_train_result_accepted(context)
        assert context.model.theta.item() == 1.0
        assert strategy.local_steps == 0
        assert strategy.client_control_variate is not None
        assert strategy.client_control_variate["theta"].item() == 1.0


@pytest.mark.parametrize("rate", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_executed_rate_fails_before_update(tmp_path, rate):
    with configure_environment(build_minimal_config(), runtime_root=tmp_path):
        context = TrainingContext()
        context.model, context.client_id = ScalarModel(), 1
        strategy = SCAFFOLDUpdateStrategy()
        optimizer = torch.optim.SGD(context.model.parameters(), lr=0.1)
        prepare(context, strategy, optimizer)
        optimizer.param_groups[0]["lr"] = rate
        context.model.theta.grad = torch.ones_like(context.model.theta)
        with pytest.raises(ValueError, match="SCAFFOLD.*positive"):
            optimizer.step()
        assert context.model.theta.item() == 1.0
        assert strategy.client_control_variate_path is not None
        assert not Path(strategy.client_control_variate_path).exists()
        strategy.on_train_cleanup(context, successful=False)
        with pytest.raises(RuntimeError, match="successful"):
            strategy.get_update_payload(context)


def test_missing_optimizer_and_post_step_rate_capture_then_tail_rejection(tmp_path):
    config = build_minimal_config()
    config["trainer"].update(batch_size=1, epochs=1)
    config["parameters"]["optimizer"]["lr"] = 0.1

    class ChangeRate(DefaultOptimizerStrategy):
        def on_optimizer_step(self, optimizer, context):
            optimizer.param_groups[0]["lr"] = 0.025

    with configure_environment(config, runtime_root=tmp_path):
        strategy = SCAFFOLDUpdateStrategy()
        trainer = ComposableTrainer(
            model=ScalarModel(),
            model_update_strategy=strategy,
            optimizer_strategy=ChangeRate(),
            training_step_strategy=GradientAccumulationStepStrategy(2),
            loss_strategy=MSELossStrategy(),
        )
        trainer.set_client_id(1)
        with pytest.raises(ValueError, match="actual optimizer"):
            strategy.before_step(trainer.context)
        strategy.client_control_variate = scalar_controls(1.0)
        trainer.context.state["server_control_variate"] = scalar_controls(2.0)
        assert isinstance(trainer.loss_strategy, MSELossStrategy)
        trainer.loss_strategy._criterion = lambda outputs, labels: outputs.sum() * 0
        data = TensorDataset(torch.ones(3, 1).double(), torch.zeros(3, 1).double())
        with pytest.raises(ValueError, match="constant within"):
            trainer.train_model(
                {**config["trainer"], "run_id": "rate"}, data, [0, 1, 2]
            )
        # First correction uses the pre-step .1 despite the post-step .025.
        assert trainer.model is not None
        assert trainer.model.theta.item() == pytest.approx(0.9)
        assert strategy.local_steps == 1
        assert strategy.client_control_variate_path is not None
        assert not Path(strategy.client_control_variate_path).exists()
        assert trainer.context.state.get("client_control_variate_delta") is None
        assert trainer.optimizer is not None
        assert not trainer.optimizer._optimizer_step_pre_hooks


def test_final_tail_denominator_uses_executed_rate_before_post_hook(tmp_path):
    class ChangeRate(DefaultOptimizerStrategy):
        def on_optimizer_step(self, optimizer, context):
            optimizer.param_groups[0]["lr"] = 0.025

    config = build_minimal_config()
    config["parameters"]["optimizer"]["lr"] = 0.1
    with configure_environment(config, runtime_root=tmp_path):
        strategy = SCAFFOLDUpdateStrategy()
        trainer = ComposableTrainer(
            model=ScalarModel(),
            model_update_strategy=strategy,
            optimizer_strategy=ChangeRate(),
            training_step_strategy=GradientAccumulationStepStrategy(2),
            loss_strategy=MSELossStrategy(),
        )
        trainer.set_client_id(1)
        strategy.client_control_variate = scalar_controls(1.0)
        trainer.context.state["server_control_variate"] = scalar_controls(2.0)
        assert isinstance(trainer.loss_strategy, MSELossStrategy)
        trainer.loss_strategy._criterion = lambda outputs, labels: outputs.sum() * 0
        data = TensorDataset(torch.ones(1, 1).double(), torch.zeros(1, 1).double())
        trainer.train_model({**config["trainer"], "run_id": "tail"}, data, [0])
        assert trainer.model is not None
        assert trainer.model.theta.item() == pytest.approx(0.9)
        assert trainer.optimizer is not None
        assert trainer.optimizer.param_groups[0]["lr"] == 0.025
        assert strategy.learning_rate == 0.1 and strategy.local_steps == 1
        assert strategy.client_control_variate["theta"].item() == pytest.approx(
            0.0, abs=1e-12
        )


class MixedParameters(ScalarModel):
    floating: torch.Tensor
    integer: torch.Tensor

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.double))
        self.bias = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.double))
        self.unused = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.double))
        self.excluded = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.double))
        self.frozen = torch.nn.Parameter(
            torch.tensor([1.0], dtype=torch.double), requires_grad=False
        )
        self.register_buffer("floating", torch.tensor([7.0], dtype=torch.double))
        self.register_buffer("integer", torch.tensor([5]))


def test_parameter_identity_reordered_equal_groups_buffers_and_exclusions(tmp_path):
    with configure_environment(build_minimal_config(), runtime_root=tmp_path):
        model = MixedParameters()
        context = TrainingContext()
        context.model, context.client_id = model, 1
        strategy = SCAFFOLDUpdateStrategy()
        strategy.setup(context)
        # Historical full-state controls include frozen parameters and buffers.
        historical = OrderedDict(
            (name, torch.ones_like(value).double())
            for name, value in model.state_dict().items()
        )
        strategy.client_control_variate = historical
        context.state["server_control_variate"] = OrderedDict(
            (name, value * 2) for name, value in historical.items()
        )
        inbound = copy.deepcopy(context.state["server_control_variate"])
        strategy.on_train_start(context)
        optimizer = torch.optim.SGD(
            [
                {"params": [model.bias, model.unused], "lr": 0.1},
                {"params": [model.weight, model.theta], "lr": 0.1},
            ]
        )
        context.state["optimizer"] = optimizer
        strategy.before_step(context)
        for parameter in (model.theta, model.weight, model.bias):
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        strategy.after_step(context)
        strategy.on_train_end(context)
        strategy.on_train_result_accepted(context)
        for parameter in (model.theta, model.weight, model.bias, model.unused):
            assert parameter.item() == pytest.approx(0.9)
        assert model.excluded.item() == model.frozen.item() == 1.0
        assert model.floating.item() == 7.0 and model.integer.item() == 5
        delta = strategy.get_update_payload(context)["control_variate_delta"]
        assert set(delta) == {"theta", "weight", "bias", "unused", "excluded"}
        assert delta["excluded"].item() == 0.0
        for name, value in inbound.items():
            torch.testing.assert_close(
                context.state["server_control_variate"][name], value
            )
        strategy.on_train_cleanup(context, successful=True)
        # Different constant LR is allowed in a later round; heterogeneous is not.
        strategy.on_train_start(context)
        optimizer.param_groups[0]["lr"] = 0.025
        context.state["optimizer"] = optimizer
        strategy.before_step(context)
        before = model.theta.clone()
        with pytest.raises(ValueError, match="equal positive LR"):
            optimizer.step()
        torch.testing.assert_close(model.theta, before)


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"theta": torch.zeros(2).double()},
        {"theta": torch.tensor([float("nan")]).double()},
        {"theta": torch.zeros(1).double(), "unknown": torch.zeros(1)},
    ],
)
def test_invalid_controls_fail_before_mutation(tmp_path, bad):
    with configure_environment(build_minimal_config(), runtime_root=tmp_path):
        context = TrainingContext()
        context.model, context.client_id = ScalarModel(), 1
        strategy = SCAFFOLDUpdateStrategy()
        strategy.setup(context)
        context.state["server_control_variate"] = bad
        with pytest.raises(ValueError, match="SCAFFOLD control"):
            strategy.on_train_start(context)
        assert context.model.theta.item() == 1.0


@pytest.mark.parametrize("optimizer_name", ["momentum", "adam"])
@pytest.mark.parametrize(
    "strategy_type", [SCAFFOLDUpdateStrategy, SCAFFOLDUpdateStrategyV2]
)
def test_generic_optimizer_composition_matches_independent_raw_updates(
    tmp_path, caplog, optimizer_name, strategy_type
):
    with configure_environment(build_minimal_config(), runtime_root=tmp_path):
        context = TrainingContext()
        context.model, context.client_id = ScalarModel(), 1
        reference = copy.deepcopy(context.model)
        factory = lambda params: (
            torch.optim.SGD(params, lr=0.01, momentum=0.9)
            if optimizer_name == "momentum"
            else torch.optim.Adam(params, lr=0.01)
        )
        optimizer, raw_optimizer = (
            factory(context.model.parameters()),
            factory(reference.parameters()),
        )
        strategy = strategy_type()
        with caplog.at_level(logging.WARNING):
            prepare(context, strategy, optimizer)
            for gradient in (3.0, 1.0):
                context.model.theta.grad = torch.full_like(
                    context.model.theta, gradient
                )
                reference.theta.grad = torch.full_like(reference.theta, gradient)
                raw_optimizer.step()
                with torch.no_grad():
                    reference.theta.sub_(0.01 * (2 - 1))
                optimizer.step()
                strategy.after_step(context)
            strategy.on_train_end(context)
            strategy.on_train_result_accepted(context)
        torch.testing.assert_close(
            context.model.theta, reference.theta, atol=1e-12, rtol=1e-12
        )
        expected_ci = 1 - 2 + (1 - reference.theta.item()) / (0.01 * 2)
        assert strategy.client_control_variate["theta"].item() == pytest.approx(
            expected_ci, abs=1e-12
        )
        assert caplog.text.count("additive-control optimizer extension") == 1


@pytest.mark.parametrize("legacy", ["example", "prefix", "canonical"])
def test_exact_legacy_load_canonical_precedence_and_client_zero_isolation(
    tmp_path, legacy
):
    config = build_minimal_config(model_name="org/model")
    with configure_environment(config, runtime_root=tmp_path):
        root = tmp_path / "personal"
        root.mkdir()

        def dump(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as state_file:
                pickle.dump(scalar_controls(value), state_file)

        dump(root / "scaffold_cv_0.pkl", 99.0)
        dump(Path(str(root) + "scaffold_cv_0.pkl"), 98.0)
        paths = {
            "example": root / "org/model_1_control_variate.pth",
            "prefix": Path(str(root) + "scaffold_cv_1.pkl"),
            "canonical": root / "scaffold_cv_1.pkl",
        }
        dump(paths[legacy], 3.0)
        strategy = SCAFFOLDUpdateStrategy(save_path=str(root))
        trainer = ComposableTrainer(model=ScalarModel(), model_update_strategy=strategy)
        assert strategy.client_control_variate is None
        trainer.set_client_id(1)
        assert strategy.client_control_variate is not None
        assert strategy.client_control_variate["theta"].item() == 3.0
        original = paths[legacy].read_bytes()
        trainer.set_client_id(2)
        assert strategy.client_control_variate is None
        trainer.set_client_id(1)
        assert strategy.client_control_variate is not None
        assert strategy.client_control_variate["theta"].item() == 3.0
        assert paths[legacy].read_bytes() == original
        dump(root / "scaffold_cv_1.pkl", 4.0)
        trainer.set_client_id(2)
        trainer.set_client_id(1)
        assert strategy.client_control_variate is not None
        assert strategy.client_control_variate["theta"].item() == 4.0
        assert strategy.client_control_variate_path is not None
        assert Path(strategy.client_control_variate_path).parent == root


def test_persistence_and_worker_state_failures_cannot_emit_old_delta(tmp_path):
    with configure_environment(build_minimal_config(), runtime_root=tmp_path):
        context = TrainingContext()
        context.model, context.client_id = ScalarModel(), 1
        strategy = SCAFFOLDUpdateStrategy()
        optimizer = torch.optim.SGD(context.model.parameters(), lr=0.1)
        prepare(context, strategy, optimizer)
        strategy.on_train_end(context)
        strategy.on_train_result_accepted(context)
        strategy.on_train_start(context)
        strategy.client_control_variate_path = str(tmp_path)  # directory, not a file
        with pytest.raises(OSError):
            strategy.on_train_end(context)
            strategy.on_train_result_accepted(context)
        with pytest.raises(RuntimeError, match="successful"):
            strategy.get_update_payload(context)
        with pytest.raises(ValueError, match="missing"):
            strategy.load_worker_state(None, context)
        (tmp_path / "models/scaffold_cv_2.pkl").write_bytes(b"corrupt")
        context.client_id = 2
        with pytest.raises(pickle.UnpicklingError):
            strategy.on_client_id_changed(context)


@pytest.mark.parametrize("existing_checkpoint", [False, True])
def test_direct_callback_failure_keeps_only_accepted_controls(
    tmp_path, existing_checkpoint
):
    class InterruptAtEnd(TrainerCallback):
        def on_train_run_end(self, trainer, config, **kwargs):
            raise RuntimeError("Deliberate post-training callback interruption")

    config = build_minimal_config()
    config["trainer"].update(batch_size=2, epochs=2)
    with configure_environment(config, runtime_root=tmp_path):
        strategy = SCAFFOLDUpdateStrategy()
        trainer = ComposableTrainer(
            model=ScalarModel(), callbacks=[InterruptAtEnd()],
            model_update_strategy=strategy,
            loss_strategy=MSELossStrategy(),
        )
        setattr(
            trainer.loss_strategy, "compute_loss",
            lambda outputs, labels, context: quadratic_loss(outputs, labels),
        )
        trainer.set_client_id(1)
        trainer.device = "cpu"
        trainer.context.device = torch.device("cpu")
        assert trainer.model is not None
        strategy.client_control_variate = scalar_controls(1.0)
        trainer.context.state["server_control_variate"] = scalar_controls(2.0)
        assert strategy.client_control_variate_path is not None
        canonical = Path(strategy.client_control_variate_path)
        if existing_checkpoint:
            canonical.write_bytes(pickle.dumps(scalar_controls(1.0)))
        original = canonical.read_bytes() if existing_checkpoint else None
        data = TensorDataset(torch.ones(2, 1).double(), torch.zeros(2, 1).double())
        with pytest.raises(RuntimeError, match="callback interruption"):
            trainer.train_model({**config["trainer"], "run_id": "direct-failure"},
                                data, [0, 1])
        assert strategy.client_control_variate["theta"].item() == 1.0
        assert canonical.exists() is existing_checkpoint
        if existing_checkpoint:
            assert canonical.read_bytes() == original
        with pytest.raises(RuntimeError, match="successful"):
            strategy.get_update_payload(trainer.context)
        assert not list(canonical.parent.glob(".scaffold_cv_*"))
