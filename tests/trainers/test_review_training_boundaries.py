"""Numerical and ownership regressions from the independent Phase 2C review."""

import copy
import types
import warnings
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from plato.callbacks.trainer import TrainerCallback
from plato.config import Config
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms.fedala_strategy import FedALAUpdateStrategy
from plato.trainers.strategies.algorithms.feddyn_strategy import FedDynUpdateStrategy
from plato.trainers.strategies.base import TrainingContext
from plato.trainers.strategies.loss_criterion import DefaultLossCriterionStrategy
from plato.trainers.strategies.optimizer import (
    DefaultOptimizerStrategy,
    GradientClippingOptimizerStrategy,
)
from plato.trainers.strategies.training_step import MixedPrecisionStepStrategy
from plato.utils.checkpoint_paths import checkpoint_name
from tests.integration.utils import build_minimal_config, configure_environment


class ObserveUpdates(TrainerCallback):
    def __init__(self):
        self.completed = 0
        self.flags = []

    def on_train_step_end(self, trainer, config, batch, loss, **kwargs):
        self.completed += 1
        self.flags.append(trainer.context.state["optimizer_step_completed"])


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("scale,gradient", [(8.0, 2.0), (65536.0, 10.0)])
@pytest.mark.parametrize("overflow", [False, True])
def test_real_amp_clipping_unscaled_success_and_overflow(
    tmp_path, fused, scale, gradient, overflow
):
    config = build_minimal_config()
    config["parameters"]["optimizer"].update(lr=0.1, fused=fused)
    with configure_environment(config, runtime_root=tmp_path):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(1.0)
        observer = ObserveUpdates()
        step = MixedPrecisionStepStrategy(enabled=False)

        def criterion(outputs, labels):
            if overflow:
                return outputs.mean() * float("inf")
            if gradient == 2.0:
                return (outputs - labels).square().mean()
            return gradient * outputs.mean()

        trainer = ComposableTrainer(
            model=model,
            callbacks=[observer],
            loss_strategy=DefaultLossCriterionStrategy(criterion),
            optimizer_strategy=GradientClippingOptimizerStrategy(
                DefaultOptimizerStrategy(), max_norm=1.0
            ),
            training_step_strategy=step,
        )
        trainer.device = "cpu"
        trainer.context.device = torch.device("cpu")
        assert trainer.model is not None
        step.enabled = True
        step.scaler = torch.amp.GradScaler("cpu", init_scale=scale)
        data = TensorDataset(torch.ones(1, 1), torch.zeros(1, 1))
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter("always")
            trainer.train_model(
                {**config["trainer"], "run_id": "amp-clip"}, data, [0]
            )
        expected = 1.0 if overflow else 1.0 - 0.1 * gradient / (gradient + 1e-6)
        assert model.weight.item() == pytest.approx(expected, abs=1e-7)
        assert observer.completed == (0 if overflow else 1)
        assert observer.flags == ([] if overflow else [True])
        assert not trainer.context.state.get("optimizer_step_completed", False)
        assert step.scaler.get_scale() == (scale / 2 if overflow else scale)
        assert notices == []


@pytest.mark.parametrize("name", ["toy", "org/model", "x" * 240])
def test_actual_worker_cleanup_full_names_and_unrelated_owners(tmp_path, name):
    config = build_minimal_config(model_name=name)
    config["trainer"]["max_concurrency"] = 1
    with configure_environment(config, runtime_root=tmp_path):
        trainer = ComposableTrainer(model=torch.nn.Linear(1, 2))
        trainer.set_client_id(7)
        trainer.device = "cpu"
        trainer.context.device = torch.device("cpu")
        assert trainer.model is not None
        data = TensorDataset(torch.ones(2, 1), torch.zeros(2, dtype=torch.long))
        run = {**config["trainer"], "run_id": Config.params["run_id"]}
        trainer.train_process(run, data, [0, 1])
        trainer.test_process(run, data)
        root = Path(Config.params["model_path"])
        primary = checkpoint_name(name, 7, run["run_id"], suffix=".safetensors")
        worker_files = {primary, primary + ".pkl"}
        worker_files.update(
            checkpoint_name(name, 7, run["run_id"], suffix=suffix)
            for suffix in (".acc", ".eval.pkl", ".train.pkl")
        )
        # Also cover an actual successful-state sidecar without changing the
        # ordinary model-only strategy's worker contract.
        state_name = checkpoint_name(name, 7, run["run_id"], suffix=".train.pkl")
        (root / state_name).write_bytes(b"this worker")
        retained = {
            checkpoint_name(name, 8, run["run_id"], suffix=".safetensors"),
            checkpoint_name(name, 7, "different-run", suffix=".eval.pkl"),
            checkpoint_name(name, suffix=".safetensors"),
            checkpoint_name(name, 7, "personalized", suffix=".pth"),
        }
        for filename in retained:
            (root / filename).write_bytes(b"retained")
        assert worker_files <= {p.name for p in root.iterdir()}
        trainer.pause_training()
        assert {p.name for p in root.iterdir()} == retained
        assert all(
            (root / filename).read_bytes() == b"retained" for filename in retained
        )


@pytest.mark.parametrize("custom_root", [False, True])
@pytest.mark.parametrize("trailing_slash", [False, True])
def test_actual_accepted_b_feddyn_writer_migration(
    tmp_path, custom_root, trailing_slash
):
    source = (
        Path(__file__).resolve().parents[1]
        / "fixtures/feddyn_accepted_b.py"
    ).read_text()
    old = types.ModuleType("accepted_b_feddyn_writer")
    exec(compile(source, "accepted_b_feddyn_writer", "exec"), old.__dict__)
    with configure_environment(build_minimal_config(), runtime_root=tmp_path):
        root = str(tmp_path / "custom") if custom_root else Config.params["model_path"]
        root = root.rstrip("/") + ("/" if trailing_slash else "")
        if not custom_root:
            Config.params["model_path"] = root
        save_path = root if custom_root else None
        Path(root).mkdir(parents=True, exist_ok=True)
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.zero_()
        context = TrainingContext()
        context.model, context.client_id = model, 1
        writer = old.FedDynUpdateStrategy(save_path=save_path)
        writer.setup(context)
        writer.on_train_start(context)
        model.weight.data.fill_(7.0)
        writer.on_train_end(context)
        legacy = Path(writer.grad_vector_path)
        original = legacy.read_bytes()
        expected = torch.load(legacy, weights_only=True)["weight"]
        assert expected.item() == 7.0
        reader = FedDynUpdateStrategy(save_path=save_path)
        reader.setup(context)
        torch.testing.assert_close(reader.read_legacy_history(context)["weight"], expected)
        context.client_id = 0
        writer.setup(context)
        torch.save({"weight": torch.full_like(expected, 99.0)}, writer.grad_vector_path)
        context.client_id = 2
        reader.on_client_id_changed(context)
        assert reader.read_legacy_history(context) is None
        canonical = Path(root) / "feddyn_grad_1.pth"
        torch.save({"weight": expected + 1}, canonical)
        context.client_id = 1
        reader.on_client_id_changed(context)
        assert reader.read_legacy_history(context)["weight"].item() == 8.0
        assert legacy.read_bytes() == original


@pytest.mark.parametrize(
    "spawn", [False, pytest.param(True, marks=pytest.mark.slow)]
)
def test_fedala_memory_only_return_matches_dedicated_rounds(tmp_path, spawn):
    config = build_minimal_config(model_name="org/model")
    config["parameters"]["optimizer"].update(lr=0.1, momentum=0.0)
    config["clients"]["random_seed"] = 17
    if spawn:
        config["trainer"]["max_concurrency"] = 1
    with configure_environment(config, runtime_root=tmp_path):
        def create():
            model = torch.nn.Linear(1, 2, bias=False)
            model.weight.data.copy_(torch.tensor([[1.0], [-1.0]]))
            strategy = FedALAUpdateStrategy(
                save_state=False, eta=1.0, max_ala_epochs=2, rand_percent=100
            )
            trainer = ComposableTrainer(model=model, model_update_strategy=strategy)
            trainer.device = "cpu"
            trainer.context.device = torch.device("cpu")
            assert trainer.model is not None
            trainer.set_client_id(1)
            return trainer

        reused, dedicated = create(), create()
        data = TensorDataset(torch.tensor([[1.0], [2.0]]), torch.tensor([1, 1]))
        for round_id, incoming in ((1, [[1.0], [-1.0]]), (2, [[2.0], [-1.0]])):
            for trainer in (reused, dedicated):
                trainer.current_round = round_id
                trainer.model.weight.data.copy_(torch.tensor(incoming))
                trainer.train(data, [0, 1])
            torch.testing.assert_close(reused.model.weight, dedicated.model.weight)
            expected = copy.deepcopy(dedicated.model_update_strategy.get_worker_state(
                dedicated.context
            ))
            reused.set_client_id(2)
            reused.model.weight.data.copy_(torch.tensor([[5.0], [-4.0]]))
            reused.train(data, [0, 1])
            second = copy.deepcopy(reused.model_update_strategy.local_model_state)
            reused.set_client_id(1)
            strategy = reused.model_update_strategy
            for name, value in expected["local_model"].items():
                torch.testing.assert_close(strategy.local_model_state[name], value)
            assert strategy.start_phase == expected["start_phase"]
            if expected["weights"] is None:
                assert strategy.weights is None
            else:
                for value, reference in zip(strategy.weights, expected["weights"]):
                    torch.testing.assert_close(value, reference)
            reused.set_client_id(2)
            for name, value in second.items():
                torch.testing.assert_close(strategy.local_model_state[name], value)
            reused.set_client_id(1)
        assert not list(Path(Config.params["model_path"]).glob("*fedala_*.pth"))
