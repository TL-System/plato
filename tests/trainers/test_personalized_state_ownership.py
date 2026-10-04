"""Logical-client isolation through actual trainer identity and training APIs."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms.apfl_strategy import (
    APFLStepStrategy,
    APFLUpdateStrategy,
)
from plato.trainers.strategies.algorithms.ditto_strategy import DittoUpdateStrategy
from plato.trainers.strategies.algorithms.fedala_strategy import FedALAUpdateStrategy
from plato.trainers.strategies.algorithms.feddyn_strategy import (
    FedDynLossStrategy,
    FedDynUpdateStrategy,
)
from plato.trainers.strategies.algorithms.scaffold_strategy import (
    SCAFFOLDUpdateStrategy,
)
from plato.trainers.strategies.model_update import CompositeUpdateStrategy
from tests.integration.utils import build_minimal_config, configure_environment


def model():
    result = torch.nn.Linear(2, 2)
    with torch.no_grad():
        result.weight.fill_(1.0)
        result.bias.zero_()
    return result


@pytest.mark.parametrize("family", ["ditto", "apfl", "feddyn", "scaffold", "fedala"])
@pytest.mark.parametrize("name", ["toy", "org/model"])
def test_identity_reassignment_isolates_and_reloads_actual_personal_state(
    tmp_path, family, name
):
    if family == "feddyn":
        # Corrected FedDyn's authority is a versioned server dispatch, rather
        # than the old mathematically inconsistent per-client disk trajectory.
        from tests.integration.test_feddyn_round_flow import run_partial

        run_partial(tmp_path, reuse=True, model_name=name)
        return
    config = build_minimal_config(model_name=name)
    config["trainer"].update(batch_size=2, epochs=1)
    with configure_environment(config, runtime_root=tmp_path):
        custom = tmp_path / "personal"
        if family == "ditto":
            strategy = DittoUpdateStrategy(
                model_fn=model, personalization_epochs=1, save_path=str(custom)
            )
            training_step_strategy = None
            loss_strategy = None
            path_attr = "personalized_model_path"
        elif family == "apfl":
            strategy = APFLUpdateStrategy(model_fn=model, save_path=str(custom))
            training_step_strategy = APFLStepStrategy()
            loss_strategy = None
            path_attr = "personalized_model_path"
        elif family == "feddyn":
            strategy = FedDynUpdateStrategy(save_path=str(custom))
            training_step_strategy = None
            loss_strategy = FedDynLossStrategy()
            path_attr = "grad_vector_path"
        elif family == "scaffold":
            strategy = SCAFFOLDUpdateStrategy(save_path=str(custom))
            training_step_strategy = None
            loss_strategy = None
            path_attr = "client_control_variate_path"
        else:
            strategy = FedALAUpdateStrategy(max_ala_epochs=2, rand_percent=100)
            training_step_strategy = None
            loss_strategy = None
            path_attr = "_local_model_path"
        trainer = ComposableTrainer(
            model=model,
            model_update_strategy=(
                CompositeUpdateStrategy([strategy]) if name == "org/model" else strategy
            ),
            training_step_strategy=training_step_strategy,
            loss_strategy=loss_strategy,
        )
        trainer.device = "cpu"
        trainer.context.device = torch.device("cpu")
        assert trainer.model is not None
        torch.manual_seed(17)
        data = TensorDataset(torch.eye(2), torch.tensor([0, 1]))
        run = {**config["trainer"], "run_id": "personal"}
        trainer.set_client_id(1)
        trainer.train_model(run, data, [0, 1])
        if isinstance(strategy, FedALAUpdateStrategy):
            # A new global model in a second real run learns ALA weights.
            trainer.current_round = 2
            with torch.no_grad():
                assert isinstance(trainer.model, torch.nn.Linear)
                trainer.model.weight.add_(0.2)
            trainer.train_model(run, data, [0, 1])
            assert strategy.weights is not None
            assert strategy.start_phase is False
        first_path = Path(getattr(strategy, path_attr))
        assert first_path.is_file()
        root = tmp_path / "models" if family == "fedala" else custom
        assert first_path.parent == root
        first_bytes = first_path.read_bytes()
        if isinstance(strategy, (DittoUpdateStrategy, APFLUpdateStrategy)):
            assert isinstance(strategy.personalized_model, torch.nn.Linear)
            first_state = {
                k: v.clone()
                for k, v in strategy.personalized_model.state_dict().items()
            }
            if isinstance(strategy, APFLUpdateStrategy):
                torch.save(0.23, strategy.alpha_path)
        elif isinstance(strategy, FedDynUpdateStrategy):
            assert strategy.cumulative_grad_vector is not None
            first_state = {
                k: v.clone() for k, v in strategy.cumulative_grad_vector.items()
            }
        elif isinstance(strategy, SCAFFOLDUpdateStrategy):
            assert strategy.client_control_variate is not None
            first_state = {
                k: v.clone() for k, v in strategy.client_control_variate.items()
            }
        else:
            assert isinstance(strategy, FedALAUpdateStrategy)
            assert strategy.local_model_state is not None
            first_state = {k: v.clone() for k, v in strategy.local_model_state.items()}
        trainer.set_client_id(2)
        second_path = Path(getattr(strategy, path_attr))
        assert second_path != first_path
        assert not second_path.exists()
        if isinstance(strategy, (DittoUpdateStrategy, APFLUpdateStrategy)):
            assert isinstance(strategy.personalized_model, torch.nn.Linear)
            torch.testing.assert_close(
                strategy.personalized_model.weight, model().weight
            )
            assert not torch.equal(
                strategy.personalized_model.weight, first_state["weight"]
            )
            if isinstance(strategy, APFLUpdateStrategy):
                assert strategy.alpha == 0.5
                assert "apfl_personalized_optimizer" not in trainer.context.state
        elif isinstance(strategy, FedDynUpdateStrategy):
            assert isinstance(trainer.loss_strategy, FedDynLossStrategy)
            assert strategy.cumulative_grad_vector is None
            assert trainer.loss_strategy.cumulative_grad_vector is None
        elif isinstance(strategy, SCAFFOLDUpdateStrategy):
            assert strategy.client_control_variate is None
        else:
            assert isinstance(strategy, FedALAUpdateStrategy)
            assert strategy.local_model_state is None
            assert strategy.weights is None
            assert strategy.start_phase is True
        trainer.train_model(run, data, [0, 1])
        assert second_path.is_file()
        assert first_path.read_bytes() == first_bytes
        trainer.set_client_id(1)
        strategy.on_train_start(trainer.context)
        if isinstance(strategy, (DittoUpdateStrategy, APFLUpdateStrategy)):
            assert isinstance(strategy.personalized_model, torch.nn.Linear)
            actual_state = strategy.personalized_model.state_dict()
            if isinstance(strategy, APFLUpdateStrategy):
                assert strategy.alpha == pytest.approx(0.23)
                # Repeating assignment for the same client preserves optimizer/state.
                trainer.context.state["apfl_personalized_optimizer"] = "same client"
                trainer.set_client_id(1)
                assert (
                    trainer.context.state["apfl_personalized_optimizer"]
                    == "same client"
                )
        elif isinstance(strategy, FedDynUpdateStrategy):
            assert isinstance(trainer.loss_strategy, FedDynLossStrategy)
            actual_state = strategy.cumulative_grad_vector
            trainer.loss_strategy.on_train_start(trainer.context)
            assert trainer.loss_strategy.cumulative_grad_vector is actual_state
        elif isinstance(strategy, SCAFFOLDUpdateStrategy):
            actual_state = strategy.client_control_variate
        else:
            assert isinstance(strategy, FedALAUpdateStrategy)
            actual_state = strategy.local_model_state
        assert actual_state is not None
        for key, value in actual_state.items():
            torch.testing.assert_close(value, first_state[key])


def test_feddyn_exact_legacy_prefix_load_is_same_client_and_canonical_wins(tmp_path):
    with configure_environment(build_minimal_config(), runtime_root=tmp_path):
        custom = tmp_path / "personal"
        custom.mkdir()
        legacy = Path(str(custom) + "_feddyn_grad_1.pth")
        saved = {
            name: torch.full_like(value, 3.0)
            for name, value in model().named_parameters()
        }
        torch.save(saved, legacy)
        torch.save(
            {
                name: torch.full_like(value, 99.0)
                for name, value in model().named_parameters()
            },
            Path(str(custom) + "_feddyn_grad_0.pth"),
        )
        original = legacy.read_bytes()
        strategy = FedDynUpdateStrategy(save_path=str(custom))
        trainer = ComposableTrainer(
            model=model,
            model_update_strategy=strategy,
            loss_strategy=FedDynLossStrategy(),
        )
        trainer.set_client_id(1)
        inspected = strategy.read_legacy_history(trainer.context)
        for name, value in saved.items():
            torch.testing.assert_close(inspected[name], value)
        with pytest.raises(ValueError, match="read-only"):
            strategy.on_train_start(trainer.context)
        trainer.set_client_id(2)
        assert strategy.read_legacy_history(trainer.context) is None
        torch.save(
            {name: value + 1 for name, value in saved.items()},
            custom / "feddyn_grad_1.pth",
        )
        trainer.set_client_id(1)
        assert strategy.read_legacy_history(trainer.context)["weight"][0, 0].item() == 4.0
        assert legacy.read_bytes() == original


@pytest.mark.parametrize("family", ["ditto", "apfl", "feddyn", "fedala"])
@pytest.mark.slow
def test_actual_spawn_returns_personal_state_and_optimizer_across_rounds(tmp_path, family):
    if family == "feddyn":
        import json
        import shlex
        import subprocess
        import sys

        # Other example-loader tests replace this module name. A guarded
        # interpreter gives real spawn stable import identities, as deployment
        # does, while retaining all three round/parent-state oracle assertions.
        output = tmp_path / "feddyn-result.json"
        command = [sys.executable, "-m", "tests.integration.feddyn_round_worker",
                   str(tmp_path / "runtime"), str(output), "uniform", "2,2"]
        completed = subprocess.run(["zsh", "-lc", shlex.join(command)],
                                   capture_output=True, text=True, timeout=100)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert [record["x"] for record in json.loads(output.read_text())] == pytest.approx(
            [1.433, 1.9717445, 1.47705273425], abs=1e-12
        )
        return
    config = build_minimal_config(model_name="org/model")
    config["trainer"].update(batch_size=2, epochs=1, max_concurrency=1)
    config["clients"]["random_seed"] = 17
    config["parameters"]["optimizer"]["momentum"] = .9
    with configure_environment(config, runtime_root=tmp_path):
        def create(label):
            training_step_strategy = None
            loss_strategy = None
            if family == "ditto":
                strategy = DittoUpdateStrategy(model_fn=model, personalization_epochs=1,
                                               save_path=str(tmp_path / label))
            elif family == "apfl":
                strategy = APFLUpdateStrategy(model_fn=model, save_path=str(tmp_path / label))
                training_step_strategy = APFLStepStrategy()
            elif family == "feddyn":
                strategy = FedDynUpdateStrategy(save_path=str(tmp_path / label))
                loss_strategy = FedDynLossStrategy()
            else:
                strategy = FedALAUpdateStrategy(save_state=False, max_ala_epochs=2, rand_percent=100)
            trainer = ComposableTrainer(
                model=model,
                model_update_strategy=strategy,
                training_step_strategy=training_step_strategy,
                loss_strategy=loss_strategy,
            )
            trainer.set_client_id(1)
            return trainer, strategy
        worker, actual_strategy = create("worker-personal")
        reference, expected_strategy = create("reference-personal")
        data = TensorDataset(torch.eye(2), torch.tensor([0, 1]))
        for round_id in (1, 2):
            for trainer in (worker, reference):
                trainer.current_round = round_id
                assert isinstance(trainer.model, torch.nn.Linear)
                trainer.model.weight.data.fill_(float(round_id))
                trainer.model.bias.data.zero_()
            worker.train(data, torch.utils.data.SequentialSampler(data))
            reference.train_model({**config["trainer"], "run_id": "direct"}, data,
                                  torch.utils.data.SequentialSampler(data))
            for actual, expected in zip(worker.model.parameters(), reference.model.parameters()):
                torch.testing.assert_close(actual, expected)
            if family in {"ditto", "apfl"}:
                actual_state = actual_strategy.personalized_model.state_dict()
                expected_state = expected_strategy.personalized_model.state_dict()
            elif family == "feddyn":
                actual_state, expected_state = actual_strategy.cumulative_grad_vector, expected_strategy.cumulative_grad_vector
            else:
                actual_state, expected_state = actual_strategy.local_model_state, expected_strategy.local_model_state
            assert actual_state is not None
            for name, expected in expected_state.items():
                torch.testing.assert_close(actual_state[name], expected)
            if family == "apfl":
                assert actual_strategy.alpha == pytest.approx(expected_strategy.alpha)
                actual_optimizer = worker.context.state["apfl_personalized_optimizer"]
                expected_optimizer = reference.context.state["apfl_personalized_optimizer"]
                for actual, expected in zip(actual_optimizer.state.values(), expected_optimizer.state.values()):
                    torch.testing.assert_close(actual["momentum_buffer"], expected["momentum_buffer"])
            if family == "fedala" and round_id == 2:
                assert actual_strategy.start_phase is expected_strategy.start_phase is False
                for actual, expected in zip(actual_strategy.weights, expected_strategy.weights):
                    torch.testing.assert_close(actual, expected)


def test_apfl_historical_numpy_alpha_reader_and_primitive_writer(tmp_path):
    import numpy as np

    with configure_environment(build_minimal_config(), runtime_root=tmp_path):
        strategy = APFLUpdateStrategy(model_fn=model, save_path=str(tmp_path / "personal"))
        trainer = ComposableTrainer(model=model, model_update_strategy=strategy,
                                    training_step_strategy=APFLStepStrategy())
        trainer.set_client_id(1)
        torch.save(np.float64(.23), strategy.alpha_path)
        strategy.on_train_start(trainer.context)
        assert strategy.alpha == pytest.approx(.23)
        strategy.on_train_end(trainer.context)
        loaded = torch.load(strategy.alpha_path, weights_only=True)
        assert isinstance(loaded, float) and loaded == pytest.approx(.23)
