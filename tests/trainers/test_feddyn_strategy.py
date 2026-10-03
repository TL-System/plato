"""Independent corrected FedDyn objective and bounded local-state contracts."""

import pytest
import torch

from plato.trainers.strategies.algorithms.feddyn_strategy import FedDynLossStrategy
from plato.trainers.strategies.base import TrainingContext
from tests.integration.utils import build_minimal_config, configure_environment


@pytest.mark.parametrize("mode,argument", [("uniform", True), ("sample", False)])
def test_explicit_adaptive_flag_conflicts_with_configuration(tmp_path, mode, argument):
    from plato.trainers.strategies.algorithms.feddyn_strategy import (
        FedDynLossStrategyFromConfig,
    )

    config = build_minimal_config()
    config["algorithm"]["feddyn_weighting"] = mode
    with configure_environment(config, runtime_root=tmp_path):
        with pytest.raises(ValueError, match="conflicts"):
            FedDynLossStrategyFromConfig(adaptive_alpha=argument)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("algorithm", "alpha_coef", 0),
        ("algorithm", "alpha_coef", True),
        ("algorithm", "alpha_coef", float("nan")),
        ("algorithm", "feddyn_weighting", "unknown"),
        ("algorithm", "feddyn_sample_counts", [2, 2]),
        ("clients", "total_clients", 0),
        ("clients", "total_clients", True),
        ("clients", "total_clients", 1.5),
        ("clients", "per_round", 0),
        ("clients", "per_round", 3),
        ("clients", "per_round", True),
        ("server", "synchronous", False),
        ("server", "request_update", True),
        ("algorithm", "total_silos", 2),
        ("trainer", "optimizer", "Adam"),
        ("trainer", "amp", True),
        ("trainer", "type", "diff_privacy"),
        ("trainer", "max_grad_norm", 1),
        ("trainer", "lr_scheduler", "StepLR"),
        ("trainer", "epochs", 0),
        ("trainer", "gradient_accumulation_steps", 0),
        ("trainer", "gradient_accumulation_steps", True),
    ],
)
def test_full_example_rejects_invalid_configuration(tmp_path, section, key, value):
    from plato.trainers.strategies.algorithms.feddyn_strategy import (
        settings_from_config,
    )

    config = build_minimal_config()
    config[section][key] = value
    with configure_environment(config, runtime_root=tmp_path):
        with pytest.raises(ValueError, match="FedDyn"):
            settings_from_config()
        assert not list((tmp_path / "models").glob("feddyn*"))


@pytest.mark.parametrize(
    "counts",
    [
        None,
        [],
        [1],
        [0, 1],
        [-1, 3],
        [True, 2],
        [1.5, 2],
        [1, float("inf")],
        [10**400, 1],
    ],
)
def test_full_sample_vector_is_required_and_finite(tmp_path, counts):
    from plato.trainers.strategies.algorithms.feddyn_strategy import (
        settings_from_config,
    )

    config = build_minimal_config()
    config["algorithm"].update(feddyn_weighting="sample")
    # TOML does not represent None or arbitrarily large ints; install those
    # programmatic configuration cases after the actual TOML loader.
    with configure_environment(config, runtime_root=tmp_path):
        from types import SimpleNamespace

        from plato.config import Config

        Config.algorithm = SimpleNamespace(
            **config["algorithm"], feddyn_sample_counts=counts
        )
        with pytest.raises(ValueError, match="count"):
            settings_from_config()


@pytest.mark.parametrize(
    "key,value",
    [
        ("momentum", 0.9),
        ("weight_decay", 0.01),
        ("dampening", 0.1),
        ("nesterov", True),
        ("maximize", True),
        ("lr", 0),
        ("lr", float("inf")),
        ("lr", True),
    ],
)
def test_optimizer_configuration_is_not_silently_coerced(tmp_path, key, value):
    from plato.trainers.strategies.algorithms.feddyn_strategy import (
        settings_from_config,
    )

    config = build_minimal_config()
    config["parameters"]["optimizer"][key] = value
    with configure_environment(config, runtime_root=tmp_path):
        with pytest.raises(ValueError, match="FedDyn"):
            settings_from_config()


def test_classification_permutation_preserves_full_objective_gradient(tmp_path):
    with configure_environment(build_minimal_config(), runtime_root=tmp_path):
        gradients = []
        for permutation in (False, True):
            model = torch.nn.Linear(2, 2, bias=False).double()
            weight = torch.tensor([[0.1, 0.2], [0.3, -0.1]], dtype=torch.double)
            h = torch.tensor([[0.5, -0.3], [0.4, 0.1]], dtype=torch.double)
            if permutation:
                weight, h = weight.flip(0), h.flip(0)
            model.weight.data.copy_(weight)
            context = TrainingContext()
            context.model = model
            context.state["feddyn_global_weights"] = {"weight": weight.clone()}
            context.state["feddyn_cumulative_grad"] = {"weight": h}
            loss = FedDynLossStrategy(alpha=0.1)
            loss.setup(context)
            labels = torch.tensor([0, 1, 1])
            if permutation:
                labels = 1 - labels
            inputs = torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.double
            )
            loss.compute_loss(model(inputs), labels, context).backward()
            gradients.append(
                model.weight.grad.flip(0) if permutation else model.weight.grad
            )
        torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=1e-12)


def test_legacy_actual_writer_weight_bias_and_corruption_are_inspection_only(tmp_path):
    import types
    from pathlib import Path

    from plato.trainers.strategies.algorithms.feddyn_strategy import (
        FedDynUpdateStrategy,
    )

    fixture = Path(__file__).resolve().parents[1] / "fixtures/feddyn_accepted_b.py"
    old = types.ModuleType("accepted_b_feddyn")
    exec(compile(fixture.read_bytes(), str(fixture), "exec"), old.__dict__)
    with configure_environment(build_minimal_config(), runtime_root=tmp_path):
        model = torch.nn.Linear(1, 1).double()
        model.weight.data.zero_()
        model.bias.data.zero_()
        context = TrainingContext()
        context.model, context.client_id = model, 1
        old_writer = old.FedDynUpdateStrategy()
        old_writer.setup(context)
        old_writer.on_train_start(context)
        model.weight.data.fill_(7)
        model.bias.data.fill_(8)
        old_writer.on_train_end(context)
        path = Path(old_writer.grad_vector_path)
        original = path.read_bytes()
        reader = FedDynUpdateStrategy()
        reader.setup(context)
        inspected = reader.read_legacy_history(context)
        assert inspected["weight"].item() == 7
        assert inspected["bias"].item() == 8
        assert reader.cumulative_grad_vector is None
        with pytest.raises(ValueError, match="versioned"):
            reader.on_train_start(context)
        assert path.read_bytes() == original
        path.write_bytes(b"corrupt")
        with pytest.raises(Exception):
            reader.read_legacy_history(context)


def scalar_context(w, x, h):
    model = torch.nn.Linear(1, 1, bias=False).double()
    model.weight.data.fill_(w)
    context = TrainingContext()
    context.model, context.client_id = model, 1
    context.state["feddyn_global_weights"] = {"weight": torch.tensor([[x]]).double()}
    context.state["feddyn_cumulative_grad"] = {"weight": torch.tensor([[h]]).double()}
    return model, context


@pytest.mark.parametrize(
    "w,x,h,gradient,y,h_next",
    [
        (2.0, 2.0, 0.0, 0.0, 2.0, 0.0),
        (1.0, 2.0, 0.5, -0.05, 1.005, -0.495),
        (3.0, 3.0, 0.5, 0.05, 2.995, 0.495),
    ],
)
def test_loss_matches_independent_scalar_equation(
    tmp_path, w, x, h, gradient, y, h_next
):
    with configure_environment(build_minimal_config(), runtime_root=tmp_path):
        model, context = scalar_context(w, x, h)
        loss = FedDynLossStrategy(
            alpha=0.1,
            adaptive_alpha=False,
            base_loss_fn=lambda output, labels: output.sum() * 0,
        )
        loss.setup(context)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss.compute_loss(
            model(torch.ones(2, 1).double()), torch.zeros(2).long(), context
        ).backward()
        assert model.weight.grad.item() == pytest.approx(gradient, abs=1e-12)
        optimizer.step()
        assert model.weight.item() == pytest.approx(y, abs=1e-12)
        assert h + model.weight.item() - x == pytest.approx(h_next, abs=1e-12)


@pytest.mark.parametrize(
    "alpha", [-1, float("nan"), float("inf"), -float("inf"), True, "0.1"]
)
def test_invalid_alpha_is_an_actionable_error(alpha):
    with pytest.raises(ValueError, match="alpha"):
        FedDynLossStrategy(alpha=alpha)


def test_zero_alpha_is_exact_task_loss(tmp_path):
    with configure_environment(build_minimal_config(), runtime_root=tmp_path):
        model, context = scalar_context(2, 3, 0.5)
        strategy = FedDynLossStrategy(
            alpha=0, base_loss_fn=lambda outputs, labels: 0.5 * outputs.square().mean()
        )
        strategy.setup(context)
        strategy.compute_loss(
            model(torch.ones(2, 1).double()), torch.zeros(2).long(), context
        ).backward()
        assert model.weight.grad.item() == 2


@pytest.mark.parametrize("labels", [[0, 0], [0, 1], [1, 1], [17, 93], [93, 17]])
def test_sample_alpha_uses_population_counts_never_labels(tmp_path, labels):
    with configure_environment(build_minimal_config(), runtime_root=tmp_path):
        model, context = scalar_context(2, 2, 0.5)
        context.state["feddyn_count_metadata"] = {
            "population": 2,
            "sample_counts": [1, 3],
            "client_id": 1,
        }
        strategy = FedDynLossStrategy(
            alpha=0.1,
            adaptive_alpha=True,
            base_loss_fn=lambda outputs, labels: outputs.sum() * 0,
        )
        strategy.setup(context)
        strategy.compute_loss(
            model(torch.ones(2, 1).double()), torch.tensor(labels), context
        ).backward()
        assert model.weight.grad.item() == pytest.approx(0.1, abs=1e-12)


def test_adaptive_alpha_requires_authoritative_metadata(tmp_path):
    with configure_environment(build_minimal_config(), runtime_root=tmp_path):
        model, context = scalar_context(2, 2, 0.5)
        strategy = FedDynLossStrategy(alpha=0.1, adaptive_alpha=True)
        strategy.setup(context)
        with pytest.raises(ValueError, match="count|metadata"):
            strategy._get_alpha_coefficient(torch.tensor([0, 1]), context)


@pytest.mark.parametrize("alpha,counts", [(1e308, [1, 100]), (5e-324, [10**300, 1])])
def test_effective_alpha_overflow_and_underflow_are_rejected(tmp_path, alpha, counts):
    from plato.trainers.strategies.algorithms.feddyn_strategy import (
        settings_from_config,
    )

    config = build_minimal_config()
    config["algorithm"].update(alpha_coef=alpha, feddyn_weighting="sample")
    with configure_environment(config, runtime_root=tmp_path):
        from types import SimpleNamespace

        from plato.config import Config

        Config.algorithm = SimpleNamespace(
            **config["algorithm"], feddyn_sample_counts=counts
        )
        with pytest.raises(ValueError, match="alpha"):
            settings_from_config()
