"""Tests for DiLoCo server-side outer aggregation."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from plato.servers.strategies.aggregation import DiLoCoAggregationStrategy
from plato.servers.strategies.base import ServerContext


class DummyAlgorithm:
    """Minimal algorithm stub for zero-delta construction."""

    def __init__(self, baseline):
        self.baseline = {
            name: value.clone() if hasattr(value, "clone") else value
            for name, value in baseline.items()
        }

    def extract_weights(self):
        return {
            name: value.clone() if hasattr(value, "clone") else value
            for name, value in self.baseline.items()
        }

    def compute_weight_deltas(self, baseline_weights, weights_list):
        return [
            {
                name: weights[name] - baseline_weights[name]
                for name in baseline_weights.keys()
            }
            for weights in weights_list
        ]


def _context(baseline=None):
    context = ServerContext()
    if baseline is not None:
        context.algorithm = DummyAlgorithm(baseline)
    return context


def _update(num_samples, report_type="weights"):
    return SimpleNamespace(
        report=SimpleNamespace(num_samples=num_samples, type=report_type)
    )


def _aggregate(strategy, updates, deltas, baseline=None):
    return asyncio.run(
        strategy.aggregate_deltas(updates, deltas, _context(baseline))
    )


def test_sgd_lr_one_uniform_matches_uniform_model_averaging(temp_config):
    """Outer SGD with lr=1 should match uniform averaging under uniform mode."""
    strategy = DiLoCoAggregationStrategy(
        outer_optimizer="sgd",
        outer_learning_rate=1.0,
        aggregation_weighting="uniform",
    )

    baseline = {"w": torch.tensor([10.0])}
    updates = [_update(1), _update(99)]
    deltas = [{"w": torch.tensor([2.0])}, {"w": torch.tensor([8.0])}]

    server_delta = _aggregate(strategy, updates, deltas, baseline)

    assert torch.allclose(baseline["w"] + server_delta["w"], torch.tensor([15.0]))


def test_sgd_lr_one_num_samples_matches_weighted_fedavg(temp_config):
    """Outer SGD with lr=1 should match sample-weighted FedAvg."""
    strategy = DiLoCoAggregationStrategy(
        outer_optimizer="sgd",
        outer_learning_rate=1.0,
        aggregation_weighting="num_samples",
    )

    baseline = {"w": torch.tensor([10.0])}
    updates = [_update(1), _update(3)]
    deltas = [{"w": torch.tensor([2.0])}, {"w": torch.tensor([8.0])}]

    server_delta = _aggregate(strategy, updates, deltas, baseline)

    assert torch.allclose(server_delta["w"], torch.tensor([6.5]))
    assert torch.allclose(baseline["w"] + server_delta["w"], torch.tensor([16.5]))


def test_sgd_lr_half_moves_halfway_to_averaged_model(temp_config):
    """A lower outer SGD lr should partially move toward the averaged model."""
    strategy = DiLoCoAggregationStrategy(
        outer_optimizer="sgd",
        outer_learning_rate=0.5,
        aggregation_weighting="uniform",
    )

    baseline = {"w": torch.tensor([10.0])}
    updates = [_update(5), _update(5)]
    deltas = [{"w": torch.tensor([2.0])}, {"w": torch.tensor([8.0])}]

    server_delta = _aggregate(strategy, updates, deltas, baseline)

    assert torch.allclose(server_delta["w"], torch.tensor([2.5]))
    assert torch.allclose(baseline["w"] + server_delta["w"], torch.tensor([12.5]))


def test_sgd_uses_diloco_outer_gradient_sign(temp_config):
    """The strategy should negate Plato deltas before applying outer SGD."""
    strategy = DiLoCoAggregationStrategy(
        outer_optimizer="sgd",
        outer_learning_rate=0.25,
        aggregation_weighting="uniform",
    )

    server_delta = _aggregate(
        strategy,
        [_update(1)],
        [{"w": torch.tensor([4.0])}],
        {"w": torch.tensor([0.0])},
    )

    assert torch.allclose(server_delta["w"], torch.tensor([1.0]))


def test_uniform_weighting_ignores_positive_sample_count_magnitude(temp_config):
    """Uniform mode should weight eligible clients equally."""
    strategy = DiLoCoAggregationStrategy(
        outer_optimizer="sgd",
        outer_learning_rate=1.0,
        aggregation_weighting="uniform",
    )

    server_delta = _aggregate(
        strategy,
        [_update(1), _update(1000)],
        [{"w": torch.tensor([0.0])}, {"w": torch.tensor([10.0])}],
        {"w": torch.tensor([0.0])},
    )

    assert torch.allclose(server_delta["w"], torch.tensor([5.0]))


def test_nonpositive_sample_reports_are_ineligible(temp_config):
    """Reports with zero or negative sample counts should not affect averages."""
    strategy = DiLoCoAggregationStrategy(
        outer_optimizer="sgd",
        outer_learning_rate=1.0,
        aggregation_weighting="num_samples",
    )

    server_delta = _aggregate(
        strategy,
        [_update(0), _update(-5), _update(10)],
        [
            {"w": torch.tensor([100.0])},
            {"w": torch.tensor([100.0])},
            {"w": torch.tensor([4.0])},
        ],
        {"w": torch.tensor([0.0])},
    )

    assert torch.allclose(server_delta["w"], torch.tensor([4.0]))


def test_empty_eligible_updates_return_zero_delta(temp_config):
    """An empty eligible set should produce a zero delta matching the baseline."""
    strategy = DiLoCoAggregationStrategy(
        outer_optimizer="sgd",
        outer_learning_rate=1.0,
        aggregation_weighting="uniform",
    )

    baseline = {"w": torch.tensor([3.0, 4.0])}
    server_delta = _aggregate(
        strategy,
        [_update(0), _update(5, report_type="features")],
        [{"w": torch.tensor([10.0, 10.0])}, {"w": torch.tensor([10.0, 10.0])}],
        baseline,
    )

    assert torch.allclose(server_delta["w"], torch.zeros_like(baseline["w"]))


def test_empty_eligible_updates_remove_stale_momentum(temp_config):
    """A round with no eligible keys should clear stale momentum buffers."""
    strategy = DiLoCoAggregationStrategy(
        outer_optimizer="sgdm",
        outer_learning_rate=1.0,
        outer_momentum=0.5,
        aggregation_weighting="uniform",
    )

    _aggregate(
        strategy,
        [_update(1)],
        [{"w": torch.tensor([2.0])}],
        {"w": torch.tensor([0.0])},
    )
    server_delta = _aggregate(
        strategy,
        [_update(0)],
        [{"w": torch.tensor([10.0])}],
        {"w": torch.tensor([0.0])},
    )

    assert torch.allclose(server_delta["w"], torch.tensor([0.0]))
    assert strategy.momentum_state == {}


def test_sgdm_persists_momentum_across_rounds(temp_config):
    """Momentum SGD should reuse server-side outer momentum across rounds."""
    strategy = DiLoCoAggregationStrategy(
        outer_optimizer="sgdm",
        outer_learning_rate=1.0,
        outer_momentum=0.5,
        aggregation_weighting="uniform",
    )

    first_delta = _aggregate(
        strategy,
        [_update(1)],
        [{"w": torch.tensor([2.0])}],
        {"w": torch.tensor([0.0])},
    )
    second_delta = _aggregate(
        strategy,
        [_update(1)],
        [{"w": torch.tensor([4.0])}],
        {"w": torch.tensor([0.0])},
    )

    assert torch.allclose(first_delta["w"], torch.tensor([2.0]))
    assert torch.allclose(second_delta["w"], torch.tensor([5.0]))
    assert torch.allclose(strategy.momentum_state["w"], torch.tensor([-5.0]))


def test_nesterov_uses_pytorch_style_two_round_recurrence(temp_config):
    """Nesterov should use g + beta * m after updating the momentum buffer."""
    strategy = DiLoCoAggregationStrategy(
        outer_optimizer="nesterov",
        outer_learning_rate=1.0,
        outer_momentum=0.5,
        aggregation_weighting="uniform",
    )

    first_delta = _aggregate(
        strategy,
        [_update(1)],
        [{"w": torch.tensor([2.0])}],
        {"w": torch.tensor([0.0])},
    )
    second_delta = _aggregate(
        strategy,
        [_update(1)],
        [{"w": torch.tensor([4.0])}],
        {"w": torch.tensor([0.0])},
    )

    assert torch.allclose(first_delta["w"], torch.tensor([3.0]))
    assert torch.allclose(second_delta["w"], torch.tensor([6.5]))
    assert torch.allclose(strategy.momentum_state["w"], torch.tensor([-5.0]))


def test_momentum_state_resets_on_shape_mismatch_and_removes_stale_keys(
    temp_config,
):
    """Momentum state should reset incompatible keys and prune missing keys."""
    strategy = DiLoCoAggregationStrategy(
        outer_optimizer="sgdm",
        outer_learning_rate=1.0,
        outer_momentum=0.5,
        aggregation_weighting="uniform",
    )

    _aggregate(
        strategy,
        [_update(1)],
        [{"w": torch.tensor([2.0]), "b": torch.tensor([1.0])}],
        {"w": torch.tensor([0.0]), "b": torch.tensor([0.0])},
    )

    server_delta = _aggregate(
        strategy,
        [_update(1)],
        [{"w": torch.tensor([4.0, 6.0])}],
        {"w": torch.tensor([0.0, 0.0])},
    )

    assert torch.allclose(server_delta["w"], torch.tensor([4.0, 6.0]))
    assert torch.allclose(strategy.momentum_state["w"], torch.tensor([-4.0, -6.0]))
    assert "b" not in strategy.momentum_state


def test_momentum_state_resets_on_dtype_mismatch(temp_config):
    """Momentum state should reset when the tensor dtype changes."""
    strategy = DiLoCoAggregationStrategy(
        outer_optimizer="sgdm",
        outer_learning_rate=1.0,
        outer_momentum=0.5,
        aggregation_weighting="uniform",
    )

    _aggregate(
        strategy,
        [_update(1)],
        [{"w": torch.tensor([2.0], dtype=torch.float32)}],
        {"w": torch.tensor([0.0], dtype=torch.float32)},
    )
    server_delta = _aggregate(
        strategy,
        [_update(1)],
        [{"w": torch.tensor([4.0], dtype=torch.float64)}],
        {"w": torch.tensor([0.0], dtype=torch.float64)},
    )

    assert torch.allclose(server_delta["w"], torch.tensor([4.0], dtype=torch.float64))
    assert strategy.momentum_state["w"].dtype == torch.float64


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"outer_optimizer": "adam"}, "outer_optimizer"),
        ({"aggregation_weighting": "weighted"}, "aggregation_weighting"),
        ({"outer_learning_rate": -0.1}, "outer_learning_rate"),
        ({"outer_momentum": -0.1}, "outer_momentum"),
        ({"outer_momentum": 1.0}, "outer_momentum"),
    ],
)
def test_invalid_config_values_fail_clearly(temp_config, kwargs, match):
    """Invalid DiLoCo aggregation configuration should raise clear errors."""
    with pytest.raises(ValueError, match=match):
        DiLoCoAggregationStrategy(**kwargs)
