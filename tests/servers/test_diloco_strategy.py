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


class ServerAlgorithm(DummyAlgorithm):
    """Algorithm stub for exercising FedAvg-compatible server dispatch."""

    def __init__(self, baseline):
        self.current = {
            name: value.clone() if hasattr(value, "clone") else value
            for name, value in baseline.items()
        }
        self.delta_payloads = None

    def extract_weights(self):
        return {
            name: value.clone() if hasattr(value, "clone") else value
            for name, value in self.current.items()
        }

    def compute_weight_deltas(self, baseline_weights, weights_list):
        self.delta_payloads = weights_list
        return super().compute_weight_deltas(baseline_weights, weights_list)

    def update_weights(self, deltas):
        self.current = {
            name: self.current[name] + deltas[name] for name in self.current
        }
        return self.extract_weights()

    def load_weights(self, weights):
        self.current = {
            name: value.clone() if hasattr(value, "clone") else value
            for name, value in weights.items()
        }


class RecordingDiLoCoStrategy(DiLoCoAggregationStrategy):
    """DiLoCo strategy recording server dispatch calls."""

    def __init__(self):
        super().__init__(
            outer_optimizer="sgd",
            outer_learning_rate=1.0,
            aggregation_weighting="uniform",
            apply_outer_optimizer_to="all_floating",
        )
        self.delta_calls = 0
        self.last_updates = None
        self.last_deltas = None

    async def aggregate_deltas(self, updates, deltas_received, context):
        self.delta_calls += 1
        self.last_updates = updates
        self.last_deltas = deltas_received
        return await super().aggregate_deltas(updates, deltas_received, context)


class MixedStateModel(torch.nn.Module):
    """Model exposing trainable, frozen, floating-buffer, and integer state."""

    def __init__(self):
        super().__init__()
        self.trainable = torch.nn.Parameter(torch.tensor([1.0]))
        self.frozen = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.register_buffer("floating_buffer", torch.tensor([1.0]))
        self.register_buffer("integer_buffer", torch.tensor([1], dtype=torch.int64))
        self.register_buffer("bool_buffer", torch.tensor([True], dtype=torch.bool))


class PeftLikeAdapterModel(torch.nn.Module):
    """Model whose adapter payload keys omit PEFT's default adapter segment."""

    def __init__(self):
        super().__init__()
        self.peft_config = {"default": object()}
        self.base_model = torch.nn.Module()
        self.base_model.model = torch.nn.Module()
        self.base_model.model.linear = torch.nn.Module()
        self.base_model.model.linear.lora_A = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(1, 1, bias=False)}
        )


class AdapterAliasCollisionModel(torch.nn.Module):
    """Model with a trainable parameter and separate payload key collision."""

    def __init__(self):
        super().__init__()
        self.peft_config = {"default": object()}
        self.foo = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(1, 1, bias=False)}
        )


def _context(baseline=None, model=None):
    context = ServerContext()
    if baseline is not None:
        context.algorithm = DummyAlgorithm(baseline)
    if model is not None:
        context.trainer = SimpleNamespace(model=model)
    return context


def _update(num_samples, report_type="weights"):
    return SimpleNamespace(
        report=SimpleNamespace(num_samples=num_samples, type=report_type)
    )


def _server_update(payload, num_samples=1, report_type="weights"):
    update = _update(num_samples, report_type)
    update.client_id = len(str(payload))
    update.report.accuracy = 0.5
    update.report.processing_time = 0.1
    update.report.comm_time = 0.1
    update.report.training_time = 0.1
    update.payload = payload
    return update


def _aggregate(strategy, updates, deltas, baseline=None, model=None):
    return asyncio.run(
        strategy.aggregate_deltas(updates, deltas, _context(baseline, model))
    )


def test_diloco_server_type_uses_fedavg_algorithm_and_strategy(temp_config):
    """server.type=diloco should select a FedAvg-compatible DiLoCo server."""
    from plato.algorithms import registry as algorithms_registry
    from plato.config import Config
    from plato.servers import diloco as diloco_server
    from plato.servers import fedavg
    from plato.servers import registry as servers_registry

    Config().server.type = "diloco"
    Config().algorithm.type = "fedavg"
    Config().server.diloco = SimpleNamespace(
        outer_optimizer="sgd",
        outer_learning_rate=0.25,
        outer_momentum=0.1,
        aggregation_weighting="num_samples",
        apply_outer_optimizer_to="all_floating",
    )

    server = servers_registry.get()

    assert isinstance(server, diloco_server.Server)
    assert isinstance(server, fedavg.Server)
    assert isinstance(server.aggregation_strategy, DiLoCoAggregationStrategy)
    assert server.aggregation_strategy.outer_optimizer == "sgd"
    assert server.aggregation_strategy.outer_learning_rate == 0.25
    assert server.aggregation_strategy.outer_momentum == 0.1
    assert server.aggregation_strategy.aggregation_weighting == "num_samples"
    assert server.aggregation_strategy.apply_outer_optimizer_to == "all_floating"
    assert Config().algorithm.type == "fedavg"
    assert "diloco" not in algorithms_registry.registered_algorithms


def test_diloco_server_process_reports_uses_delta_aggregation(temp_config):
    """DiLoCo server processing should reach the delta aggregation path."""
    from plato.config import Config
    from plato.servers import diloco

    Config().server.do_test = False
    strategy = RecordingDiLoCoStrategy()
    server = diloco.Server(aggregation_strategy=strategy)
    baseline = {"w": torch.zeros(1)}
    server.algorithm = ServerAlgorithm(baseline)
    server.context.algorithm = server.algorithm
    server.context.server = server
    server.context.state["prng_state"] = None
    server.updates = [
        _server_update({"w": torch.tensor([2.0])}),
        _server_update({"w": torch.tensor([4.0])}),
    ]

    asyncio.run(server._process_reports())

    assert strategy.delta_calls == 1
    assert strategy.last_updates == server.updates
    assert len(strategy.last_deltas) == 2
    assert torch.allclose(server.algorithm.current["w"], torch.tensor([3.0]))


def test_diloco_server_does_not_use_inherited_weight_aggregation(temp_config):
    """DiLoCo must not bypass delta aggregation via inherited FedAvg weights."""
    from plato.config import Config
    from plato.servers import diloco

    Config().server.do_test = False
    strategy = RecordingDiLoCoStrategy()

    async def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Inherited aggregate_weights() must not be called.")

    strategy.aggregate_weights = fail_if_called
    server = diloco.Server(aggregation_strategy=strategy)
    baseline = {"w": torch.zeros(1)}
    server.algorithm = ServerAlgorithm(baseline)
    server.context.algorithm = server.algorithm
    server.context.server = server
    server.context.state["prng_state"] = None
    server.updates = [_server_update({"w": torch.tensor([2.0])})]

    asyncio.run(server._process_reports())

    assert strategy.delta_calls == 1
    assert torch.allclose(server.algorithm.current["w"], torch.tensor([2.0]))


def test_diloco_server_filters_non_weight_reports_before_delta_computation(
    temp_config,
):
    """Non-weight payloads should not reach compute_weight_deltas()."""
    from plato.config import Config
    from plato.servers import diloco

    Config().server.do_test = False
    strategy = RecordingDiLoCoStrategy()
    server = diloco.Server(aggregation_strategy=strategy)
    baseline = {"w": torch.zeros(1)}
    server.algorithm = ServerAlgorithm(baseline)
    server.context.algorithm = server.algorithm
    server.context.server = server
    server.context.state["prng_state"] = None
    weight_payload = {"w": torch.tensor([2.0])}
    server.updates = [
        _server_update("feature payload", report_type="features"),
        _server_update({"metrics": 1.0}, report_type="metrics"),
        _server_update(weight_payload),
    ]

    asyncio.run(server._process_reports())

    assert server.algorithm.delta_payloads == [weight_payload]
    assert strategy.last_updates == [server.updates[2]]
    assert len(strategy.last_deltas) == 1
    assert torch.allclose(server.algorithm.current["w"], torch.tensor([2.0]))


def test_sgd_lr_one_uniform_matches_uniform_model_averaging(temp_config):
    """Outer SGD with lr=1 should match uniform averaging under uniform mode."""
    strategy = DiLoCoAggregationStrategy(
        outer_optimizer="sgd",
        outer_learning_rate=1.0,
        aggregation_weighting="uniform",
        apply_outer_optimizer_to="all_floating",
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
        apply_outer_optimizer_to="all_floating",
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
        apply_outer_optimizer_to="all_floating",
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
        apply_outer_optimizer_to="all_floating",
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
        apply_outer_optimizer_to="all_floating",
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
        apply_outer_optimizer_to="all_floating",
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
        apply_outer_optimizer_to="all_floating",
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
        apply_outer_optimizer_to="all_floating",
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
        apply_outer_optimizer_to="all_floating",
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
        apply_outer_optimizer_to="all_floating",
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
        apply_outer_optimizer_to="all_floating",
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
        apply_outer_optimizer_to="all_floating",
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


def test_parameters_policy_optimizes_only_trainable_floating_parameters(
    temp_config,
):
    """Default policy should leave frozen parameters and buffers on FedAvg deltas."""
    strategy = DiLoCoAggregationStrategy(
        outer_optimizer="sgdm",
        outer_learning_rate=0.5,
        outer_momentum=0.5,
        aggregation_weighting="uniform",
    )
    model = MixedStateModel()
    baseline = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    first_delta = _aggregate(
        strategy,
        [_update(1), _update(1)],
        [
            {
                "trainable": torch.tensor([2.0]),
                "frozen": torch.tensor([2.0]),
                "floating_buffer": torch.tensor([2.0]),
                "integer_buffer": torch.tensor([1], dtype=torch.int64),
                "bool_buffer": torch.tensor([False]),
            },
            {
                "trainable": torch.tensor([6.0]),
                "frozen": torch.tensor([6.0]),
                "floating_buffer": torch.tensor([6.0]),
                "integer_buffer": torch.tensor([2], dtype=torch.int64),
                "bool_buffer": torch.tensor([True]),
            },
        ],
        baseline,
        model,
    )

    assert torch.allclose(first_delta["trainable"], torch.tensor([2.0]))
    assert torch.allclose(first_delta["frozen"], torch.tensor([4.0]))
    assert torch.allclose(first_delta["floating_buffer"], torch.tensor([4.0]))
    assert torch.equal(first_delta["integer_buffer"], torch.tensor([2]))
    assert torch.equal(first_delta["bool_buffer"], torch.tensor([True]))
    assert set(strategy.momentum_state) == {"trainable"}
    assert torch.allclose(strategy.momentum_state["trainable"], torch.tensor([-4.0]))

    second_delta = _aggregate(
        strategy,
        [_update(1)],
        [
            {
                "trainable": torch.tensor([6.0]),
                "frozen": torch.tensor([6.0]),
                "floating_buffer": torch.tensor([6.0]),
                "integer_buffer": torch.tensor([1], dtype=torch.int64),
                "bool_buffer": torch.tensor([False]),
            }
        ],
        baseline,
        model,
    )

    assert torch.allclose(second_delta["trainable"], torch.tensor([4.0]))
    assert torch.allclose(second_delta["frozen"], torch.tensor([6.0]))
    assert torch.allclose(second_delta["floating_buffer"], torch.tensor([6.0]))
    assert torch.equal(second_delta["integer_buffer"], torch.tensor([1]))
    assert torch.equal(second_delta["bool_buffer"], torch.tensor([False]))
    assert set(strategy.momentum_state) == {"trainable"}
    assert torch.allclose(strategy.momentum_state["trainable"], torch.tensor([-8.0]))


def test_all_floating_policy_optimizes_every_floating_state_tensor(temp_config):
    """All-floating mode should not require model context for eligibility."""
    strategy = DiLoCoAggregationStrategy(
        outer_optimizer="sgdm",
        outer_learning_rate=0.5,
        outer_momentum=0.5,
        aggregation_weighting="uniform",
        apply_outer_optimizer_to="all_floating",
    )
    model = MixedStateModel()
    baseline = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    server_delta = _aggregate(
        strategy,
        [_update(1), _update(1)],
        [
            {
                "trainable": torch.tensor([2.0]),
                "frozen": torch.tensor([2.0]),
                "floating_buffer": torch.tensor([2.0]),
                "integer_buffer": torch.tensor([1], dtype=torch.int64),
                "bool_buffer": torch.tensor([False]),
            },
            {
                "trainable": torch.tensor([6.0]),
                "frozen": torch.tensor([6.0]),
                "floating_buffer": torch.tensor([6.0]),
                "integer_buffer": torch.tensor([2], dtype=torch.int64),
                "bool_buffer": torch.tensor([True]),
            },
        ],
        baseline,
    )

    assert torch.allclose(server_delta["trainable"], torch.tensor([2.0]))
    assert torch.allclose(server_delta["frozen"], torch.tensor([2.0]))
    assert torch.allclose(server_delta["floating_buffer"], torch.tensor([2.0]))
    assert torch.equal(server_delta["integer_buffer"], torch.tensor([2]))
    assert torch.equal(server_delta["bool_buffer"], torch.tensor([True]))
    assert set(strategy.momentum_state) == {
        "trainable",
        "frozen",
        "floating_buffer",
    }

    second_delta = _aggregate(
        strategy,
        [_update(1)],
        [
            {
                "trainable": torch.tensor([6.0]),
                "frozen": torch.tensor([6.0]),
                "floating_buffer": torch.tensor([6.0]),
                "integer_buffer": torch.tensor([1], dtype=torch.int64),
                "bool_buffer": torch.tensor([False]),
            }
        ],
        baseline,
    )

    assert torch.allclose(second_delta["trainable"], torch.tensor([4.0]))
    assert torch.allclose(second_delta["frozen"], torch.tensor([4.0]))
    assert torch.allclose(second_delta["floating_buffer"], torch.tensor([4.0]))
    assert torch.equal(second_delta["integer_buffer"], torch.tensor([1]))
    assert torch.equal(second_delta["bool_buffer"], torch.tensor([False]))
    assert set(strategy.momentum_state) == {
        "trainable",
        "frozen",
        "floating_buffer",
    }


def test_parameters_policy_requires_trainer_model_context(temp_config):
    """Default parameter eligibility should fail clearly without a model."""
    strategy = DiLoCoAggregationStrategy(
        outer_optimizer="sgd",
        outer_learning_rate=1.0,
        aggregation_weighting="uniform",
    )

    with pytest.raises(AttributeError, match="context.trainer.model"):
        _aggregate(
            strategy,
            [_update(1)],
            [{"trainable": torch.tensor([2.0])}],
            {"trainable": torch.tensor([0.0])},
        )


def test_parameters_policy_maps_peft_adapter_payload_names(temp_config):
    """PEFT payloads can omit adapter-name segments from trainable param names."""
    strategy = DiLoCoAggregationStrategy(
        outer_optimizer="sgdm",
        outer_learning_rate=0.5,
        outer_momentum=0.5,
        aggregation_weighting="uniform",
    )
    model = PeftLikeAdapterModel()
    payload_name = "base_model.model.linear.lora_A.weight"
    baseline = {payload_name: torch.zeros((1, 1))}

    server_delta = _aggregate(
        strategy,
        [_update(1)],
        [{payload_name: torch.full((1, 1), 4.0)}],
        baseline,
        model,
    )

    assert torch.allclose(server_delta[payload_name], torch.full((1, 1), 2.0))
    assert set(strategy.momentum_state) == {payload_name}
    assert torch.allclose(
        strategy.momentum_state[payload_name], torch.full((1, 1), -4.0)
    )


def test_parameters_policy_does_not_overmatch_adapter_alias_collisions(temp_config):
    """Alias support should not optimize unrelated colliding payload names."""
    strategy = DiLoCoAggregationStrategy(
        outer_optimizer="sgdm",
        outer_learning_rate=0.5,
        outer_momentum=0.5,
        aggregation_weighting="uniform",
    )
    model = AdapterAliasCollisionModel()
    trainable_name = "foo.default.weight"
    colliding_name = "foo.weight"
    baseline = {
        trainable_name: torch.zeros((1, 1)),
        colliding_name: torch.zeros((1, 1)),
    }

    server_delta = _aggregate(
        strategy,
        [_update(1)],
        [
            {
                trainable_name: torch.full((1, 1), 4.0),
                colliding_name: torch.full((1, 1), 4.0),
            }
        ],
        baseline,
        model,
    )

    assert torch.allclose(server_delta[trainable_name], torch.full((1, 1), 2.0))
    assert torch.allclose(server_delta[colliding_name], torch.full((1, 1), 4.0))
    assert set(strategy.momentum_state) == {trainable_name}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"outer_optimizer": "adam"}, "outer_optimizer"),
        ({"aggregation_weighting": "weighted"}, "aggregation_weighting"),
        ({"apply_outer_optimizer_to": "buffers"}, "apply_outer_optimizer_to"),
        ({"outer_learning_rate": -0.1}, "outer_learning_rate"),
        ({"outer_momentum": -0.1}, "outer_momentum"),
        ({"outer_momentum": 1.0}, "outer_momentum"),
    ],
)
def test_invalid_config_values_fail_clearly(temp_config, kwargs, match):
    """Invalid DiLoCo aggregation configuration should raise clear errors."""
    with pytest.raises(ValueError, match=match):
        DiLoCoAggregationStrategy(**kwargs)
