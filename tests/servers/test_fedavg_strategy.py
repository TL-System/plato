"""Tests for FedAvg aggregation and algorithm utilities."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from plato.servers.strategies.aggregation import FedAvgAggregationStrategy
from plato.servers.strategies.base import ServerContext
from plato.trainers.composable import ComposableTrainer


def _mock_evaluation_state():
    from plato.evaluators.runner import (
        EVALUATION_PRIMARY_KEY,
        EVALUATION_RESULTS_KEY,
    )

    payload = {
        "evaluator": "mock",
        "primary_metric": "mock_score",
        "metrics": {"mock_score": 0.8, "aux_metric": 0.2},
        "higher_is_better": {"mock_score": True, "aux_metric": False},
        "metadata": {"source": "unit-test"},
        "artifacts": {"report": "mock.json"},
        "primary_value": 0.8,
    }

    return {
        EVALUATION_PRIMARY_KEY: {
            "evaluator": "mock",
            "metric": "mock_score",
            "value": 0.8,
        },
        EVALUATION_RESULTS_KEY: {"mock": payload},
    }


def _mock_lighteval_state():
    from plato.evaluators.runner import (
        EVALUATION_PRIMARY_KEY,
        EVALUATION_RESULTS_KEY,
    )

    payload = {
        "evaluator": "lighteval",
        "primary_metric": "ifeval_avg",
        "metrics": {
            "ifeval_avg": 0.21875,
            "hellaswag": 0.0,
            "arc_avg": 0.28125,
            "arc_easy": 0.375,
            "arc_challenge": 0.1875,
            "piqa": 0.0,
        },
        "higher_is_better": {},
        "metadata": {
            "raw_metrics": {
                "all": {
                    "acc": 0.28125,
                    "prompt_level_strict_acc": 0.21875,
                },
                "arc:challenge:0": {
                    "acc": 0.1875,
                    "acc_stderr": 0.0701,
                },
                "arc:easy:0": {
                    "acc": 0.375,
                    "acc_stderr": 0.0869,
                },
                "ifeval:0": {
                    "inst_level_loose_acc": 0.3191489361702128,
                    "prompt_level_loose_acc": 0.21875,
                    "prompt_level_strict_acc": 0.21875,
                },
                "hellaswag:0": {
                    "em": 0.0,
                    "em_stderr": 0.0,
                },
                "piqa_hf:0": {
                    "em": 0.0,
                    "em_stderr": 0.0,
                },
            }
        },
        "artifacts": {},
        "primary_value": 0.21875,
    }

    return {
        EVALUATION_PRIMARY_KEY: {
            "evaluator": "lighteval",
            "metric": "ifeval_avg",
            "value": 0.21875,
        },
        EVALUATION_RESULTS_KEY: {"lighteval": payload},
    }


def _runtime_update():
    return SimpleNamespace(
        report=SimpleNamespace(
            num_samples=4,
            accuracy=0.5,
            processing_time=0.1,
            comm_time=0.2,
            training_time=0.3,
        )
    )


def test_fedavg_aggregation_weighted_mean(temp_config):
    """FedAvg aggregation should compute the weighted mean of client deltas."""
    trainer = ComposableTrainer(model=lambda: torch.nn.Linear(2, 1))
    trainer.set_client_id(0)

    context = ServerContext()
    context.trainer = trainer

    deltas = [
        {"weight": torch.ones((1, 2)), "bias": torch.tensor([0.5])},
        {"weight": torch.full((1, 2), 3.0), "bias": torch.tensor([1.5])},
    ]
    updates = [
        SimpleNamespace(report=SimpleNamespace(num_samples=10)),
        SimpleNamespace(report=SimpleNamespace(num_samples=30)),
    ]

    aggregated = asyncio.run(
        FedAvgAggregationStrategy().aggregate_deltas(updates, deltas, context)
    )

    expected_weight = deltas[0]["weight"] * 0.25 + deltas[1]["weight"] * 0.75
    expected_bias = deltas[0]["bias"] * 0.25 + deltas[1]["bias"] * 0.75

    assert torch.allclose(aggregated["weight"], expected_weight)
    assert torch.allclose(aggregated["bias"], expected_bias)


def test_fedavg_aggregation_skips_feature_payloads(temp_config):
    """Feature updates should be ignored by the FedAvg aggregator."""
    trainer = ComposableTrainer(model=lambda: torch.nn.Linear(2, 1))
    trainer.set_client_id(0)

    context = ServerContext()
    context.trainer = trainer

    updates = [
        SimpleNamespace(report=SimpleNamespace(num_samples=10, type="features")),
    ]

    model = trainer.model
    assert model is not None
    aggregated = asyncio.run(
        FedAvgAggregationStrategy().aggregate_weights(
            updates, model.state_dict(), [{}], context
        )
    )

    assert aggregated is None


class DummyAlgorithm:
    """Minimal algorithm stub for server aggregation dispatch tests."""

    def __init__(self, baseline):
        self.current = {name: tensor.clone() for name, tensor in baseline.items()}

    def extract_weights(self):
        return {name: tensor.clone() for name, tensor in self.current.items()}

    def compute_weight_deltas(self, baseline_weights, weights_list):
        return [
            {
                name: weights[name] - baseline_weights[name]
                for name in baseline_weights.keys()
            }
            for weights in weights_list
        ]

    def update_weights(self, deltas):
        self.current = {
            name: self.current[name] + deltas[name] for name in self.current.keys()
        }
        return self.extract_weights()

    def load_weights(self, weights):
        self.current = {name: tensor.clone() for name, tensor in weights.items()}


class DeltaOnlyStrategy(FedAvgAggregationStrategy):
    """Strategy overriding only delta aggregation to exercise dispatch."""

    def __init__(self):
        super().__init__()
        self.delta_calls = 0

    async def aggregate_deltas(self, updates, deltas_received, context):
        self.delta_calls += 1
        return await super().aggregate_deltas(updates, deltas_received, context)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("bad_count", [-1, float("nan"), float("inf"), -float("inf")])
def test_aggregation_rejects_invalid_weights(temp_config, direct, reverse, bad_count):
    updates = [
        SimpleNamespace(report=SimpleNamespace(num_samples=bad_count)),
        SimpleNamespace(report=SimpleNamespace(num_samples=2)),
    ]
    payloads = [{"weight": torch.tensor([1.0])}, {"weight": torch.tensor([4.0])}]
    if reverse:
        updates.reverse()
        payloads.reverse()
    strategy = FedAvgAggregationStrategy()
    context = ServerContext()
    operation = (
        strategy.aggregate_weights(updates, payloads[0], payloads, context)
        if direct
        else strategy.aggregate_deltas(updates, payloads, context)
    )
    with pytest.raises(ValueError, match="sample"):
        asyncio.run(operation)
    assert all(torch.isfinite(p["weight"]).all() for p in payloads)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("payload_count", [0, 2])
def test_aggregation_rejects_cardinality_mismatch(temp_config, direct, payload_count):
    updates = [SimpleNamespace(report=SimpleNamespace(num_samples=1))]
    payloads = [{"weight": torch.tensor([1.0])} for _ in range(payload_count)]
    strategy = FedAvgAggregationStrategy()
    context = ServerContext()
    operation = (
        strategy.aggregate_weights(updates, {}, payloads, context)
        if direct
        else strategy.aggregate_deltas(updates, payloads, context)
    )
    with pytest.raises(ValueError, match="payload"):
        asyncio.run(operation)


def _dispatch_server(kind):
    from plato.servers import fedavg

    class LegacyServer(fedavg.Server):
        async def aggregate_weights(self, updates, baseline, weights):
            self.legacy_calls += 1
            result = await FedAvgAggregationStrategy().aggregate_weights(
                updates, baseline, weights, self.context
            )
            return baseline if result is None else result

    server = (
        LegacyServer(aggregation_strategy=DeltaOnlyStrategy())
        if kind == "legacy"
        else fedavg.Server(
            aggregation_strategy=DeltaOnlyStrategy() if kind == "delta" else None
        )
    )
    server.legacy_calls = 0
    server.algorithm = DummyAlgorithm({"weight": torch.tensor([10.0])})
    server.context.algorithm = server.algorithm
    server.context.server = server
    server.clients_processed = lambda: None
    server.updates = [
        SimpleNamespace(
            client_id=index,
            report=SimpleNamespace(
                num_samples=count,
                accuracy=accuracy,
                processing_time=0,
                training_time=0,
                comm_time=0,
            ),
            payload={"weight": torch.tensor([value])},
        )
        for index, count, accuracy, value in [(1, 0.25, 0.2, 1.0), (2, 0.75, 0.8, 4.0)]
    ]
    return server


@pytest.mark.parametrize("kind", ["ordinary", "legacy", "delta"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "defect", ["negative", "nonfinite", "hook_extra", "hook_missing"]
)
def test_server_validates_raw_and_hook_inputs_before_any_dispatch(
    temp_config, kind, reverse, defect
):
    from plato.config import Config

    Config().server.do_test = False
    server = _dispatch_server(kind)
    if defect == "negative":
        server.updates[0].report.num_samples = -1
    elif defect == "nonfinite":
        server.updates[0].report.num_samples = float("nan")
    else:
        server.weights_received = (
            (lambda weights: weights + [weights[0]])
            if defect == "hook_extra"
            else (lambda weights: weights[:1])
        )
    if reverse:
        server.updates.reverse()
    before = server.algorithm.extract_weights()
    selection_state = dict(server.context.state)
    with pytest.raises(ValueError, match="sample|payload"):
        asyncio.run(server._process_reports())
    assert torch.equal(server.algorithm.current["weight"], before["weight"])
    assert server.context.state == selection_state
    assert server.legacy_calls == 0
    if isinstance(server.aggregation_strategy, DeltaOnlyStrategy):
        assert server.aggregation_strategy.delta_calls == 0


@pytest.mark.parametrize("kind", ["ordinary", "legacy", "delta"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("zero", ["none", "one", "all"])
def test_server_preserves_fractional_and_zero_sample_dispatch(
    temp_config, kind, reverse, zero
):
    from plato.config import Config

    Config().server.do_test = False
    server = _dispatch_server(kind)
    if zero != "none":
        server.updates[0].report.num_samples = 0
    if zero == "all":
        server.updates[1].report.num_samples = 0
    if reverse:
        server.updates.reverse()
    asyncio.run(server._process_reports())
    expected = {"none": 3.25, "one": 4.0, "all": 10.0}[zero]
    assert server.algorithm.current["weight"].item() == pytest.approx(expected)
    if kind == "legacy":
        assert server.legacy_calls == 1
    if kind == "delta":
        assert server.aggregation_strategy.delta_calls == 1


@pytest.mark.parametrize("reverse", [False, True])
def test_feature_samples_do_not_dilute_fractional_weight_reference(
    temp_config, reverse
):
    updates = [
        SimpleNamespace(report=SimpleNamespace(num_samples=100, type="features")),
        SimpleNamespace(report=SimpleNamespace(num_samples=0.25)),
        SimpleNamespace(report=SimpleNamespace(num_samples=0.75)),
        SimpleNamespace(report=SimpleNamespace(num_samples=0)),
    ]
    payloads = [{}, {"w": torch.tensor([1.0])}, {"w": torch.tensor([4.0])}, {}]
    if reverse:
        updates.reverse()
        payloads.reverse()
    result = asyncio.run(
        FedAvgAggregationStrategy().aggregate_weights(
            updates, {"w": torch.tensor([0.0])}, payloads, ServerContext()
        )
    )
    assert result["w"].item() == pytest.approx(3.25)


def test_fedavg_server_prefers_custom_delta_strategy_over_inherited_weights(
    temp_config,
):
    """Custom delta strategies should not be bypassed by inherited weight hooks."""
    from plato.config import Config
    from plato.servers import fedavg

    Config().server.do_test = False

    strategy = DeltaOnlyStrategy()
    server = fedavg.Server(aggregation_strategy=strategy)

    baseline = {"weight": torch.zeros((1, 2)), "bias": torch.zeros(1)}
    server.algorithm = DummyAlgorithm(baseline)
    server.context.algorithm = server.algorithm
    server.context.server = server
    server.context.state["prng_state"] = None

    server.updates = [
        SimpleNamespace(
            client_id=1,
            report=SimpleNamespace(
                num_samples=1,
                accuracy=0.5,
                processing_time=0.1,
                comm_time=0.1,
                training_time=0.1,
            ),
            payload={
                "weight": torch.ones((1, 2)),
                "bias": torch.ones(1),
            },
        )
    ]

    asyncio.run(server._process_reports())

    assert strategy.delta_calls == 1
    assert torch.allclose(server.algorithm.current["weight"], torch.ones((1, 2)))
    assert torch.allclose(server.algorithm.current["bias"], torch.ones(1))


def test_fedavg_server_logged_items_flatten_evaluator_metrics(
    temp_config, tmp_path
):
    """FedAvg should keep accuracy while surfacing evaluator summary metrics."""
    from plato.config import Config
    from plato.servers import fedavg

    result_path = tmp_path / "results"
    result_path.mkdir()
    Config.params["result_path"] = str(result_path)

    server = fedavg.Server()
    server.current_round = 2
    server.accuracy = 0.5
    server.accuracy_std = 0.0
    server.initial_wall_time = 10.0
    server.wall_time = 15.0
    server.comm_overhead = 1.5
    server.updates = [_runtime_update()]
    server.trainer = SimpleNamespace(
        context=SimpleNamespace(state=_mock_evaluation_state())
    )

    logged_items = server.get_logged_items()

    assert logged_items["accuracy"] == 0.5
    assert logged_items["evaluation_primary_value"] == 0.8
    assert logged_items["evaluation_mock_score"] == 0.8
    assert logged_items["evaluation_aux_metric"] == 0.2


def test_fedavg_server_logged_items_include_detailed_lighteval_metrics(
    temp_config, tmp_path
):
    """FedAvg should expose detailed Lighteval task metrics for CSV logging."""
    from plato.config import Config
    from plato.servers import fedavg

    result_path = tmp_path / "results"
    result_path.mkdir()
    Config.params["result_path"] = str(result_path)

    server = fedavg.Server()
    server.current_round = 2
    server.accuracy = 0.5
    server.accuracy_std = 0.0
    server.initial_wall_time = 10.0
    server.wall_time = 15.0
    server.comm_overhead = 1.5
    server.updates = [_runtime_update()]
    server.trainer = SimpleNamespace(
        context=SimpleNamespace(state=_mock_lighteval_state())
    )

    logged_items = server.get_logged_items()

    assert logged_items["evaluation_ifeval_avg"] == 0.21875
    assert logged_items["evaluation_arc_easy"] == 0.375
    assert logged_items["evaluation_arc_challenge"] == 0.1875
    assert logged_items["evaluation_ifeval_prompt_level_strict_acc"] == 0.21875
    assert logged_items["evaluation_ifeval_inst_level_loose_acc"] == 0.3191489361702128
    assert logged_items["evaluation_hellaswag_em"] == 0.0
    assert logged_items["evaluation_piqa_em"] == 0.0
    assert logged_items["evaluation_arc_easy_acc"] == 0.375
    assert logged_items["evaluation_arc_challenge_acc_stderr"] == 0.0701


def test_fedavg_server_does_not_persist_evaluator_jsonl_sidecar(
    temp_config, tmp_path
):
    """FedAvg should rely on CSV logging instead of a JSONL sidecar."""
    from plato.config import Config
    from plato.servers import fedavg

    result_path = tmp_path / "results"
    result_path.mkdir()
    Config.params["result_path"] = str(result_path)

    server = fedavg.Server()
    server.current_round = 3
    server.accuracy = 0.5
    server.trainer = SimpleNamespace(
        context=SimpleNamespace(state=_mock_evaluation_state())
    )

    server.clients_processed()

    assert not any(result_path.glob("*_evaluation.jsonl"))
