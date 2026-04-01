"""Tests for FedAvg aggregation and algorithm utilities."""

import asyncio
from types import SimpleNamespace

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
