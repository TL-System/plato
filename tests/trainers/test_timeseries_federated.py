"""Regression coverage for time-series training across shared runtime updates."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from plato.algorithms.fedavg import Algorithm
from plato.config import Config
from plato.models.huggingface import TimesFmMultivariateWrapper
from plato.servers.strategies.aggregation.fedavg import FedAvgAggregationStrategy
from plato.servers.strategies.base import ServerContext
from plato.trainers.huggingface import Trainer
from tests.integration.utils import configure_environment


class TinyForecast(torch.nn.Module):
    """Differentiable stand-in for the univariate TimesFM forward API."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, past_values, **kwargs):
        values = torch.stack(past_values)
        return SimpleNamespace(mean_predictions=self.weight * values[:, -2:])


@pytest.mark.parametrize("transformers_api", [False, True])
def test_multivariate_forecast_preserves_channels_and_mse(transformers_api):
    model = TimesFmMultivariateWrapper(
        TinyForecast(), prediction_length=2, use_transformers_api=transformers_api
    )
    past = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    future = torch.ones(2, 2, 1)
    output = model(past_values=past, future_values=future)
    expected = past[:, -2:, :] * 0.25
    torch.testing.assert_close(output.prediction_outputs, expected)
    torch.testing.assert_close(output.loss, ((expected[:, :, :1] - future) ** 2).mean())
    output.loss.backward()
    assert model.model.weight.grad is not None


def test_timeseries_local_updates_aggregate_without_tokenizer(tmp_path, monkeypatch):
    from plato.trainers import huggingface

    def unexpected_tokenizer(*args, **kwargs):
        raise AssertionError("Time-series training must not load an NLP tokenizer.")

    monkeypatch.setattr(
        huggingface.AutoTokenizer, "from_pretrained", unexpected_tokenizer
    )
    config = {
        "clients": {"total_clients": 2, "per_round": 2},
        "server": {"address": "127.0.0.1", "port": 8000},
        "data": {"datasource": "EVCharging", "sampler": "iid"},
        "trainer": {
            "type": "HuggingFace",
            "model_type": "timesfm",
            "model_name": "google/timesfm-test",
            "rounds": 1,
            "epochs": 3,
            "batch_size": 1,
            "optimizer": "SGD",
            "local_steps_per_round": 2,
            "gradient_accumulation_steps": 2,
            "preserve_optimizer_state": True,
        },
        "algorithm": {"type": "fedavg"},
        "parameters": {"optimizer": {"lr": 0.01, "momentum": 0.9}},
    }
    with configure_environment(config, runtime_root=tmp_path):
        Config.args.cpu = True
        client_weights = []
        for client_id in (1, 2):
            model = TimesFmMultivariateWrapper(TinyForecast(), prediction_length=2)
            trainer = Trainer(model=model)
            trainer.device = torch.device("cpu")
            trainer.context.device = trainer.device
            trainer.set_client_id(client_id)
            data = [
                {
                    "past_values": torch.ones(4, 2),
                    "future_values": torch.full((2, 1), float(client_id)),
                }
                for _ in range(4)
            ]
            before = trainer.test_model({"batch_size": 2}, data)
            trainer.train(data, list(range(4)))
            assert trainer.context.state["local_optimizer_steps"] == 2
            assert trainer.tokenizer is None
            assert trainer.testing_strategy.metric_name == "mse"
            assert trainer.test_model({"batch_size": 2}, data) < before
            assert client_id in trainer._preserved_optimizer_states
            client_weights.append(Algorithm(trainer).extract_weights())

        updates = [
            SimpleNamespace(report=SimpleNamespace(num_samples=n)) for n in (1, 3)
        ]
        aggregated = asyncio.run(
            FedAvgAggregationStrategy().aggregate_weights(
                updates, client_weights[0], client_weights, ServerContext()
            )
        )
        for name, value in aggregated.items():
            torch.testing.assert_close(
                value, (client_weights[0][name] + 3 * client_weights[1][name]) / 4
            )
