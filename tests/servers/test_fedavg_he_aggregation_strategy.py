"""Tests for the homomorphic-encrypted FedAvg aggregation strategy."""

import asyncio
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch

from plato.servers.strategies.aggregation import FedAvgHEAggregationStrategy
from plato.servers.strategies.base import ServerContext
from plato.utils import homo_enc


class DummyTrainer:
    def __init__(self):
        self.device = torch.device("cpu")

    def zeros(self, shape):
        return torch.zeros(shape, device=self.device)


class DummyServer:
    def __init__(self):
        self.trainer = DummyTrainer()
        self.he_context = None
        self.weight_shapes = OrderedDict({"layer": (2,)})
        self.para_nums = {"layer": 2}
        self.total_samples = 0
        self.encrypted_model = None

    def _fedavg_hybrid(self, updates, weights_received):
        deserialized = [
            homo_enc.deserialize_weights(payload, self.he_context)
            for payload in weights_received
        ]

        unencrypted_weights = [
            homo_enc.extract_encrypted_model(item)[0] for item in deserialized
        ]
        encrypted_weights = [
            homo_enc.extract_encrypted_model(item)[1] for item in deserialized
        ]
        indices_list = [
            homo_enc.extract_encrypted_model(item)[2] for item in deserialized
        ]

        if indices_list:
            indices = indices_list[0]
        else:
            indices = []

        self.total_samples = sum(update.report.num_samples for update in updates)

        unencrypted_avg_update = self.trainer.zeros(unencrypted_weights[0].size)
        encrypted_avg_update = None

        for (unenc_w, enc_w), update in zip(
            zip(unencrypted_weights, encrypted_weights), updates
        ):
            weight = update.report.num_samples / self.total_samples
            unencrypted_avg_update += unenc_w * weight

            if enc_w is not None:
                if encrypted_avg_update is None:
                    encrypted_avg_update = enc_w * 0
                encrypted_avg_update += enc_w * weight

        if len(indices) == 0:
            encrypted_avg_update = None

        unencrypted_vector = unencrypted_avg_update
        if hasattr(unencrypted_vector, "detach"):
            unencrypted_vector = unencrypted_vector.detach().cpu().numpy()

        return homo_enc.wrap_encrypted_model(
            unencrypted_vector, encrypted_avg_update, indices
        )


def _build_context():
    server = DummyServer()
    context = ServerContext()
    context.server = server
    context.trainer = server.trainer
    context.algorithm = SimpleNamespace()
    return context


def test_he_aggregation_strategy_weighted_average_without_encryption():
    context = _build_context()
    strategy = FedAvgHEAggregationStrategy()
    strategy.setup(context)

    payload_a = homo_enc.wrap_encrypted_model(np.array([1.0, 2.0]), None, [])
    payload_b = homo_enc.wrap_encrypted_model(np.array([3.0, 4.0]), None, [])

    weights_received = [payload_a, payload_b]
    updates = [
        SimpleNamespace(report=SimpleNamespace(num_samples=2)),
        SimpleNamespace(report=SimpleNamespace(num_samples=1)),
    ]

    aggregated = asyncio.run(
        strategy.aggregate_weights(updates, {}, weights_received, context)
    )

    expected = torch.tensor([5.0 / 3.0, 8.0 / 3.0], dtype=aggregated["layer"].dtype)
    assert torch.allclose(aggregated["layer"], expected)
    assert context.server.encrypted_model is not None
