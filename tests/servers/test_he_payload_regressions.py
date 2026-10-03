"""Native CKKS aggregation through the production server and strategy methods."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import tenseal as ts
import torch

from plato.servers.strategies.aggregation import FedAvgHEAggregationStrategy
from plato.utils import homo_enc


@pytest.fixture(autouse=True)
def runtime_config(temp_config):
    """Initialize Config before importing the server's legacy defaults."""


@pytest.fixture(scope="module")
def ckks():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
        n_threads=1,
    )
    context.global_scale = 2**40
    return context


def server_context(ckks):
    from plato.servers.fedavg_he import Server

    # Execute production HE methods without starting sockets or creating key files.
    server = Server.__new__(Server)
    server.trainer = SimpleNamespace(zeros=torch.zeros)
    server.he_context = ckks
    server.weight_shapes = {"layer.weight": (3,)}
    server.para_nums = {"layer.weight": 3}
    server.encrypted_model = "previous"
    return SimpleNamespace(server=server, trainer=server.trainer)


@pytest.mark.parametrize("indices", [[], [0, 2], [0, 1, 2]])
def test_real_ckks_weighted_average_matches_plaintext(ckks, indices):
    context = server_context(ckks)
    weights = [torch.tensor([1.0, -2.0, 0.5]), torch.tensor([-3.0, 4.0, 1.5])]
    payloads = [
        homo_enc.encrypt_weights(
            {"layer.weight": w}, context=ckks, indices=indices.copy()
        )
        for w in weights
    ]
    updates = [SimpleNamespace(report=SimpleNamespace(num_samples=n)) for n in [2, 6]]
    result = asyncio.run(
        FedAvgHEAggregationStrategy().aggregate_weights(updates, {}, payloads, context)
    )
    expected = (2 * weights[0].double() + 6 * weights[1].double()) / 8
    torch.testing.assert_close(
        result["layer.weight"].double(), expected, atol=2e-5, rtol=0
    )
    assert context.server.total_samples == 8
    if indices:
        assert isinstance(context.server.encrypted_model["encrypted_weights"], bytes)


@pytest.mark.parametrize(
    "damage", ["cardinality", "negative", "nan", "inf", "mask", "length"]
)
def test_invalid_he_payload_fails_before_model_mutation(ckks, damage):
    context = server_context(ckks)
    payloads = [
        homo_enc.encrypt_weights(
            {"layer.weight": torch.ones(3)}, context=ckks, indices=[]
        )
        for _ in range(2)
    ]
    updates = [SimpleNamespace(report=SimpleNamespace(num_samples=n)) for n in [2, 6]]
    if damage == "cardinality":
        updates.pop()
    elif damage in {"negative", "nan", "inf"}:
        updates[0].report.num_samples = {
            "negative": -1,
            "nan": float("nan"),
            "inf": float("inf"),
        }[damage]
    elif damage == "mask":
        payloads[0]["indices"] = [0, 0]
    elif damage == "length":
        payloads[0]["unencrypted_weights"] = np.ones(1)
    with pytest.raises(ValueError):
        asyncio.run(
            FedAvgHEAggregationStrategy().aggregate_weights(
                updates, {"layer.weight": torch.zeros(3)}, payloads, context
            )
        )
    assert context.server.encrypted_model == "previous"


def test_he_zero_sample_round_keeps_baseline(ckks):
    context = server_context(ckks)
    baseline = {"layer.weight": torch.tensor([1.0, 2.0, 3.0])}
    payload = homo_enc.encrypt_weights(
        {"layer.weight": torch.ones(3)}, context=ckks, indices=[]
    )
    result = asyncio.run(
        FedAvgHEAggregationStrategy().aggregate_weights(
            [SimpleNamespace(report=SimpleNamespace(num_samples=0))],
            baseline,
            [payload],
            context,
        )
    )
    assert result is baseline
    assert context.server.encrypted_model == "previous"


def test_encryption_does_not_reorder_caller_mask(ckks):
    indices = [2, 0]
    homo_enc.encrypt_weights({"w": torch.ones(3)}, context=ckks, indices=indices)
    assert indices == [2, 0]


@pytest.mark.parametrize("indices", [[0, 0], [-1], [3]])
def test_encryption_rejects_ambiguous_mask(ckks, indices):
    with pytest.raises(ValueError):
        homo_enc.encrypt_weights({"w": torch.ones(3)}, context=ckks, indices=indices)


def test_decrypt_rejects_excess_plaintext_instead_of_dropping_it():
    payload = homo_enc.wrap_encrypted_model(np.arange(4), None, [])
    with pytest.raises(ValueError, match="length|size"):
        homo_enc.decrypt_weights(payload, {"w": (3,)}, {"w": 3})


def test_default_he_processor_wire_pipeline_roundtrip(ckks, temp_config, monkeypatch):
    from plato.config import Config
    from plato.processors import registry

    model = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -2.0, 0.5]]))
    trainer = SimpleNamespace(model=model)
    monkeypatch.setattr(homo_enc, "get_ckks_context", lambda: ckks)
    monkeypatch.setattr(
        Config,
        "clients",
        SimpleNamespace(
            outbound_processors=["model_encrypt"], inbound_processors=["model_decrypt"]
        ),
    )
    outbound, inbound = registry.get(
        "Client",
        trainer=trainer,
        client_id=1,
        processor_kwargs={"model_encrypt": {"mask": [0, 2]}},
    )
    encoded = outbound.process(model.state_dict())
    assert isinstance(encoded, bytes)
    restored = inbound.process(encoded)
    torch.testing.assert_close(
        restored["weight"].double(), model.weight.detach().double(), atol=2e-5, rtol=0
    )
