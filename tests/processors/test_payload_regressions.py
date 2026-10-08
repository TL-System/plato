"""Payload fidelity and explicit failures at processor boundaries."""

import random
from struct import pack
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from plato.config import Config
from plato.processors import (
    compress,
    decompress,
    feature_additive_noise,
    model_dequantize,
    model_dequantize_qsgd,
    model_quantize,
    model_quantize_qsgd,
    model_randomized_response,
    unstructured_pruning,
)


def test_bfloat_quantization_preserves_integer_and_boolean_buffers():
    state = {"counter": torch.tensor(16777219), "mask": torch.tensor([True, False])}
    encoded = model_quantize.Processor(client_id=1).process(state)
    restored = model_dequantize.Processor(client_id=1).process(encoded)
    for key in state:
        assert encoded[key].dtype == state[key].dtype
        assert restored[key].dtype == state[key].dtype
        assert torch.equal(restored[key], state[key])


@pytest.mark.parametrize(
    "tensor", [torch.zeros(3), torch.empty(0, 2), torch.tensor(0.0)]
)
def test_qsgd_zero_and_empty_layers_roundtrip(tensor):
    encoded = model_quantize_qsgd.Processor(client_id=1).process({"w": tensor})
    restored = model_dequantize_qsgd.Processor(client_id=1).process(encoded)
    assert restored["w"].shape == tensor.shape
    assert torch.equal(restored["w"], tensor)


def test_qsgd_has_bounded_error_and_respects_random_seed():
    tensor = torch.tensor([[-3.2, -0.04, 0, 1.2], [2.0, -1.1, 0.75, 3.2]]).T
    encoder = model_quantize_qsgd.Processor(client_id=1, quantization_level=16)
    decoder = model_dequantize_qsgd.Processor(client_id=1, quantization_level=16)
    random.seed(17)
    first = encoder.process({"w": tensor})
    random.seed(17)
    second = encoder.process({"w": tensor})
    assert first == second
    restored = decoder.process(first)["w"]
    assert restored.shape == tensor.shape
    assert torch.max(torch.abs(restored - tensor)) <= 3.2 / 15 + 1e-6


@pytest.mark.parametrize("level", [2, 16, 64, 128])
@pytest.mark.parametrize("scale", [3.75, 1e38, 3e38])
def test_qsgd_finite_endpoints_and_error_bound_at_large_scales(level, scale):
    tensor = torch.tensor([scale, -scale, 0, 0.2 * scale], dtype=torch.float32)
    encoder = model_quantize_qsgd.Processor(client_id=1, quantization_level=level)
    decoder = model_dequantize_qsgd.Processor(client_id=1, quantization_level=level)
    random.seed(17)
    restored = decoder.process(encoder.process({"w": tensor}))["w"]
    assert restored.dtype == torch.float32
    assert torch.isfinite(restored).all()
    # Endpoints have no rounding uncertainty. Compare in float64 so the
    # independent error calculation cannot overflow at these valid scales.
    torch.testing.assert_close(
        restored[:3].double(), tensor[:3].double(), atol=0, rtol=1e-7
    )
    scale64 = tensor.abs().max().double()
    tolerance = scale64 * torch.finfo(torch.float32).eps * 2
    assert (restored.double() - tensor.double()).abs().max() <= (
        scale64 / (level - 1) + tolerance
    )


@pytest.mark.parametrize("level", [0, 1, 129])
def test_qsgd_invalid_levels_fail_explicitly(level):
    with pytest.raises(ValueError, match="level"):
        model_quantize_qsgd.Processor(quantization_level=level)
    with pytest.raises(ValueError, match="level"):
        model_dequantize_qsgd.Processor(quantization_level=level)


@pytest.mark.parametrize(
    "blob",
    [
        b"",
        pack("!fIh", 1, 1000000, 1) + pack("!h", 1),
        pack("!fIh", 1, 1, 0) + b"\x01trailing",
    ],
)
def test_qsgd_malformed_bytes_fail_explicitly(blob):
    with pytest.raises(ValueError):
        model_dequantize_qsgd.Processor(client_id=1).process({"w": blob})


def test_numpy_feature_compression_preserves_empty_target_dtype():
    features = np.arange(12, dtype=np.float32).reshape(3, 4)
    data = [(features, np.empty((0,), dtype=np.int64))]
    result = decompress.Processor().process(compress.Processor().process(data))
    np.testing.assert_array_equal(result[0][0], features)
    assert result[0][1].shape == (0,)
    assert result[0][1].dtype == np.int64


def test_numpy_compression_empty_batch_and_strided_array():
    encoder, decoder = compress.Processor(), decompress.Processor()
    assert decoder.process(encoder.process([])) == []
    array = np.arange(12, dtype=np.int16).reshape(3, 4).T
    np.testing.assert_array_equal(decoder.process(encoder.process(array)), array)


def test_heterogeneous_feature_batches_are_rejected_before_compression():
    data = [(np.ones((2, 3)), np.ones(2)), (np.ones((1, 3)), np.ones(1))]
    with pytest.raises(ValueError, match="shape|dtype"):
        compress.Processor().process(data)


def test_randomized_response_does_not_modify_source_model(temp_config, monkeypatch):
    monkeypatch.setattr(Config, "algorithm", SimpleNamespace(epsilon=0))
    tensor = torch.tensor([-2.0, 0.2, 4.0])
    before = tensor.clone()
    result = model_randomized_response.Processor(client_id=1).process({"w": tensor})
    assert torch.equal(tensor, before)
    assert set(result["w"].tolist()) <= {0, 1}


def test_pruning_uses_replaced_trainer_model():
    trainer = SimpleNamespace(model=torch.nn.Linear(4, 2, bias=False))
    processor = unstructured_pruning.Processor(trainer=trainer, client_id=1, amount=0.5)
    processor.process(trainer.model.state_dict())
    trainer.model = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        trainer.model.weight.copy_(torch.arange(1, 9).reshape(2, 4))
    result = processor.process(trainer.model.state_dict())
    expected = torch.tensor([[0.0, 0, 0, 0], [5.0, 6, 7, 8]])
    assert torch.equal(result["weight"], expected)


def test_zero_noise_feature_processor_keeps_cpu_precision_and_targets():
    logits = torch.tensor([[1.0001, -0.33333]], dtype=torch.float32)
    targets = torch.tensor([2])
    trainer = SimpleNamespace(device=torch.device("cpu"))
    processor = feature_additive_noise.Processor(
        method="gaussian", scale=0, trainer=trainer, client_id=1
    )
    restored = processor.process([(logits, targets)])
    assert torch.equal(restored[0][0], logits)
    assert restored[0][0].dtype == torch.float32
    assert restored[0][1] is targets
