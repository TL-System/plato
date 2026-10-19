"""Unit tests for processor pipelines and pruning utilities."""

from collections import OrderedDict
from types import SimpleNamespace
from typing import cast

import torch

from plato.processors import (
    model_compress,
    model_decompress,
    model_dequantize_int4,
    model_quantize,
    model_quantize_int4,
    pipeline,
    structured_pruning,
    unstructured_pruning,
)


def test_quantize_then_compress_pipeline_roundtrip():
    """Quantization should run before compression inside a processor pipeline."""
    torch.manual_seed(0)
    state = OrderedDict(
        {"weight": torch.randn(2, 2, dtype=torch.float32), "bias": torch.ones(2)}
    )

    quant = model_quantize.Processor(name="quantize", client_id=1)
    compress = model_compress.Processor(
        name="compress", client_id=1, compression_level=3
    )
    pipe = pipeline.Processor([quant, compress])

    compressed = pipe.process(state)
    decompressor = model_decompress.Processor(name="decompress", client_id=1)
    roundtrip = decompressor.process(compressed)

    assert roundtrip["weight"].dtype == torch.bfloat16
    assert torch.allclose(
        roundtrip["weight"].to(torch.float32), state["weight"], atol=1e-2
    )
    assert torch.equal(roundtrip["bias"], state["bias"])


def test_int4_processor_pipeline_roundtrip():
    """INT4 quantization and dequantization pipeline roundtrip."""
    torch.manual_seed(42)
    state = OrderedDict(
        {
            "weight": torch.randn(4, 8, dtype=torch.float32),
            "bias": torch.zeros(4, dtype=torch.int64),
        }
    )

    quant = model_quantize_int4.Processor(name="quantize_int4", client_id=1)
    dequant = model_dequantize_int4.Processor(name="dequantize_int4", server_id=1)

    quantized = quant.process(state)
    assert isinstance(quantized["weight"], bytes)
    assert torch.equal(quantized["bias"], state["bias"])

    restored = dequant.process(quantized)
    assert restored["weight"].shape == state["weight"].shape
    assert torch.equal(restored["bias"], state["bias"])
    assert torch.mean((restored["weight"] - state["weight"]) ** 2).item() < 0.05



def test_structured_pruning_zeroes_all_channels():
    """Structured pruning with amount=1 should prune every structured channel."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    trainer = SimpleNamespace(model=model)
    layer = cast(torch.nn.Linear, model[0])
    bias = layer.bias
    assert bias is not None
    original_bias = bias.detach().cpu().clone()

    processor = structured_pruning.Processor(
        name="structured",
        trainer=trainer,
        amount=1.0,
        dim=0,
        pruning_method="ln",
        norm=2,
        client_id=1,
    )

    pruned_state = processor.process(model.state_dict())
    weight = pruned_state["0.weight"]
    assert torch.equal(weight, torch.zeros_like(weight))
    assert torch.equal(pruned_state["0.bias"], original_bias)


def test_unstructured_pruning_removes_fraction_of_weights():
    """Global unstructured pruning should set the lowest-magnitude weights to zero."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
    trainer = SimpleNamespace(model=model)

    processor = unstructured_pruning.Processor(
        name="unstructured",
        trainer=trainer,
        amount=0.5,
        client_id=1,
    )

    pruned_state = processor.process(model.state_dict())
    weight = pruned_state["0.weight"]
    zero_fraction = (weight == 0).sum().item() / weight.numel()
    assert zero_fraction >= 0.5
