"""Tests for Safetensor-based payload serialization helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plato.processors import safetensor_decode, safetensor_encode
from plato.serialization.safetensor import deserialize_tree, serialize_tree


def _sample_tree() -> dict:
    rng = np.random.default_rng(42)
    return {
        "layer1": {
            "weight": rng.normal(size=(4, 3, 2)).astype(np.float32),
            "bias": rng.normal(size=(4,)).astype(np.float32),
        },
        "layer2": [
            rng.uniform(size=(2, 5)).astype(np.float32),
            rng.integers(0, 10, size=(5,), dtype=np.int32),
        ],
        "layer3": (
            rng.normal(size=(3, 3)).astype(np.float64),
            rng.normal(size=(3,)).astype(np.float32),
        ),
    }


def _assert_trees_allclose(actual, expected) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual.keys()) == set(expected.keys())
        for key in expected:
            _assert_trees_allclose(actual[key], expected[key])
        return

    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_trees_allclose(left, right)
        return

    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_trees_allclose(left, right)
        return
    if expected is None:
        assert actual is None
        return
    if isinstance(expected, str):
        assert actual == expected
        return
    if isinstance(expected, bytes):
        assert isinstance(actual, bytes)
        assert actual == expected
        return

    np.testing.assert_allclose(actual, expected)
    assert actual.dtype == expected.dtype


def test_serialize_tree_roundtrip_preserves_structure():
    tree = _sample_tree()
    blob = serialize_tree(tree)

    restored = deserialize_tree(blob)

    _assert_trees_allclose(restored, tree)


@pytest.mark.parametrize("buffer_type", [bytes, bytearray, memoryview])
def test_processors_encode_decode_roundtrip(buffer_type):
    tree = _sample_tree()

    encoder = safetensor_encode.Processor()
    decoder = safetensor_decode.Processor()

    encoded = encoder.process(tree)
    assert isinstance(encoded, bytes)

    buffer = buffer_type(encoded)
    decoded = decoder.process(buffer)

    _assert_trees_allclose(decoded, tree)


def test_serialize_tree_handles_strings_and_none():
    tree = (
        None,
        "prompt",
        {"meta": b"bytes", "values": [np.array(1.0, dtype=np.float32)]},
    )

    blob = serialize_tree(tree)
    restored = deserialize_tree(blob)

    _assert_trees_allclose(restored, tree)


def test_serialize_tree_handles_root_level_leaf():
    leaf = np.arange(5, dtype=np.int64)

    blob = serialize_tree(leaf)
    restored = deserialize_tree(blob)

    np.testing.assert_array_equal(restored, leaf)
    assert restored.dtype == leaf.dtype


def test_serialize_tree_roundtrip_preserves_torch_bfloat16_tensors():
    tree = {
        "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.bfloat16)
    }

    blob = serialize_tree(tree)
    restored = deserialize_tree(blob)

    assert isinstance(restored["weight"], torch.Tensor)
    assert restored["weight"].dtype == torch.bfloat16
    assert torch.equal(restored["weight"], tree["weight"])


@pytest.mark.parametrize(
    "tree",
    [
        {"a.b": np.array(1), "a": {"b": np.array(2)}},
        {"a": {"b": np.array(2)}, "a.b": np.array(1)},
        {"a[0]": np.array(1), "a": [np.array(2)]},
        {"a": [np.array(2)], "a[0]": np.array(1)},
        {"a": {"[0]": np.array(1)}, "a[0]": np.array(2)},
        {"a.b": {}, "a": {"b": np.array(2)}},
        {"": np.array(1)},
        {1: np.array(1), "1": np.array(2)},
        {"_tree_metadata": np.array([1, 2])},
    ],
)
def test_ambiguous_tree_is_rejected_before_encoding(tree):
    with pytest.raises(ValueError, match="[Aa]mbiguous|reserved"):
        safetensor_encode.Processor().process(tree)


def test_dotted_state_dict_and_unambiguous_brackets_remain_compatible():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    tree = {
        "model": model.state_dict(),
        "literal.dot": np.arange(3, dtype=np.int16),
        "literal[0]": torch.tensor([True, False]),
        "nested": ([], {}, [torch.tensor(2.0, dtype=torch.float64)]),
    }
    restored = deserialize_tree(serialize_tree(tree))
    assert set(restored) == set(tree)
    for key, expected in tree["model"].items():
        assert restored["model"][key].dtype == expected.dtype
        assert restored["model"][key].shape == expected.shape
        assert torch.equal(restored["model"][key], expected)
    np.testing.assert_array_equal(restored["literal.dot"], tree["literal.dot"])
    assert restored["literal.dot"].dtype == np.int16
    assert torch.equal(restored["literal[0]"], tree["literal[0]"])
    assert restored["nested"][:2] == ([], {})
    assert restored["nested"][2][0].dtype == torch.float64
    assert restored["nested"][2][0].item() == 2.0


def test_noncontiguous_torch_tree_roundtrip():
    tensor = torch.arange(12, dtype=torch.int64).reshape(3, 4).T
    restored = deserialize_tree(serialize_tree({"tensor": tensor}))
    assert torch.equal(restored["tensor"], tensor)
    assert restored["tensor"].dtype == tensor.dtype


def test_native_scalar_types_roundtrip_in_hybrid_payload_metadata():
    tree = {"indices": [0, 2], "scale": 1.25, "enabled": True}
    restored = deserialize_tree(serialize_tree(tree))
    assert restored == tree
    assert all(type(value) is int for value in restored["indices"])
    assert type(restored["scale"]) is float
    assert type(restored["enabled"]) is bool
