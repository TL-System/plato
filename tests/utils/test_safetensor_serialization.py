"""Tests for Safetensor-based payload serialization helpers."""

from __future__ import annotations

import numpy as np
import pytest

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
