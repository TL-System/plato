"""Tests for unary encoding local differential privacy utilities."""

import numpy as np

from plato.utils import unary_encoding


def _unary_epsilon(p, q):
    """Compute epsilon from probabilities p and q."""
    return np.log((p * (1 - q)) / ((1 - p) * q))


def test_symmetric_encoding_matches_randomised_response():
    """Symmetric unary encoding should reproduce direct randomised response."""
    p = 0.75
    q = 0.25
    epsilon = _unary_epsilon(p, q)

    np.random.seed(1)
    values = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    symmetric = unary_encoding.symmetric_unary_encoding(values, epsilon)

    np.random.seed(1)
    randomized = unary_encoding.produce_randomized_response(values, p, q)

    assert symmetric.tolist() == randomized.tolist()


def test_randomised_response_probability_matches_expectation():
    """The fraction of ones should approximate p within a tight tolerance."""
    p = 0.75
    runs = 100_000
    values = np.ones(runs, dtype=int)

    np.random.seed(7)
    randomized = unary_encoding.produce_randomized_response(values, p)
    observed = (randomized == 1).mean()

    assert np.isclose(observed, p, atol=0.005)
