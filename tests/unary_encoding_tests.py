"""
Unit tests for unary encoding, a local differential privacy mechanism that adds
noise to model weights or features before transmitting to the federated learning server.
"""

import unittest

import numpy as np

from plato.utils import unary_encoding


class UnaryEncodingTest(unittest.TestCase):
    """Tests for unary encoding and randomized response."""

    @staticmethod
    def unary_epsilon(p, q):
        """Computes epsilon from p and q.

        Reference:

        Wang, et al. "Optimizing Locally Differentially Private Protocols," ATC USENIX 2017.
        """
        return np.log((p * (1 - q)) / ((1 - p) * q))

    def test_epsilon_computation(self):
        """Test the correctness of computing p and q from a given epsilon."""
        p = 0.75
        q = 0.25
        computed_epsilon = UnaryEncodingTest.unary_epsilon(p, q)

        np.random.seed(1)
        arr = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        symmetric = unary_encoding.symmetric_unary_encoding(arr, computed_epsilon)

        np.random.seed(1)
        randomized_response = unary_encoding.produce_randomized_response(arr, p, q)
        self.assertSequenceEqual(symmetric.tolist(), randomized_response.tolist())

    def test_distribution_probability(self):
        """Test the distribution probability of the results."""
        p = 0.75
        runs = 100000
        arr = np.array([1] * runs)
        symmetric = unary_encoding.produce_randomized_response(arr, p)
        total_ones = (symmetric == 1).sum()
        print(f"Probability of ones = {total_ones / len(symmetric.tolist())}")
        self.assertAlmostEqual(total_ones / len(symmetric.tolist()), p, delta=0.005)


if __name__ == "__main__":
    unittest.main()
