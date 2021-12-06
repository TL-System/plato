"""Implements unary encoding, used by Google's RAPPOR, as the local differential privacy mechanism.

References:

Wang, et al. "Optimizing Locally Differentially Private Protocols," ATC USENIX 2017.

Erlingsson, et al. "RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response,"
ACM CCS 2014.

"""

import numpy as np


def encode(x: np.ndarray):
    x[x > 0] = 1
    x[x <= 0] = 0
    return x


def randomize(bit_array: np.ndarray, epsilon):
    """
    The default unary encoding method is symmetric.
    """
    assert isinstance(bit_array, np.ndarray)
    return symmetric_unary_encoding(bit_array, epsilon)


def symmetric_unary_encoding(bit_array: np.ndarray, epsilon):
    p = np.e**(epsilon / 2) / (np.e**(epsilon / 2) + 1)
    q = 1 / (np.e**(epsilon / 2) + 1)
    return produce_randomized_response(bit_array, p, q)


def optimized_unary_encoding(bit_array: np.ndarray, epsilon):
    p = 1 / 2
    q = 1 / (np.e**epsilon + 1)
    return produce_randomized_response(bit_array, p, q)


def produce_randomized_response(bit_array: np.ndarray, p, q=None):
    """Implements randomized response as the perturbation method."""
    q = 1 - p if q is None else q

    p_binomial = np.random.binomial(1, p, bit_array.shape)
    q_binomial = np.random.binomial(1, q, bit_array.shape)
    return np.where(bit_array == 1, p_binomial, q_binomial)
