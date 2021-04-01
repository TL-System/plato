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


def randomize_obj(bit_array: np.ndarray, targets: np.ndarray, epsilon):
    """
    The default unary encoding method is symmetric.
    """
    assert isinstance(bit_array, np.ndarray)
    img = symmetric_unary_encoding(bit_array, 1)
    label = symmetric_unary_encoding(bit_array, epsilon)
    for i in range(targets.shape[1]):
        box = convert(bit_array.shape[2:], targets[0][i][2:])
        img[:,:,box[0]:box[2],box[1]:box[3]] = label[:,:,box[0]:box[2],box[1]:box[3]]
    return img

def randomize(bit_array: np.ndarray, epsilon):
    """
    The default unary encoding method is symmetric.
    """
    assert isinstance(bit_array, np.ndarray)
    return symmetric_unary_encoding(bit_array, epsilon)


def symmetric_unary_encoding(bit_array: np.ndarray, epsilon):
    p = np.e**(epsilon / 2) / (np.e**(epsilon / 2) + 1)
    q = 1 / (np.e**(epsilon / 2) + 1)
    return produce_random_response(bit_array, p, q)


def optimized_unary_encoding(bit_array: np.ndarray, epsilon):
    p = 1 / 2
    q = 1 / (np.e**epsilon + 1)
    return produce_random_response(bit_array, p, q)


def produce_random_response(bit_array: np.ndarray, p, q=None):
    """Implements random response as the perturbation method."""
    q = 1 - p if q is None else q

    p_binomial = np.random.binomial(1, p, bit_array.shape)
    q_binomial = np.random.binomial(1, q, bit_array.shape)
    return np.where(bit_array == 1, p_binomial, q_binomial)

def convert(size, box): # size:(w,h) , box:(xmin,xmax,ymin,ymax)
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    x1 = max(x - 0.5 * w - 3, 0)
    x2 = min(x + 0.5 * w + 3,  size[0])
    y1 = max(y - 0.5 * h - 3, 0)
    y2 = min(y + 0.5 * h + 3, size[1])

    x1 = round(x1 * size[0])
    x2 = round(x2 * size[0])
    y1 = round(y1 * size[1])
    y2 = round(y2 * size[1])

    return (x1,y1,x2,y2)