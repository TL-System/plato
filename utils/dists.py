"""
Functions that generate a variety of data distributions.
"""

import random
import numpy as np

def uniform(N, k):
    """Returns a uniform distribution of 'N' items into 'k' groups."""
    dist = []
    avg = N / k

    # Generate this distribution
    for i in range(k):
        dist.append(int((i + 1) * avg) - int(i * avg))

    # Return shuffled distribution
    random.shuffle(dist)
    return dist


def normal(N, k):
    """Returns a normal distribution of 'N' items into 'k' groups."""
    dist = []

    # Make distribution
    for i in range(k):
        x = i - (k - 1) / 2
        dist.append(int(N * (np.exp(-x) / (np.exp(-x) + 1)**2)))

    # Add remainders
    remainder = N - sum(dist)
    dist = list(np.add(dist, uniform(remainder, k)))

    # Return a non-shuffled distribution
    return dist
