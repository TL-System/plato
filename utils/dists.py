"""
Functions that generate a variety of data distributions.
"""

import random
import numpy as np

def uniform(N, k):
    """Returns a uniform distribution of 'N' items into 'k' groups."""
    random.seed()

    dist = []
    avg = N / k

    # Generate this distribution
    for i in range(k):
        dist.append(int((i + 1) * avg) - int(i * avg))

    # Return shuffled distribution
    samples = np.random.uniform(N)
    random.shuffle(dist)
    return dist, samples


def normal(N, k):
    """
    Generates a histogram for a normal distribution, partitioning 'N' items into 'k' groups.
    """
    random.seed()

    samples = np.random.normal(0, 1, N)
    dist, __ = np.histogram(samples, bins=k)
    return dist, samples
