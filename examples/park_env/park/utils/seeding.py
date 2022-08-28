import numpy as np


def np_random(seed=42):
    if not (isinstance(seed, int) and seed >= 0):
        raise ValueError('Seed must be a non-negative integer.')
    rng = np.random.RandomState()
    rng.seed(seed)
    return rng
