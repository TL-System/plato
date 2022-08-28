import numpy as np


"""
Separate the random number generator from the environment.
This is used for all random sample in the space native methods.
We expect new algorithms to have their own rngs.
"""

np_random = np.random.RandomState()
np_random.seed(42)
