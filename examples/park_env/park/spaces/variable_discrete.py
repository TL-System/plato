import numpy as np
from park import core
from park.spaces.rng import np_random


class VariableDiscrete(core.Space):
    """
    The size of the space is changing in each step
    """
    def __init__(self):
        self.n = None
        core.Space.__init__(self, 'tensor_int64', (), np.int64)

    def update(self, n):
        self.n = n

    def sample(self):
        return np_random.randint(self.n)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n
