import numpy as np
from park import core
from park.spaces.rng import np_random


class MultiDiscrete(core.Space):
    def __init__(self, nvec):
        """
        nvec: vector of counts of each categorical variable
        """
        assert (np.array(nvec) > 0).all(), 'nvec (counts) have to be positive'
        self.nvec = np.asarray(nvec, dtype=np.int64)
        core.Space.__init__(self, 'tensor_int64', self.nvec.shape, np.int64)

    def sample(self):
        return (gym.spaces.np_random.random_sample(self.nvec.shape) * self.nvec).astype(self.dtype)

    def contains(self, x):
        # if nvec is uint32 and space dtype is uint32, then 0 <= x < self.nvec guarantees that x
        # is within correct bounds for space dtype (even though x does not have to be unsigned)
        return (0 <= x).all() and (x < self.nvec).all()
