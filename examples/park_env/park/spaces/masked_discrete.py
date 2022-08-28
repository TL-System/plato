import numpy as np
from park import core
from park.spaces.rng import np_random


class MaskedDiscrete(core.Space):
    """
    {0,1,...,n-1}
    With a mask (list of numbers) for eligible entries.
    Ex: [0, 2, 3, 7, 8], 'None' for all entries eligible
    """
    def __init__(self, n):
        self.n = n
        self.mask = None
        core.Space.__init__(self, 'tensor_int64', (), np.int64)

    def sample(self):
        if self.mask is None:
            return np_random.randint(self.n)
        elif len(self.mask) > 0:
            num_eligible_entries = len(self.mask)
            return self.mask[np_random.randint(num_eligible_entries)]
        else:
            return None  # no eligible entries to sample

    def update_mask(self, mask):
        self.mask = mask

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        if self.mask is None:
            return as_int >= 0 and as_int < self.n
        else:
            # TODO: use set data structure will make this O(1)
            return as_int in self.mask
