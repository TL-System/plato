import numpy as np
from park import core
from park.spaces.rng import np_random

class PowerSet(core.Space):
    """
    The element of this space is an element in the power set of the given set.
    The space can change its size during execution (e.g., a node
    being added into the graph, an edge being deleted from a graph)
    """
    def __init__(self, init_set):
        self.set = init_set
        core.Space.__init__(self, 'set', (), None)

    def add(self, elements):
        for e in elements:
            self.set.add(e)

    def delete(self, elements):
        # the element has to be in the set already
        for e in elements:
            self.set.remove(e)

    def update(self, new_set):
        self.set = new_set

    def sample(self, valid_list=None):
        # TODO: need a better implementation for O(1) sampling
        tmp_list = list(self.set)
        idx = np_random.randint(len(tmp_list))
        return tmp_list[idx]

    def contains(self, x):
        '''
        @x: if it is an array, then check if each element is part of this
        space. If it is a set, then just check if that is a part of the space
        '''
        if isinstance(x, set):
            return x.issubset(self.set)
        elif isinstance(x, np.ndarray):
            for xi in x:
                if not xi.issubset(self.set):
                    return False
            return True

