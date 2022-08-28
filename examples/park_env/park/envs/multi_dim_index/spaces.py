import random
from park.core import Space
from park.envs.multi_dim_index.params import Params as params
from park.envs.multi_dim_index.config import Action, Query

class ActionSpace(Space):
    def sample(self):
        n = random.randint(1, params.NDIMS)
        dims = random.sample(range(params.NDIMS), n)
        cols = []
        for i in range(n-1):
            cols.append(random.randint(1, 100))
        return Action(dims, cols)

    def contains(self, a):
        valid = True
        valid &= (len(a.dimensions) <= params.NDIMS)
        # Make sure no dimensions are duplicated in the grid list (except the sort dimension
        # can be the same as a grid dimension).
        valid &= len(set(a.dimensions[:-1])) == len(a.dimensions[:-1])
        for d in a.dimensions:
            valid &= isinstance(d, int)
            valid &= (d < params.NDIMS) and (d >= 0)
        valid &= (len(a.columns) == len(a.dimensions)-1)
        for c in a.columns:
            valid &= isinstance(c, int)
            valid &= (c > 0)
        return valid

class DataObsSpace(Space):
    def sample(self):
        pass

    def contains(self, s):
        return s.data_iterator is not None

class QueryObsSpace(Space):
    def sample(self):
        pass

    def contains(self, s):
        return isinstance(s, Query) and s.valid()


