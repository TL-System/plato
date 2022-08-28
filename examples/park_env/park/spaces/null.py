from park import core


class Null(core.Space):
    '''
    Nullspace
    '''
    def __init__(self, low=None, high=None, struct=None, shape=None, dtype=None):
        core.Space.__init__(self, None, None, None)

    def sample(self):
        return None

    def contains(self, x):
        return x is None
