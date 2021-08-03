""" Useful decorators. """
import time
from functools import wraps


def timeit(func_timed):
    """ Measures the time elapsed for a particular function 'func_timed'. """
    @wraps(func_timed)
    def timed(*args, **kwargs):
        started = time.perf_counter()
        output = func_timed(*args, **kwargs)
        ended = time.perf_counter()
        elapsed = ended - started
        print('"{}" took {:.3f} seconds to execute\n'.format(func_timed.__name__, elapsed))
        return output, elapsed
    return timed
