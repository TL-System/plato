"""Useful decorators."""

import time
from functools import wraps


def timeit(func_timed):
    """Measures the time elapsed for a particular function 'func_timed'."""

    @wraps(func_timed)
    def timed(*args, **kwargs):
        started = time.perf_counter()
        output = func_timed(*args, **kwargs)
        ended = time.perf_counter()
        elapsed = ended - started
        print(f'"{func_timed.__name__}" took {elapsed:.2f} seconds to execute.')
        if output is None:
            return elapsed
        else:
            return output, elapsed

    return timed
