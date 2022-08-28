import multiprocessing
import multiprocessing.pool
import os

__all__ = ['make_pool']


def make_pool(mode, workers, propagate_process_signal=False) -> multiprocessing.pool.Pool:
    assert mode in ('thread', 'process'), 'mode can only be thread or process'
    if mode == 'thread':
        return multiprocessing.pool.ThreadPool(workers)
    else:
        return multiprocessing.Pool(workers, initializer=None if propagate_process_signal else os.setpgrp)
