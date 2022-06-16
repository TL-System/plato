import sys
sys.path.append('/home/ubuntu/park')
import unittest
from park.unittest.run_env import run_env_with_random_agent


class TestMultiDimIndex(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_name = 'multi_dim_index'

    def test_run_env_n_times(self, n=10):
        for _ in range(n):
            run_env_with_random_agent(self.env_name, seed=n)

TestMultiDimIndex().test_run_env_n_times(1)

