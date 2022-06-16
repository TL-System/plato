import unittest
from park.unittest.run_env import run_env_with_random_agent


class TestSparkSim(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_name = 'spark_sim'

    def test_run_env_n_times(self, n=10):
        for _ in range(n):
            run_env_with_random_agent(self.env_name, seed=n)
