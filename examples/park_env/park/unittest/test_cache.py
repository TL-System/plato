import unittest
from park.unittest.run_env import run_env_with_random_agent
import park
from park.param import config

class TestCache(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_name = 'cache'

    def test_trace_recency(self):
        # Ensure that config for unseen recency is passed to unseen items, first observation will always be unseen
        env = park.make('cache')
        config.cache_unseen_recency = 1000
        obs = env.reset()
        self.assertTrue(obs[2] == 1000)

        config.cache_unseen_recency = 500
        obs = env.reset()
        self.assertTrue(obs[2] == 500)

    @unittest.expectedFailure
    def test_bounds(self):
        # New upper bound for the cache test traces, test trace numbers end at 999
        env = park.make('cache')
        env.reset(low=1000, high=1001)

    def test_bounds_low(self):
        # New lower bound for the cache test traces, test trace numbers start at 0
        env = park.make('cache')
        env.reset(low=0, high=1)

    def test_run_env_n_times(self, n=10):
        for _ in range(n):
            run_env_with_random_agent(self.env_name, seed=n)
