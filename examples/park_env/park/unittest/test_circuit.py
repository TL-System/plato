import unittest

from park.unittest.run_env import run_env_with_random_agent


class TestCircuit(unittest.TestCase):
    def test_evaluator(self):
        run_env_with_random_agent('circuit_three_stage_transimpedance', seed=None)
