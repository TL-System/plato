"""
An environment of the reinforcement learning agent for tuning parameters
during the training of federated learning.
This environment follows the gym interface, in order to use stable-baselines3:
https://github.com/DLR-RM/stable-baselines3.

To create and use other custom environments, check out:
https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html.
"""
# pylint: disable=E1101

import logging
import asyncio
import gym
from gym import spaces
import numpy as np

from config import Config


class FLEnv(gym.Env):
    """The environment of federated learning."""
    metadata = {'render.modes': ['fl']}

    def __init__(self, rl_agent):
        super(FLEnv, self).__init__()

        self.rl_agent = rl_agent
        self.state = None
        self.is_done = False
        self.rl_tuned_para_value = 0
        self.is_state_got = False
        """
        Normalize action space and make it symmetric when continuous.
        The reasons behind:
        https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment
        """
        n_actions = 1
        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       shape=(n_actions, ),
                                       dtype="float32")

        # Use only global model accurarcy as state for now
        self.n_states = 1
        # Also normalize observation space for better RL training
        self.observation_space = spaces.Box(low=-1,
                                            high=1,
                                            shape=(self.n_states, ),
                                            dtype="float32")

        self.state = [0 for i in range(self.n_states)]

    def reset(self):
        logging.info("Reseting RL environment...")

        # Let the RL agent restart FL training
        self.rl_agent.reset_env()

        self.state = [0 for i in range(self.n_states)]
        return np.array(self.state)

    def step(self, action):
        """One step of reinforcement learning."""
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))
        reward = float(0)
        self.is_done = False

        # Rescale the action from [-1, 1] to [1, 2, ... , 9]
        # The action is to choose the number of aggregations on edge servers
        #current_edge_agg_num = int((action + 2) * (action + 2))
        current_edge_agg_num = int((action + 3) / 2)

        #print('Current number of aggregations on edge servers:',current_edge_agg_num)

        self.rl_tuned_para_value = current_edge_agg_num

        self.rl_agent.get_tuned_para(self.rl_tuned_para_value)

        # Send the tuned parameter to the RL agent and get state
        current_loop = asyncio.get_event_loop()
        get_state_task_obj = current_loop.create_task(self.wait_for_state())
        current_loop.run_until_complete(get_state_task_obj)
        self.is_state_got = False

        #print('State:', self.state)
        self.normalize_state()
        #print('Normalized state:', self.state)

        reward = self.get_reward()
        info = {}
        return np.array([self.state]), reward, self.is_done, info

    def start_rl_agent(self):
        """Startup function for a RL agent."""
        logging.info("Start a RL agent on the central server...")
        tuned_para = {
            'edge_agg_num': 'number of aggregations on edge servers',
        }[Config().rl.tuned_para]
        logging.info("This RL agent will tune the %s.", tuned_para)

    async def wait_for_state(self):
        """Wait for getting the current state."""
        while not self.is_state_got:
            await asyncio.sleep(1)

    def get_state_from_rl_agent(self, state, is_done):
        """
        Get transitted state from RL agent.
        This function is called by RL agent.
        """
        self.state = state
        self.is_done = is_done
        self.is_state_got = True

    def normalize_state(self):
        """Normalize each element of state."""
        self.state = 2 * (self.state - 0.5)

    def get_reward(self):
        """Get reward based on the state."""
        accuracy = self.state
        # Use accuracy as reward for now.
        reward = accuracy
        return reward

    def render(self, mode='rl'):
        pass

    def close(self):
        pass
