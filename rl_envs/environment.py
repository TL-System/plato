"""
An environment of the reinforcement learning agent for tuning parameters
during the training of federated learning.
This environment follows the gym interface, in order to use stable-baselines3:
https://github.com/DLR-RM/stable-baselines3.

To create and use other custom environments, check out:
https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html.
"""

import logging
import asyncio
import sys
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
        self.time_step = 0
        self.state = None
        self.is_episode_done = False
        self.is_state_got = False
        self.is_step_done = False
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
        if self.rl_agent.rl_episode >= Config().rl.episodes:
            while True:
                # Give some time (2 seconds) to close connections
                current_loop = asyncio.get_event_loop()
                task = current_loop.create_task(asyncio.sleep(2))
                current_loop.run_until_complete(task)
                sys.exit()

        logging.info("Reseting RL environment...")

        self.time_step = 0
        # Let the RL agent restart FL training
        self.rl_agent.reset_rl_env()

        self.state = [0 for i in range(self.n_states)]
        return np.array(self.state)

    def step(self, action):
        """One step of reinforcement learning."""
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))
        self.time_step += 1
        reward = float(0)
        self.is_episode_done = False

        # Rescale the action from [-1, 1] to [1, 2, ... , 9]
        # The action is to choose the number of aggregations on edge servers
        #current_edge_agg_num = int((action + 2) * (action + 2))
        current_edge_agg_num = self.time_step

        print('Number of aggregations on edge servers of time step',
              self.time_step, 'is', current_edge_agg_num)

        # Pass the tuned parameter to RL agent
        self.rl_agent.get_tuned_para(current_edge_agg_num, self.time_step)

        # Wait for state
        current_loop = asyncio.get_event_loop()
        get_state_task = current_loop.create_task(
            self.wait_for_state(self.time_step))
        current_loop.run_until_complete(get_state_task)

        #print('State:', self.state)
        self.normalize_state()
        #print('Normalized state:', self.state)

        reward = self.get_reward()
        info = {}

        self.is_step_done = True
        return np.array([self.state]), reward, self.is_episode_done, info

    async def wait_for_state(self, time_step):
        """Wait for getting the current state."""
        print("RL env: Start waiting for state of time step", time_step)
        while not self.is_state_got:
            await asyncio.sleep(1)
            self.is_step_done = False
        self.is_state_got = False
        print("RL env: Stop waiting for state of time step", time_step)

    def get_state(self, state, is_episode_done):
        """
        Get transitted state from RL agent.
        This function is called by RL agent.
        """
        self.state = state
        self.is_episode_done = is_episode_done
        self.is_state_got = True
        print("RL env: Get state", state)
        self.rl_agent.is_rl_tuned_para_got = False

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
