"""
An environment of the reinforcement learning agent for tuning parameters
during the training of federated learning.
This environment follows the gym interface, in order to use stable-baselines3:
https://github.com/DLR-RM/stable-baselines3.

To create and use other custom environments, check out:
https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html.
"""

import asyncio
import logging

import gym
import numpy as np
from plato.config import Config
from gym import spaces


class RLEnv(gym.Env):
    """The environment of federated learning."""
    metadata = {'render.modes': ['fl']}

    def __init__(self, rl_agent):
        super().__init__()

        self.rl_agent = rl_agent
        self.time_step = 0
        self.state = None
        self.is_episode_done = False

        # An RL env waits for the event that it gets the current state from RL agent
        self.state_got = asyncio.Event()

        # An RL agent waits for the event that the RL env finishes step()
        # so that it can start a new FL round
        self.step_done = asyncio.Event()

        # Normalize action space and make it symmetric when continuous.
        # The reasons behind:
        # https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment
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
        if self.rl_agent.rl_episode >= Config().algorithm.rl_episodes:
            while True:
                # Give RL agent some time to close connections and exit
                current_loop = asyncio.get_event_loop()
                task = current_loop.create_task(asyncio.sleep(1))
                current_loop.run_until_complete(task)

        logging.info("Reseting RL environment.")

        self.time_step = 0
        # Let the RL agent restart FL training
        self.rl_agent.reset_rl_env()

        self.rl_agent.new_episode_begin.set()

        self.state = [0 for i in range(self.n_states)]
        return np.array(self.state)

    def step(self, action):
        """One step of reinforcement learning."""
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))
        self.time_step += 1
        reward = float(0)
        self.is_episode_done = False

        # For testing code
        current_edge_agg_num = self.time_step

        # Rescale the action from [-1, 1] to [1, 2, ... , 9]
        # The action is the number of aggregations on edge servers
        # current_edge_agg_num = int((action + 2) * (action + 2))

        logging.info("RL Agent: Start time step #%s...", self.time_step)
        logging.info(
            "Each edge server will run %s rounds of local aggregation.",
            current_edge_agg_num)

        # Pass the tuned parameter to RL agent
        self.rl_agent.get_tuned_para(current_edge_agg_num, self.time_step)

        # Wait for state
        current_loop = asyncio.get_event_loop()
        get_state_task = current_loop.create_task(self.wait_for_state())
        current_loop.run_until_complete(get_state_task)
        #print('State:', self.state)

        self.normalize_state()
        #print('Normalized state:', self.state)

        reward = self.get_reward()
        info = {}

        self.rl_agent.cumulative_reward += reward

        # Signal the RL agent to start next time step (next round of FL)
        self.step_done.set()

        return np.array([self.state]), reward, self.is_episode_done, info

    async def wait_for_state(self):
        """Wait for getting the current state."""
        await self.state_got.wait()
        assert self.time_step == self.rl_agent.current_round
        self.state_got.clear()

    def get_state(self, state, is_episode_done):
        """
        Get transitted state from RL agent.
        This function is called by RL agent.
        """
        self.state = state
        self.is_episode_done = is_episode_done
        # Signal the RL env that it gets the current state
        self.state_got.set()
        print("RL env: Get state", state)
        self.rl_agent.is_rl_tuned_para_got = False

    def normalize_state(self):
        """Normalize each element of state to [-1,1]."""
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
