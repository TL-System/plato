"""
A basic RL environment for FL server using Gym for RL control.
"""
import asyncio
import logging
import os
from abc import abstractmethod

import numpy as np
from gym import spaces
from plato.config import Config
from plato.utils import csv_processor


class RLAgent(object):
    """ A basic RL environment for FL server using Gym for RL control. """
    def __init__(self):
        self.n_actions = Config().clients.per_round
        self.n_states = Config().clients.per_round * Config(
        ).algorithm.n_features

        if Config().algorithm.discrete_action_space:
            self.action_space = spaces.Discrete(self.n_actions)
        else:
            self.action_space = spaces.Box(low=int(
                Config().algorithm.min_action),
                                           high=Config().algorithm.max_action,
                                           shape=(self.n_actions, ),
                                           dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(self.n_states, ),
                                            dtype=np.float32)

        self.state = None
        self.next_state = None
        self.new_state = None
        self.action = None
        self.next_action = None
        self.reward = 0
        self.episode_reward = 0
        self.current_step = 0
        self.total_steps = 0
        self.current_episode = 0
        self.is_done = False
        self.reset_env = False
        self.finished = False

        # RL server waits for the event that the next action is updated
        self.action_updated = asyncio.Event()

    def step(self):
        """ Update the followings using server update. """
        self.new_state = self.get_state()
        self.is_done = self.get_done()
        self.reward = self.get_reward()
        info = self.get_info()

        return self.new_state, self.reward, self.is_done, info

    async def reset(self):
        """ Reset RL environment. """
        # Start a new training session
        logging.info("[RL Agent] Reseting RL environment.")

        # Reset the episode-related variables
        self.current_step = 0
        self.is_done = False
        self.episode_reward = 0
        self.current_episode += 1
        self.reset_env = True
        logging.info("[RL Agent] Starting RL episode #%d.",
                     self.current_episode)

    def prep_action(self):
        """ Get action from RL policy. """
        logging.info("[RL Agent] Selecting action...")
        self.action = self.policy.select_action(self.state)

    def get_state(self):
        """ Get state for agent. """
        return self.new_state

    def get_reward(self):
        """ Get reward for agent. """
        return 0.0

    def get_done(self):
        """ Get done condition for agent. """
        if Config().algorithm.mode == 'train' and self.current_step >= Config(
        ).algorithm.steps_per_episode:
            logging.info("[RL Agent] Episode #%d ended.", self.current_episode)
            return True
        return False

    def get_info(self):
        """ Get info used for benchmarking. """
        return {}

    def process_env_update(self):
        """ Process state update to RL Agent. """
        if self.current_step == 0:
            self.state = self.get_state()
        else:
            self.next_state, self.reward, self.is_done, __ = self.step()
            if Config().algorithm.mode == 'train':
                self.process_experience()
            self.state = self.next_state
            self.episode_reward += self.reward

    async def prep_agent_update(self):
        """ Update RL Agent. """
        self.current_step += 1
        self.total_steps += 1
        logging.info("[RL Agent] Preparing action...")
        self.prep_action()
        self.action_updated.set()

        # when episode ends
        if Config().algorithm.mode == 'train' and self.is_done:
            self.update_policy()

            # Break the loop when RL training is concluded
            if self.current_episode >= Config().algorithm.max_episode:
                self.finished = True
            else:
                await self.reset()
        elif Config().algorithm.mode == 'test' and self.current_step >= Config(
        ).algorithm.test_step:
            # Break the loop when RL testing is concluded
            self.finished = True

    @abstractmethod
    def update_policy(self):
        """ Update policy if needed in training mode. """

    @abstractmethod
    def process_experience(self):
        """ Process step experience if needed in training mode. """
