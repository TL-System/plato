"""
A basic RL environment for FL server using Gym for RL control.
"""
import asyncio
import logging
import os
import random
import time
from abc import abstractmethod
from dataclasses import dataclass

import gym
import numpy as np
from gym import spaces

import base_rl_agent
from config import RLConfig
from plato.utils import csv_processor


class RandomPolicy(object):
    """ The world's simplest agent. """
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, state):
        return self.action_space.sample()


class RLAgent(base_rl_agent.RLAgent, gym.Env):
    """ A basic RL environment for FL server using Gym for RL control. """
    def __init__(self, config):
        super().__init__()
        self.agent = 'simple'
        self.config = config

        if self.config.discrete_action_space:
            self.action_space = spaces.Discrete(self.config.n_actions)
        else:
            self.action_space = spaces.Box(low=self.config.min_action,
                                           high=self.config.max_action,
                                           shape=(self.config.n_actions, ),
                                           dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(self.config.n_states, ),
                                            dtype=np.float32)

        self.policy = RandomPolicy(self.action_space)

        self.state = None
        self.next_state = None
        self.action = None
        self.next_action = None
        self.reward = None
        self.episode_reward = 0
        self.current_step = 0
        self.total_steps = 0
        self.current_episode = 0
        self.is_done = False

    def step(self, action):
        """ Update the followings using server update. """
        self.next_state = self.get_state()
        self.is_done = self.get_done()
        self.reward = self.get_reward()
        info = self.get_info()

        return self.next_state, self.reward, self.is_done, info

    async def reset(self):
        """ Reset RL environment. """
        # Start a new training session
        logging.info("[RL Agent] Reseting RL environment.")

        # Reset the episode-related variables
        self.current_step = 0
        self.is_done = False
        self.episode_reward = 0

        self.current_episode += 1
        logging.info("[RL Agent] Starting RL episode #%d.",
                     self.current_episode)

        # Reboot/reconfigure the FL server
        await self.sio.emit('env_reset',
                            {'current_episode': self.current_episode})

        return

    async def prep_action(self):
        """ Get action from RL policy. """
        logging.info("[RL Agent] Selecting action...")
        self.action = self.policy.select_action(self.state)
        return self.action

    def get_state(self):
        """ Get state for agent. """
        if self.server_update:
            return self.server_update
        # Initial state is random when env resets
        return [round(random.random(), 4) for i in range(self.config.n_states)]

    def get_reward(self):
        """ Get reward for agent. """
        return 0.0

    def get_done(self):
        """ Get done condition for agent. """
        if self.config.mode == 'train' and self.current_step >= self.config.steps_per_episode:
            logging.info("[RL Agent] Episode #%d ended.", self.current_episode)
            return True
        return False

    def get_info(self):
        """ Get info used for benchmarking. """
        return {}

    def render(self, mode="human"):
        """ Render the Gym env. """
        pass

    def close(self):
        """ Closing the RL Agent. """
        logging.info("[RL Agent] RL control concluded.")

    async def wrap_up(self):
        """ Wrap up when RL control is concluded. """
        # Close FL environment
        await self.sio.emit('agent_dead', {'agent': self.agent})

    # Implement methods for communication between RL agent and env
    def process_env_response(self, response):
        """ Additional RL-specific processing upon the server response. """
        if 'current_round' in response:
            assert self.current_step == response['current_round']
        if 'current_rl_episode' in response:
            assert self.current_episode == response['current_rl_episode']

    def process_env_update(self):
        """ Process state update to RL Agent. """
        if self.current_step == 0:
            self.state = self.get_state()
        else:
            self.step(self.action)
            if self.config.mode == 'train':
                self.process_experience()
            self.state = self.next_state
            self.episode_reward += self.reward
            step_result_csv_file = self.config.result_dir + 'step_result.csv'
            csv_processor.write_csv(step_result_csv_file,
                                    [self.current_episode, self.current_step] +
                                    list(self.state) + list(self.action))

    async def prep_agent_update(self):
        """ Update RL Agent. """
        if self.is_done and self.config.mode == 'train':
            self.update_policy()

            # Break the loop when RL training is concluded
            if self.current_episode >= self.config.max_episode:
                await self.wrap_up()
            else:
                await self.reset()
        elif self.current_step >= self.config.test_step:
            # Break the loop when RL testing is concluded
            await self.wrap_up()
        else:
            self.current_step += 1
            self.total_steps += 1
            logging.info("[RL Agent] Preparing action...")
            agent_response = {'current_step': self.current_step}
            agent_response['current_episode'] = self.current_episode
            agent_response = await self.customize_agent_response(agent_response
                                                                 )

            # Sending the response as metadata to the server (update to follow)
            await self.sio.emit('update_to_arrive',
                                {'agent_response': agent_response})

            # Sending the agent update to server
            action = await self.prep_action()

            logging.info(
                "[RL Agent] Sending the current action at episode %d timestep %d to server.",
                self.current_episode, self.current_step)

            await self.send_update(action)

    async def customize_agent_response(self, response):
        """ Wrap up generating the agent response with any additional information. """
        return response

    @abstractmethod
    def update_policy(self):
        """ Update policy if needed in training mode. """

    @abstractmethod
    def process_experience(self):
        """ Process step experience if needed in training mode. """
