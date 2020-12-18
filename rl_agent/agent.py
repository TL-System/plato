"""
Starting point for a reinforcement learning agent.
This agent will tune a parameter for federated learning.
"""

# pylint: disable=E1101

import logging
import asyncio
import sys
import time

from config import Config
import utils.plot_figures as plot_figures


class RLAgent:
    """A basic reinforcement learning agent."""
    def __init__(self):
        self.episode_num = 0
        self.time_step = 1
        self.finished_time_step = 0
        self.central_server = None
        self.env = None
        self.rl_tuned_para_value = None
        self.rl_state = None
        self.is_tuned_para_got = False
        self.is_state_got = False
        self.is_rl_episode_done = False

    def configure(self, env, fl_central_server):
        """Booting the reinforcement learning agent."""
        logging.info('Configuring the RL agent...')

        self.env = env
        self.central_server = fl_central_server

        rl_tuned_para_name = {
            'edge_agg_num': 'number of aggregations on edge servers',
        }[Config().rl.tuned_para]
        logging.info("This RL agent will tune the %s.", rl_tuned_para_name)

        total_episodes = Config().rl.episodes
        target_reward = Config().rl.target_reward

        if target_reward:
            logging.info('RL Training: %s episodes or %s%% reward\n',
                         total_episodes, 100 * target_reward)
        else:
            logging.info('RL Training: %s episodes\n', total_episodes)

    def start_federated_learning(self):
        """
        Starting federated learning by configuring the central server
        (loading the initial global model.)
        """
        self.central_server.reconfigure()

    async def wait_until_central_server_start(self):
        """Wait for a while to let the central server start first"""
        #await asyncio.sleep(5)
        time.sleep(5)

    async def serve(self):
        """Running the reinforcement learning agent."""

        while True:

            # Wait until env passes the tuned para
            while not self.is_tuned_para_got:
                await asyncio.sleep(1)
            self.is_tuned_para_got = False

            # Pass the tuned parameter to the central server
            self.central_server.get_tuned_para(self.time_step,
                                               self.rl_tuned_para_value)

            # Wait until central server passes the new RL state
            while not self.is_state_got:
                await asyncio.sleep(1)
            self.is_state_got = False

            assert self.finished_time_step == self.time_step
            logging.info(
                "RL Agent: Update from central server of time step %s received.",
                self.time_step)
            self.time_step += 1

            # Let env get the current state
            self.env.get_state_from_rl_agent(self.rl_state,
                                             self.is_rl_episode_done)

            if self.is_rl_episode_done:

                # Break the 'while True' loop when the target number of episodes is achieved
                if self.episode_num >= Config().rl.episodes:
                    logging.info(
                        'RL Agent: Target number of training episodes reached.'
                    )
                    self.plot_rl_figures()
                    sys.exit()

    def get_tuned_para(self, rl_tuned_para_value):
        """
        Get tuned parameter from env.
        This function is called by env.
        """
        self.rl_tuned_para_value = rl_tuned_para_value
        self.is_tuned_para_got = True

    def get_state(self, rl_state, is_rl_episode_done, finished_time_step):
        """
        Get RL state from central server.
        This function is called by central server.
        """
        self.rl_state = rl_state
        self.is_rl_episode_done = is_rl_episode_done
        self.finished_time_step = finished_time_step
        self.is_state_got = True
        logging.info(
            "RL Agent: Got the state of time step %s from the central server.",
            self.finished_time_step)

    def reset_env(self):
        """Reset the RL environment when an episode is finished."""
        self.episode_num += 1
        self.time_step = 1
        logging.info('\nRL Agent: starting episode %s...', self.episode_num)
        self.start_federated_learning()

    def plot_rl_figures(self):
        """Plot figures showing the results of reinforcement learning."""
        # To be implemented
        pass
