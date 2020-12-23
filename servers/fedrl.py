"""
A federated server with a reinforcement learning agent.
This federated server uses reinforcement learning
to tune a parameter.
"""

import logging
import asyncio
from stable_baselines3.common.env_checker import check_env

from config import Config
from servers import FedAvgServer
import servers
from rl_envs import FLEnv

FLServer = FedAvgServer
if Config().rl:
    FLServer = {"fedavg": servers.fedavg.FedAvgServer}[Config().rl.fl_server]


class FedRLServer(FLServer):
    """Federated server using RL."""
    def __init__(self):
        super().__init__()

        self.rl_env = FLEnv(self)

        self.rl_episode = 0
        self.rl_tuned_para_value = None
        self.rl_state = None
        self.is_rl_tuned_para_got = False
        self.is_rl_episode_done = False

    def configure(self):
        """
        Booting the RL agent and the FL server
        """
        logging.info('Configuring a RL agent and a %s server...',
                     Config().rl.fl_server)

        rl_tuned_para_name = {
            'edge_agg_num': 'number of aggregations on edge servers',
        }[Config().rl.tuned_para]
        logging.info("This RL agent will tune the %s of FL.",
                     rl_tuned_para_name)

        total_episodes = Config().rl.episodes
        target_reward = Config().rl.target_reward

        if target_reward:
            logging.info('RL Training: %s episodes or %s%% reward\n',
                         total_episodes, 100 * target_reward)
        else:
            logging.info('RL Training: %s episodes\n', total_episodes)

    def start_clients(self, as_server=False):
        """Start all clients and RL training."""
        super().start_clients(as_server)

        # The starting point of RL training
        #self.start_rl()

        # Run RL training as a coroutine
        if not as_server:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(asyncio.gather(self.start_rl()))

    def start_rl(self):
        """The starting point of RL training."""
        # Test the environment of reinforcement learning.
        #self.check_with_sb3_env_checker(self.rl_env)
        self.try_a_random_agent(self.rl_env)

    def reset_rl_env(self):
        """Reset the RL environment at the beginning of each episode."""
        # The number of finished FL training round
        self.current_round = 0

        self.is_rl_tuned_para_got = False
        self.is_rl_episode_done = False

        self.rl_episode += 1
        logging.info('\nRL Agent: starting episode %s...', self.rl_episode)

        # Configure the FL central server
        super().configure()

        # starting time of a gloabl training round
        self.round_start_time = 0
        # training time spent in each round
        self.training_time_list = []
        # global model accuracy of each round
        self.accuracy_list = []

    async def wrap_up_one_round(self):
        """Wrapping up when one round of FL training is done."""
        await super().wrap_up_one_round()

        # Get the RL state
        # Use accuracy as state for now
        self.rl_state = self.accuracy

        target_accuracy = Config().training.target_accuracy
        if (target_accuracy and self.accuracy >= target_accuracy
            ) or self.current_round >= Config().training.rounds:
            self.is_rl_episode_done = True

        # Pass the RL state to the RL env
        self.rl_env.get_state(self.rl_state, self.is_rl_episode_done)

        # Give RL env some time to finish step() before FL starts next round
        while not self.rl_env.is_step_done:
            await asyncio.sleep(1)

    async def generate_rl_info(self, server_response):
        """Get RL tuned parameter that will be sent to clients."""
        while not self.is_rl_tuned_para_got:
            await asyncio.sleep(1)
        self.rl_env.is_state_got = False

        server_response['rl_tuned_para_name'] = Config().rl.tuned_para
        server_response['rl_tuned_para_value'] = self.rl_tuned_para_value
        return server_response

    def get_tuned_para(self, rl_tuned_para_value, time_step):
        """
        Get tuned parameter from RL env.
        This function is called by RL env.
        """
        self.rl_tuned_para_value = rl_tuned_para_value
        self.is_rl_tuned_para_got = True
        print("RL agent: Get tuned para of time step", time_step)

    def wrap_up(self):
        """Wrapping up when the FL training is done."""
        if self.rl_episode >= Config().rl.episodes:
            logging.info(
                'RL Agent: Target number of training episodes reached.')

    @staticmethod
    def check_with_sb3_env_checker(env):
        """
        Use helper provided by stable_baselines3
        to check that the environment runs without error.
        """
        # It will check the environment and output additional warnings if needed
        check_env(env)

    @staticmethod
    def try_a_random_agent(env):
        """Quickly try a random agent on the environment."""
        obs = env.reset()
        episodes = Config().rl.episodes
        n_steps = Config().training.rounds

        for i in range(episodes):
            for _ in range(n_steps):
                # Random action

                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                if done:
                    obs = env.reset()
                    break
