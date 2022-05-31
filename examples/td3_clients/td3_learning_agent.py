"""
An RL agent for FL training.
"""
import logging
import math
import os
from collections import deque
from statistics import mean, stdev
from turtle import update

import numpy as np
from plato.config import Config
from plato.utils import csv_processor
from plato.utils.reinforcement_learning import rl_agent
from plato.utils.reinforcement_learning.policies import \
    registry as policies_registry
from fei import fei_agent
from plato.utils.reinforcement_learning.policies import td3
from plato.utils.reinforcement_learning.policies.base import Policy

class RLAgent(rl_agent.RLAgent):

    def __init__(self):
        super().__init__
        if hasattr(Config().server,
                   'synchronous') and not Config().server.synchronous:
            self.policy = policies_registry.get(Config().algorithm.n_features,
                                                self.n_actions)
        else:
            self.policy = policies_registry.get(self.n_states, self.n_actions)


    async def reset(self):
        return await super().reset()


    def prep_action(self):
        return super().prep_action()

    def get_state(self):
        return super().get_state()

    def get_reward(self):
        return super().get_reward()


    def get_done(self):
        return super().get_done()



    def process_env_update(self):
        return super().process_env_update()


    def update_policy(self):
        if len(self.policy.replay_buffer) > Config().algorithm.batch_size:
            # TD3
            critic_loss, actor_loss = self.policy.update()

            new_row = []
            for item in self.recorded_rl_items:
                item_value = {
                    'episode': self.current_episode,
                    'actor_loss': actor_loss,
                    'critic_loss': critic_loss
                }[item]
                new_row.append(item_value)

            episode_result_csv_file = f"{Config().params['result_path']}/{os.getpid()}_episode_result.csv"
            csv_processor.write_csv(episode_result_csv_file, new_row)

        episode_reward_csv_file = f"{Config().params['result_path']}/{os.getpid()}_episode_reward.csv"
        csv_processor.write_csv(episode_reward_csv_file, [
            self.current_episode, self.current_step,
            mean(self.pre_acc), self.episode_reward
        ])
        if self.current_episode % Config().algorithm.log_interval == 0:
            self.policy.save_model(self.current_episode)

    def process_experience(self):
        return super().process_experience()
        
        
