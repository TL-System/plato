"""
An RL agent for FL training.
"""
import logging
import math
import os
from collections import deque
from statistics import mean, stdev

import numpy as np
from plato.config import Config
from plato.utils import csv_processor
from plato.utils.reinforcement_learning import simple_rl_agent
from plato.utils.reinforcement_learning.policies import \
    registry as policies_registry


class RLAgent(simple_rl_agent.RLAgent):
    """ An RL agent for FL training using FEI. """
    def __init__(self):
        super().__init__()
        self.policy = policies_registry.get(Config().algorithm.n_features,
                                            self.action_space)

        if Config().algorithm.recurrent_actor:
            self.h, self.c = self.policy.get_initial_states()
            self.nh, self.nc = self.h, self.c

        if Config().algorithm.mode == 'train' and Config(
        ).algorithm.pretrained or Config().algorithm.mode == 'test':
            self.policy.load_model(Config().algorithm.pretrained_iter)
            self.current_episode = Config().algorithm.pretrained_iter + 1

        self.recorded_rl_items = ['episode', 'actor_loss', 'critic_loss']

        if self.current_episode == 0:
            results_dir = Config().results_dir
            episode_result_csv_file = f'{results_dir}/{os.getpid()}_episode_result.csv'
            csv_processor.initialize_csv(episode_result_csv_file,
                                         self.recorded_rl_items, results_dir)
            episode_reward_csv_file = f'{results_dir}/{os.getpid()}_episode_reward.csv'
            csv_processor.initialize_csv(
                episode_reward_csv_file,
                ['Episode #', 'Steps', 'Final accuracy', 'Reward'],
                results_dir)
            step_result_csv_file = f'{results_dir}/{os.getpid()}_step_result.csv'
            csv_processor.initialize_csv(
                step_result_csv_file,
                ['Episode #', 'Step #', 'state', 'action'], results_dir)

        # Record test accuracy of the latest 5 rounds/steps
        self.pre_acc = deque(5 * [0], maxlen=5)
        self.test_accuracy = None

    # Override RL-related methods of simple RL agent
    async def reset(self):
        """ Reset RL environment. """
        await super().reset()

        if Config().algorithm.recurrent_actor:
            self.h, self.c = self.policy.get_initial_states()

    def prep_action(self):
        """ Get action from RL policy. """
        logging.info("[RL Agent] Selecting action...")
        if Config().algorithm.mode == 'train':
            if Config().algorithm.start_steps > self.total_steps:
                self.action = np.array(
                    self.action_space.sample())  # Sample random action
                if Config().algorithm.recurrent_actor:
                    self.action = np.reshape(self.action, (-1, 1))
            else:  # Sample action from policy
                if Config().algorithm.recurrent_actor:
                    self.action, (self.nh,
                                  self.nc) = self.policy.select_action(
                                      self.state, (self.h, self.c))
                    self.action = np.reshape(np.array(self.action), (-1, 1))
                else:
                    self.action = self.policy.select_action(self.state)
        else:
            if Config().algorithm.recurrent_actor:
                # don't pass hidden states
                self.action, __ = self.policy.select_action(self.state)
                self.action = np.reshape(np.array(self.action), (-1, 1))
            else:
                self.action = self.policy.select_action(self.state)

    def get_reward(self):
        """ Get reward for agent. """
        # punish more time steps
        reward = -1
        # reward for average accuracy in the last a few time steps
        if self.is_done:
            avg_accuracy = mean(self.pre_acc)
            reward += math.log(avg_accuracy /
                               (1 - avg_accuracy)) * Config().algorithm.beta
        return reward

    def get_done(self):
        """ Get done condition for agent. """
        if Config().algorithm.mode == 'train':
            self.pre_acc.append(self.test_accuracy)
            if stdev(self.pre_acc) < Config().algorithm.theta:
                logging.info("[RL Agent] Episode #%d ended.",
                             self.current_episode)
                return True
        return False

    def process_env_update(self):
        """ Process state update to RL Agent. """
        super().process_env_update()
        if Config().algorithm.recurrent_actor:
            self.h, self.c = self.nh, self.nc

    def update_policy(self):
        """ Update agent if needed in training mode. """
        logging.info("[RL Agent] Updating the policy.")
        if len(self.policy.replay_buffer) > Config().algorithm.batch_size:
            # TD3-LSTM
            critic_loss, actor_loss = self.policy.update()

            new_row = []
            for item in self.recorded_rl_items:
                item_value = {
                    'episode': self.current_episode,
                    'actor_loss': actor_loss,
                    'critic_loss': critic_loss
                }[item]
                new_row.append(item_value)
            results_dir = Config().results_dir
            episode_result_csv_file = f'{results_dir}/{os.getpid()}_episode_result.csv'
            csv_processor.write_csv(episode_result_csv_file, new_row)

        episode_reward_csv_file = Config().results_dir + 'episode_reward.csv'
        csv_processor.write_csv(episode_reward_csv_file, [
            self.current_episode, self.current_step,
            mean(self.pre_acc), self.episode_reward
        ])

        # Reinitialize the previous accuracy queue
        for _ in range(5):
            self.pre_acc.append(0)

        if self.current_episode % Config().algorithm.log_interval == 0:
            self.policy.save_model(self.current_episode)

    def process_experience(self):
        """ Process step experience if needed in training mode. """
        logging.info("[RL Agent] Saving the experience into replay buffer.")
        if Config().algorithm.recurrent_actor:
            self.policy.replay_buffer.push(
                (self.state, self.action, self.reward, self.next_state,
                 np.float(self.is_done), self.h, self.c, self.nh, self.nc))
        else:
            self.policy.replay_buffer.push(
                (self.state, self.action, self.reward, self.next_state,
                 np.float(self.is_done)))
