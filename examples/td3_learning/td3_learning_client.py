"""
A federated learning client for td3.
"""
import logging
import math
from dataclasses import dataclass
from plato.clients import simple
from plato.config import Config
import td3_learning_trainer
import td3
import os
import numpy as np
import globals

from torch import nn

file_name = "TD3_RL"
models_dir = "./pytorch_models"
results_dir = "./results"

class Report(simple.Report):
    """A client report to be sent to the federated learning server."""
    client_id: int


class RLClient(simple.Client):
    
    def __init__(self, trainer = None, model = None, algorithm = None):
        super().__init__(model=model, algorithm=algorithm)
        self.RL_Online_trainer = trainer
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        if Config().algorithm.save_models and not os.path.exists(models_dir):
            os.makedirs(models_dir)
        self.timesteps_since_eval = 0
        self.episode_num = 0
        self.total_timesteps = 0
        self.done = True

    async def train(self):
        episode_reward = 0
        episode_timesteps = 0 #fixing error about using before assignment
        obs = 0 #fixing error about using before assignment
        round_episode_steps = 0
        if self.total_timesteps > Config().algorithm.max_steps:
            # TODO: when max number of steps is hit, we should stop training and terminate the process. How?
            print("Done training")
            return
        while round_episode_steps < Config().algorithm.max_round_episode_steps:

            #If episode is done
            if self.done:
                #if not at beginning
                if self.total_timesteps != 0:
                    logging.info("Total Timesteps: {} Episode Num: {} Reward: {}".format(self.total_timesteps, self.episode_num, episode_reward))
                    #train here call td3_trainer
                    self.RL_Online_trainer.update()

                #evaluate episode and save policy
                if self.timesteps_since_eval >= Config().algorithm.policy_freq:
                    self.timesteps_since_eval %= Config().algorithm.policy_freq
                    td3.evaluations.append(self.RL_Online_trainer.evaluate_policy(self.RL_Online_trainer))
                    np.save("./results/%s" % (file_name), td3.evaluations)
                
                #When the training step is done, we reset the state of the env
                obs = globals.env.reset()

                #Set done to false
                self.done = False

                # Set rewards and episode timesteps to zero
                episode_reward = 0
                episode_timesteps = 0
                self.episode_num += 1
                
            #Before the number of specified timesteps from config file we sample random actions
            if self.total_timesteps < Config().algorithm.start_steps:
                action = globals.env.action_space.sample()
            else: #after we pass the threshold we switch model
                action = self.RL_Online_trainer.select_action(np.array(obs))

                #if not 0 we add noise
                if Config().algorithm.expl_noise != 0:
                    expl_noise = Config().algorithm.expl_noise
                    action = (action+np.random.normal(0, expl_noise, size = globals.env.action_space.shape[0])).clip(
                        globals.env.action_space.low, globals.env.action_space.high
                    )

            #performs action in environment, then reaches next state and receives the reward
            new_obs, reward, self.done, _ = globals.env.step(action)

            #is episode done?
            done_bool = 0 if episode_timesteps + 1 == globals.env._max_episode_steps else float(self.done)
            
            #update total reward
            episode_reward += reward
            
            #add to replay buffer in this order due to push method in replay buffer
            new = (obs, action, reward, new_obs, done_bool)
            self.RL_Online_trainer.add(new)

            #Update state, episode time_step, total timesteps, and timesteps since last eval
            obs = new_obs
            episode_timesteps += 1
            self.total_timesteps += 1
            round_episode_steps += 1
            self.timesteps_since_eval += 1
        
        #Add the last policy evaluation to our list of evaluations and save evaluations
        td3.evaluations.append(self.RL_Online_trainer.evaluate_policy(self.RL_Online_trainer))
        np.save("./results/%s" % (file_name), td3.evaluations)
        
        #returns report and weights
        report, weights = await super().train()
        return Report(report.num_samples, report.accuracy,
                      report.training_time, report.comm_time,
                      report.update_response), weights




