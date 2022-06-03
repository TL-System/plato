"""
A federated learning client for td3.
"""
import logging
import math
from dataclasses import dataclass
from plato.clients import simple
from plato.config import Config
import td3_trainer
import td3
import os
import numpy as np


file_name = "TD3_RL"
models_dir = "./pytorch_models"
results_dir = "./results"

class RLClient(simple.Client):
    def __init__(self, trainer=None, model=None):
        super().__init__(model=model,trainer=trainer)
        self.trainer = trainer
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        if Config().algorithm.save_models and not os.path.exists(models_dir):
            os.makedirs(models_dir)


    async def train(self):
        total_timesteps = 0
        done = True
        episode_num = 0
        episode_reward = 0
        timesteps_since_eval = 0
        while total_timesteps < Config().algorithm.max_timesteps:

            #If episode is done
            if done:
                #if not at beginning
                if total_timesteps != 0:
                    print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
                    #train here call td3_trainer
                    td3_trainer.Trainer.update()

                #evaluate episode and save policy
                if timesteps_since_eval >= Config().algorithm.policy_freq:
                    timesteps_since_eval %= Config().algorithm.policy_freq
                    td3.evaluations.append(td3_trainer.Trainer.evaluate_policy(self.trainer))
                    np.save("./results/%s" % (file_name), td3.evaluations)
                
                #When the training step is done, we reset the state of the env
                obs = td3.env.reset()

                #Set done to false
                done = False

                # Set rewards and episode timesteps to zero
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
                
            #Before the number of specified timesteps from config file we sample random actions
            if total_timesteps < Config().algorithm.start_timesteps:
                action = td3.env.action_space.sample()
            else: #after we pass the threshold we switch model
                action = self.trainer.select_action(np.array(obs))

                #if not 0 we add noise
                if Config().algorithm.expl_noise != 0:
                    expl_noise = Config().algorithm.expl_noise
                    action = (action+np.random.normal(0, expl_noise, size = td3.env.action_space.shape[0])).clip(
                        td3.env.action_space.low, td3.env.action_space.high
                    )

            #performs action in environment, then reaches next state and receives the reward
            new_obs, reward, done, _ = td3.env.step(action)

            #is episode done?
            done_bool = 0 if episode_timesteps + 1 == td3.env.max_episode_steps else float(done)
            
            #update total reward
            episode_reward += reward
            
            #add to replay buffer in this order due to push method in replay buffer
            td3_trainer.Trainer.add((obs, action, reward, new_obs, done_bool))

            #Update state, episode time_step, total timesteps, and timesteps since last eval
            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1
        
        #Add the last policy evaluation to our list of evaluations and save evaluations
        td3.evaluations.append(td3_trainer.Trainer.evaluate_policy(self.trainer))
        np.save("./results/%s" % (file_name), td3.evaluations)
            




