from difflib import restore
import numpy as np
import torch
import gym
import park
from copy import deepcopy
from plato.trainers import basic
from plato.config import Config
from plato.trainers import basic
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.autograd import Variable
import random

import os
import logging

import csv
import pickle

import park
import shutil
# Memory
# Stores results from the networks, instead of calculating the operations again from states, etc.

class StateNormalizer(object):
    def __init__(self, obs_space):
        self.shift = obs_space.low
        self.range = obs_space.high - obs_space.low

    def normalize(self, obs):
        return (obs - self.shift) / self.range

class Memory():
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()  
    
    def _zip(self):
        return zip(self.log_probs,
                self.values,
                self.rewards,
                self.dones)
    
    def __iter__(self):
        for data in self._zip():
            return data
    
    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data
    
    def __len__(self):
        return len(self.rewards)

class Trainer(basic.Trainer):
    def __init__(self, model=None):
        super().__init__(model=model)

        self.env = park.make(Config().algorithm.env_park_name)

        self.env_name = Config().algorithm.env_name
        self.algorithm_name =  Config().algorithm.algorithm_name
        
        self.actor = self.model.actor
        self.critic = self.model.critic
        self.actor_old = deepcopy(self.actor.state_dict())
        self.critic_old = deepcopy(self.critic.state_dict())
        self.adam_actor = torch.optim.Adam(self.actor.parameters(), lr=Config().algorithm.learning_rate)
        self.adam_critic = torch.optim.Adam(self.critic.parameters(), lr=Config().algorithm.learning_rate)

        self.l2_loss = torch.nn.MSELoss(reduction='mean')

        self.dist = torch.distributions.Categorical        

        self.memory = Memory()
        self.obs_normalizer = StateNormalizer(self.env.observation_space)

        self.server_reward = []
        self.avg_reward = []
        self.first_ittr_server = True

        self.episode_num = 0
        self.trace_idx = 0
        self.total_reward = 0
        self.done = True
        self.steps = 0

        self.actor_state_dict = None
        self.critic_state_dict = None

        self.critic_loss = []
        self.actor_loss = []
        self.entropy_loss = []

        self.avg_critic_loss = 0
        self.avg_actor_loss = 0
        self.avg_entropy_loss = 0

        # Fisher estimation parameters
        self.fisher_critic, self.fisher_actor = {}, {}
        self.fisher_critic_old, self.fisher_actor_old = {}, {}
        self.critic_fisher_sum, self.actor_fisher_sum = 0, 0

        # Omega holding important parameters for other parameters
        self.omega_actor, self.omega_critic = {}, {}
        
        self.updates = 0
        self.timesteps_since_eval = 0
        self.round_no = 0

        if not os.path.exists(f'{Config().results.results_dir}_seed_{Config().server.random_seed}'):
            os.makedirs(f'{Config().results.results_dir}_seed_{Config().server.random_seed}', exist_ok=True)

        if not os.path.exists(f'{Config().results.seed_random_path}_seed_{Config().server.random_seed}'):
            os.makedirs(f'{Config().results.seed_random_path}_seed_{Config().server.random_seed}', exist_ok=True)
        else:
            shutil.rmtree(f'{Config().results.seed_random_path}_seed_{Config().server.random_seed}', ignore_errors=True)
            os.makedirs(f'{Config().results.seed_random_path}_seed_{Config().server.random_seed}', exist_ok=True)


    def t(self, x): 
        return torch.from_numpy(x).float()


    def train_model(self, config, trainset, sampler, cut_layer):
        """Main Training"""
        # Fisher information matrix load from previous rounds of training
        # Fisher Information matrix path
        model_path = Config().params['model_path']
        env_algorithm = self.env_name + self.algorithm_name
        client_id = str(self.client_id)
        model_seed_path = f'_seed_{Config().server.random_seed}'
        actor, critic = "actor_fisher_matrix", "critic_fisher_matrix"
        actor_fisher_path = f'{model_path}/{env_algorithm}{actor}{model_seed_path}{client_id}.pth'
        critic_fisher_path = f'{model_path}/{env_algorithm}{critic}{model_seed_path}{client_id}.pth'
        
        if self.episode_num >= Config().algorithm.max_round_episodes:
            self.load_fisher()
        else:
            self.fisher_critic, self.fisher_actor = {}, {}

        # Load Omega 
        if self.episode_num >= Config().algorithm.max_round_episodes:
            self.load_omega()

        # Seeds path
        seed_file_name = f'{"id_"}{str(self.client_id)}'
        self.seed_path = f'{Config().results.seed_random_path}_seed_{Config().server.random_seed}/{seed_file_name}'
        
        if not os.path.exists(self.seed_path):
            torch.manual_seed(Config().trainer.manual_seed)
        else:
            self.restore_seeds()

        common_path = f'{Config().results.results_dir}_seed_{Config().server.random_seed}/{Config().results.file_name}_{str(self.client_id)}'
        
        round_episodes = 0

        while round_episodes < Config().algorithm.max_round_episodes:
            first_itr = self.episode_num < Config().algorithm.max_round_episodes
            #Evaluates policy at a frequency set in config file
            if self.timesteps_since_eval >= Config().algorithm.eval_freq:
                self.save_seeds()
                self.avg_reward = self.evaluate_policy()
                self.restore_seeds()
                # Save avg reward
                avg_reward_path = f'{common_path}{"_avg_reward"}'
                #common_path+"_avg_reward"
                self.timesteps_since_eval = 0
                first_itr = self.episode_num <= Config().algorithm.eval_freq
                self.save_metric(avg_reward_path, self.avg_reward, first = first_itr)

            self.done = False
            self.total_reward = 0
            
            #Make difficulty level (trace file) depend on client_id
            self.trace_idx = ((self.client_id - 1) % Config().algorithm.difficulty_levels)
            state = self.env.reset(trace_idx=self.trace_idx, test= True)
            state = self.obs_normalizer.normalize(state)
            self.steps = 0
            
            while not self.done:
                probs = self.actor(self.t(state))
                action = self.dist(probs=probs).sample()
                
                next_state, reward, self.done, info = self.env.step(action.detach().data.numpy())
                next_state = self.obs_normalizer.normalize(next_state)

                self.total_reward += reward
                self.steps += 1
                
                self.memory.add(self.dist(probs=probs).log_prob(action), self.critic(self.t(state)), reward, self.done)

                state = next_state
                
                if self.done or (self.steps % Config().algorithm.batch_size == 0):
                    last_q_val = self.critic(self.t(next_state)).detach().data.numpy()
                    if self.updates > 1 or self.episode_num > 1:
                        fisher_critic_old = {}
                        fisher_actor_old = {}
                        for (n, _) in self.critic.named_parameters():
                            fisher_critic_old[n] = self.fisher_critic[n].clone()
                        for (n, _) in self.actor.named_parameters():
                            fisher_actor_old[n] = self.fisher_actor[n].clone()
                    # Estimate diagonals of fisher information matrix
                    self.estimate_fisher(self.train_helper(self.memory, last_q_val, fisher = True))
                    self.sum_fisher_diagonals()
                    # Accumulate fishers
                    if self.updates > 1 or self.episode_num > 1:
                        for n,_ in self.critic.named_parameters():
                            self.fisher_critic[n] = (self.fisher_critic[n] + fisher_critic_old[n] * self.updates)/(self.updates + 1)
                        for n,_ in self.actor.named_parameters():
                            self.fisher_actor[n] = (self.fisher_actor[n] + fisher_actor_old[n] * self.updates)/(self.updates + 1)
                        

                    # Save sum of diagonals in fisher
                    actor_fisher_path = f'{common_path}{"_actor_fisher"}'
                    #common_path + "_actor_fisher"
                    critic_fisher_path = f'{common_path}{"_critic_fisher"}'
                    #common_path + "_critic_fisher"
                    self.save_metric(actor_fisher_path, [self.actor_fisher_sum.tolist()], first = (self.episode_num == 0) and (self.steps == Config().algorithm.batch_size))
                    self.save_metric(critic_fisher_path, [self.critic_fisher_sum.tolist()], first = (self.episode_num == 0) and (self.steps == Config().algorithm.batch_size))

                    critic_loss, actor_loss, entropy_loss = self.train_helper(self.memory, last_q_val)
                    self.updates += 1
                    
                    self.critic_loss.append(critic_loss)
                    self.actor_loss.append(actor_loss)
                    self.entropy_loss.append(entropy_loss)
                    self.memory.clear()

            self.episode_num += 1
            self.timesteps_since_eval += 1
            round_episodes += 1
            print("Episode number: %d, Reward: %d" % (self.episode_num, self.total_reward))
        
        # End of round: 
        # 1- Evaluate policy on traces
        first_itr = self.episode_num <= Config().algorithm.eval_freq
        self.save_seeds()
        self.avg_reward = self.evaluate_policy()
        self.restore_seeds()
        avg_reward_path = f'{common_path}{"_avg_reward"}'
        #common_path + "_avg_reward"
        self.save_metric(avg_reward_path, self.avg_reward, first=first_itr)

        # 2- Get gradients of change in actor and critic loss
        x = np.array(range(len(self.actor_loss)))+1
        actor_grad, _ = np.polyfit(x, np.array(self.actor_loss), 1)
        critic_grad, _ = np.polyfit(x, np.array(self.critic_loss), 1)
        # and save in file
        critic_grad_path = f'{common_path}{"_critic_grad"}'
        #common_path + "_critic_grad"
        actor_grad_path =  f'{common_path}{"_actor_grad"}'
        #common_path + "_actor_grad"
        self.save_metric(critic_grad_path, [critic_grad], first = first_itr)
        self.save_metric(actor_grad_path, [actor_grad], first = first_itr)
        
        self.avg_actor_loss = sum(self.actor_loss)/len(self.actor_loss)
        self.avg_critic_loss =  sum(self.critic_loss)/len(self.critic_loss)
        self.avg_entropy_loss = sum(self.entropy_loss)/len(self.entropy_loss)

        self.save_seeds()

        # Save Fisher Information Matrix
        with open(actor_fisher_path, "wb") as file:
            torch.save(self.fisher_actor, file)
        with open(critic_fisher_path, "wb") as file:
            torch.save(self.fisher_critic, file)


    def save_seeds(self):
        """ Saving the random seeds in the trainer for resuming its session later on. """
        with open(f'{self.seed_path}.pkl', 'wb') as checkpoint_file:
            pickle.dump(torch.get_rng_state(), checkpoint_file)

    def restore_seeds(self):
        """Restoring the random seeds in the trainer for resuming its session later on"""
        #seed_path should have client id in it!!!!!
        rng_state_to_load = None

        with open(f'{self.seed_path}.pkl', 'rb') as checkpoint_file:
            rng_state_to_load = pickle.load(checkpoint_file)
        
        torch.set_rng_state(rng_state_to_load)
               
    def train_helper(self, memory, q_val, fisher = False):
        #We will put the train loop here
        values = torch.stack(memory.values)
        q_vals = np.zeros((len(memory), 1))

        #target values calculated backward
        #important to handle correctly done states
        #for those cases we want our target to be equal to the reward only
        for i, (_, _, reward, done) in enumerate(memory.reversed()):
            q_val = reward + Config().algorithm.gamma*q_val*(1.0-done)
            q_vals[len(memory)-1 - i] = q_val #store values from end to the start
        
        # Advantage function
        advantage = torch.Tensor(q_vals) - values.detach()
        
        critic_loss = self.l2_loss(values, torch.Tensor(q_vals)) 
        
        entropy_loss = (torch.stack(memory.log_probs) * torch.exp(torch.stack(memory.log_probs))).mean()
        entropy_coef = max((Config().algorithm.entropy_ratio - (self.episode_num/Config().algorithm.batch_size) * Config().algorithm.entropy_decay), Config().algorithm.entropy_min)
        actor_loss = (-torch.stack(memory.log_probs)*advantage.detach()).mean() + entropy_loss * entropy_coef

        if Config().trainer.penalize_omega and self.episode_num >= 3 * Config().algorithm.max_round_episodes:
            actor_reg_term, critic_reg_term = self.criterion_reg()
            actor_loss += actor_reg_term
            critic_loss += critic_reg_term

        if Config().algorithm.grad_clip_val > 0:
            clip_grad_norm_(self.actor.parameters(), Config().algorithm.grad_clip_val)

        if not fisher:
            self.adam_critic.zero_grad()
            critic_loss.backward()
            self.adam_critic.step()

            self.adam_actor.zero_grad()
            actor_loss.backward()
            self.adam_actor.step()

            return critic_loss.item(), actor_loss.item(), entropy_loss.item()
        else:
            return critic_loss, actor_loss

    def estimate_fisher(self, loss):
        """Estimate diagonals of fisher information matrix"""
        critic_loss, actor_loss = loss
        # Get fisher for critic model
        self.adam_critic.zero_grad()
        critic_loss.backward(retain_graph=True)
        
        for n, p in self.critic.named_parameters():
            pg = p.grad.data.clone().pow(2)
            self.fisher_critic[n] = pg

        # Get fisher for actor model
        self.adam_actor.zero_grad()
        actor_loss.backward(retain_graph=True)
        
        for n, p in self.actor.named_parameters():
            pg = p.grad.data.clone().pow(2)
            self.fisher_actor[n] = pg

    def sum_fisher_diagonals(self):
        """Sum the diagonals of the Fisher Information Matrix"""
        actor_fisher_sum = 0
        critic_fisher_sum = 0
        
        for n, _ in self.critic.named_parameters():
            l = Variable(self.fisher_critic[n])
            critic_fisher_sum += l.sum()
    
        for n, _ in self.actor.named_parameters():
            l = Variable(self.fisher_actor[n])
            actor_fisher_sum += l.sum()

        self.critic_fisher_sum, self.actor_fisher_sum = critic_fisher_sum, actor_fisher_sum
    
    def criterion_reg(self):
        loss_reg_actor = 0
        loss_reg_critic = 0 
        for (name,param), (_, param_old) in zip(self.actor.named_parameters(), self.actor_old.named_parameters()):
            loss_reg_actor += torch.sum(self.omega_actor[name]*(param_old-param).pow(2))/2
        for (name,param), (_, param_old) in zip(self.critic.named_parameters(), self.critic_old.named_parameters()):
            loss_reg_critic += torch.sum(self.omega_critic[name]*(param_old-param).pow(2))/2
        return Config().trainer.lamda * loss_reg_actor, Config().trainer.lamda * loss_reg_critic


    def load_fisher(self):
        """ Load last fisher from file"""
        path = f'{Config().results.results_dir}_seed_{Config().server.random_seed}/{Config().results.file_name}_{self.client_id}'
        
        SIZE = 10
        act_pos, cri_pos = 0, 0
        GRAD_SIZE = 2 * Config().algorithm.max_round_episodes
        actor_fisher_round, critic_fisher_round = np.zeros(GRAD_SIZE), np.zeros(GRAD_SIZE)
        with open(f'{path}{"_actor_fisher.csv"}', 'r') as file:
            rows = file.readlines()
            # TODO: moving average of the last 10 or more! 
            # One round: 2 * max_round_episodes episodes each episode, two fishers
            for row in rows:
                actor_fisher_round[act_pos] = row
                act_pos = (act_pos + 1) % GRAD_SIZE
        with open(f'{path}{"_critic_fisher.csv"}', 'r') as file:
            rows = file.readlines()
            for row in rows:
                critic_fisher_round[cri_pos] = row
                cri_pos = (cri_pos + 1) % GRAD_SIZE
        avg_actor_fisher = np.mean(actor_fisher_round[-SIZE:])
        avg_critic_fisher = np.mean(critic_fisher_round[-SIZE:])
        x = np.array(range(len(actor_fisher_round)))+1
        actor_fisher_grad, _ = np.polyfit(x, actor_fisher_round, 1)
        critic_fisher_grad, _ = np.polyfit(x, critic_fisher_round, 1)

        # Save fisher grads, may need later, may delete later
        self.save_metric(f'{path}{"_actor_fisher_grad"}', [actor_fisher_grad], first = self.episode_num <= Config().algorithm.max_round_episodes)
        self.save_metric(f'{path}{"_critic_fisher_grad"}', [critic_fisher_grad], first = self.episode_num <= Config().algorithm.max_round_episodes)
        
        actor, critic = "actor_fisher_matrix", "critic_fisher_matrix"
        model_path = Config().params['model_path']
        client_id = str(self.client_id)
        env_algorithm = self.env_name+ self.algorithm_name
        model_seed_path = f'_seed_{Config().server.random_seed}'
        actor_fisher_path = f'{model_path}/{env_algorithm}{actor}{model_seed_path}{client_id}.pth'
        critic_fisher_path = f'{model_path}/{env_algorithm}{critic}{model_seed_path}{client_id}.pth'
        
        with open(actor_fisher_path, "rb") as file:
            self.fisher_actor = torch.load(file)
        with open(critic_fisher_path, "rb") as file:
            self.fisher_critic = torch.load(file)

        return avg_actor_fisher, avg_critic_fisher, actor_fisher_grad, critic_fisher_grad, self.fisher_actor, self.fisher_critic

    def load_omega(self):
        omega_path = f"{Config().general.base_path}/{Config().server.model_path}"
        actor_path = f"{omega_path}/{self.env_name}{self.algorithm_name}omega_actor_client_{self.client_id}_seed_{Config().server.random_seed}.pth"
        critic_path = f"{omega_path}/{self.env_name}{self.algorithm_name}omega_critic_client_{self.client_id}_seed_{Config().server.random_seed}.pth"
        
        with open(actor_path, 'rb') as omg_actor_path:
            self.omega_actor = torch.load(omg_actor_path)
        with open(critic_path, 'rb') as omg_critic_path:
            self.omega_critic = torch.load(omg_critic_path)

    def load_model(self, filename=None, location=None):
        """Loading pre-trained model weights from a file."""
        #We will load actor and critic models here
        
        model_path = Config(
        ).params['model_path'] if location is None else location
        actor_model_name = 'actor_model'
        critic_model_name = 'critic_model'
        env_algorithm = self.env_name+ self.algorithm_name
        model_seed_path = f'_seed_{Config().server.random_seed}'

        if filename is not None and self.client_id == 0:
            actor_filename = f'{filename}{"_actor"}{model_seed_path}.pth'
            actor_model_path = f'{model_path}/{actor_filename}'
            critic_filename = f'{filename}{"_critic"}{model_seed_path}.pth'
            critic_model_path = f'{model_path}/{critic_filename}'
        else:
            actor_model_path = f'{model_path}/{env_algorithm}{actor_model_name}{model_seed_path}.pth'
            critic_model_path = f'{model_path}/{env_algorithm}{critic_model_name}{model_seed_path}.pth'

        if self.client_id == 0:
            logging.info("[Server #%d] Loading models from %s, and %s.", os.getpid(),
                         actor_model_path, critic_model_path)
        else:
            logging.info("[Client #%d] Loading a model from %s, and %s.",
                         self.client_id, actor_model_path, critic_model_path)

        if self.client_id != 0:
            file_name = "%s_%s.npz" % ("training_status", str(self.client_id)) 
            file_path = os.path.join(model_path, file_name)
            data = np.load(file_path)
            self.episode_num = int((data['a'])[0])

    
            self.actor.load_state_dict(torch.load(actor_model_path), strict=True)
            self.critic.load_state_dict(torch.load(critic_model_path), strict=True)
            self.actor_old = deepcopy(self.actor)
            self.critic_old = deepcopy(self.critic)

            #unsure if we need these
            self.adam_actor = torch.optim.Adam(self.actor.parameters(), lr=Config().algorithm.learning_rate)
            self.adam_critic = torch.optim.Adam(self.critic.parameters(), lr=Config().algorithm.learning_rate)

    def load_loss(self):
        path = f'{Config().results.results_dir}_seed_{Config().server.random_seed}/{Config().results.file_name}_{self.client_id}'
       # Config().results.results_dir +"/"+Config().results.file_name+"_"+str(self.client_id)
        with open(f'{path}{"_actor_loss.csv"}', 'r') as file:
            rows = file.readlines()
            for row in rows:
                actor_loss = float(row)

        with open(f'{path}{"_critic_loss.csv"}', 'r') as file:
            rows = file.readlines()
            for row in rows:
                critic_loss = float(row)

        with open(f'{path}{"_entropy_loss.csv"}', 'r') as file:
            rows = file.readlines()
            for row in rows:
                entropy_loss = float(row)

        return actor_loss, critic_loss, entropy_loss

    def load_grads(self):
        path = f'{Config().results.results_dir}_seed_{Config().server.random_seed}/{Config().results.file_name}_{self.client_id}'
        actor_grad= []
        critic_grad = []
        with open(f'{path}{"_actor_grad.csv"}', 'r') as file:
            rows = file.readlines()
            for row in rows:
                actor_grad = float(row)
        with open(f'{path}{"_critic_grad.csv"}', 'r') as file:
            rows = file.readlines()
            for row in rows:
                critic_grad = float(row)

        return actor_grad, critic_grad

    def save_metric(self, path, value, first = False):
        with open(f'{path}.csv', 'w' if first else 'a') as filehandle:
            writer = csv.writer(filehandle)
            writer.writerow(value)

    def save_model(self, filename=None, location=None):
        """Saving the model, training status and fisher information matrix here to a file."""
        #We will save actor and critic models here
        model_path = Config(
        ).params['model_path'] if location is None else location
        actor_model_name = 'actor_model'
        critic_model_name = 'critic_model'
        env_algorithm = f'{self.env_name}{self.algorithm_name}'
        #self.env_name+ self.algorithm_name
        model_seed_path = f'_seed_{Config().server.random_seed}'

        #call save loss here
        if self.client_id != 0:
            path = f'{Config().results.results_dir}_seed_{Config().server.random_seed}/{Config().results.file_name}_{self.client_id}'
            self.save_loss(path)
    
        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except FileExistsError:
            pass

        if filename is not None and self.client_id == 0:
            actor_filename = f'{filename}{"_actor"}{model_seed_path}.pth'
            actor_model_path = f'{model_path}/{actor_filename}'
            critic_filename = f'{filename}{"_critic"}{model_seed_path}.pth'
            critic_model_path = f'{model_path}/{critic_filename}'
        else:
            actor_model_path = f'{model_path}/{env_algorithm}{actor_model_name}{model_seed_path}.pth'
            critic_model_path = f'{model_path}/{env_algorithm}{critic_model_name}{model_seed_path}.pth'

        if self.model_state_dict is None:
            torch.save(self.actor.state_dict(), actor_model_path)
            torch.save(self.critic.state_dict(),critic_model_path)
        else:
            torch.save(self.actor_state_dict, actor_model_path)
            torch.save(self.critic_state_dict, critic_model_path)


        if self.client_id != 0:
            # Training status
            file_name = "%s_%s.npz" % ("training_status", str(self.client_id)) 
            file_path = os.path.join(model_path, file_name)
            np.savez(file_path, a=np.array([self.episode_num])) 
            

        if self.client_id == 0:
            logging.info("[Server #%d] Saving models to %s, and %s.", os.getpid(),
                         actor_model_path, critic_model_path)
        else:
            logging.info("[Client #%d] Saving a model to %s, and %s.",
                         self.client_id, actor_model_path, critic_model_path)

    def save_loss(self, path):

        actor_loss_path = f'{path}{"_actor_loss.csv"}'
        #path+"_actor_loss.csv"
        critic_loss_path = f'{path}{"_critic_loss.csv"}'
        #path + "_critic_loss.csv"
        entropy_loss_path = f'{path}{"_entropy_loss.csv"}'
        #path+"_entropy_loss.csv"

        #If it is the first iteration write OVER potnetially existing files, else append
        first_itr = self.episode_num <= Config().algorithm.max_round_episodes
        
        with open(actor_loss_path, 'w' if first_itr else 'a') as filehandle:
            writer = csv.writer(filehandle)
            writer.writerow([self.avg_actor_loss])

        with open(critic_loss_path, 'w' if first_itr else 'a') as filehandle:
            writer = csv.writer(filehandle)
            writer.writerow([self.avg_critic_loss])

        with open(entropy_loss_path, 'w' if first_itr else 'a') as filehandle:
            writer = csv.writer(filehandle)
            writer.writerow([self.avg_entropy_loss])


    async def server_test(self, testset, sampler=None, **kwargs):
        #We will return the average reward here
        avg_reward = self.evaluate_policy()
        self.server_reward = avg_reward
        file_name = ""

        if not Config().server.percentile_aggregate:
            file_name = "A2C_RL_SERVER_FED_AVG"
        else:
            file_name = "A2C_RL_SERVER_PERCENTILE_AGGREGATE"
        results_seed_path = f'{Config().results.results_dir}_seed_{Config().server.random_seed}'
        path = f'{results_seed_path}/{file_name}.csv'
        #Config().results.results_dir +"/"+file_name
        
        #first_itr = self.episode_num <= Config().algorithm.max_round_episodes
        #the reason this doesn't work is because episode_num gets reset to 0 when we are in this function


        with open(path, 'w' if self.first_ittr_server else 'a') as filehandle:
                writer = csv.writer(filehandle)
                writer.writerow(self.server_reward)
                
        self.first_ittr_server = False
        return sum(avg_reward)/len(avg_reward)

        
    def evaluate_policy(self, eval_episodes = 10):
        # TODO: after aggregation, rewards are not same across different runs, why?
        self.model.eval()
        
        avg_rewards = []
        for trace_idx in range(3):
            avg_reward = 0
            if self.client_id == 0:
                torch.manual_seed(Config().trainer.manual_seed)
            for _ in range(eval_episodes):
                episode_reward = 0
                done = False
                state = self.env.reset(trace_idx=trace_idx, test= True)
                state = self.obs_normalizer.normalize(state)
                steps = 0
                while not done:
                    probs = self.actor(self.t(state))
                

                    action = self.dist(probs=probs).sample()
                    steps += 1
                    next_state, reward, done, info = self.env.step(action.detach().data.numpy())
                    next_state = self.obs_normalizer.normalize(next_state)
                    state = next_state
                    episode_reward += reward

                avg_reward += episode_reward
            avg_reward /= eval_episodes
            print("------------------")
            print("Average Reward for client %s over trace %s for %s steps is %s" % (str(self.client_id), str(trace_idx), str(steps), str(avg_reward)))
            print("------------------")
            avg_rewards.append(avg_reward)
        self.model.train()
        return avg_rewards
        
