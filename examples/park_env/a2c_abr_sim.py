import numpy as np
import torch
import gym
from torch import nn
import matplotlib.pyplot as plt
import park
import csv
from torch.nn.utils.clip_grad import clip_grad_norm_

ENTROPY_RATIO = 10.0
SEED = 10
GRAD_CLIP_VAL = 10
#Seed set:
torch.manual_seed(SEED)

class StateNormalizer(object):
    def __init__(self, obs_space):
        self.shift = obs_space.low
        self.range = obs_space.high - obs_space.low

    def normalize(self, obs):
        return (obs - self.shift) / self.range

# helper function to convert numpy arrays to tensors
def t(x): return torch.from_numpy(x).float()
# Actor module, categorical actions only
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
            nn.Softmax(dim = 0)
        )
    
    def forward(self, X):
        return self.model(X)

# Critic module
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, X):
        return self.model(X)


# Memory
# Stores results from the networks, instead of calculating the operations again from states, etc.
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

#env = gym.make("CartPole-v1")
env = park.make('abr_sim')
env.seed(SEED)

# config
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
actor = Actor(state_dim, n_actions)
critic = Critic(state_dim)
adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
gamma = 0.99
memory = Memory() 
obs_normalizer = StateNormalizer(env.observation_space)
max_steps = 256

def evaluate_policy(eval_episodes = 10):
    avg_rewards = []
    for trace_idx in range(3):
        avg_reward = 0
        for _ in range(eval_episodes):
            episode_reward = 0
            done = False
            state = env.reset(trace_idx=trace_idx, test= True)
            state = obs_normalizer.normalize(state)
            while not done:
                probs = actor(t(state))
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample()
            
                next_state, reward, done, info = env.step(action.detach().data.numpy())
                next_state = obs_normalizer.normalize(next_state)
                
                state = next_state
                episode_reward += reward
            avg_reward += episode_reward
        avg_reward /= eval_episodes
        avg_rewards.append(avg_reward)
        print("Average Reward over trace %s is %s" % (str(trace_idx), str(avg_reward)))
    with open("evaluations.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(avg_rewards)

# train function
def train(memory, q_val):
    values = torch.stack(memory.values)
    q_vals = np.zeros((len(memory), 1))

    # target values are calculated backward
    # it's super important to handle correctly done states,
    # for those cases we want our to target to be equal to the reward only
    for i, (_, _, reward, done) in enumerate(memory.reversed()):
        q_val = reward + gamma*q_val*(1.0-done)
        q_vals[len(memory)-1 - i] = q_val # store values from the end to the beginning
        
    advantage = torch.Tensor(q_vals) - values
    
    critic_loss = advantage.pow(2).mean()
    adam_critic.zero_grad()
    critic_loss.backward()
    adam_critic.step()
    
    
    entropy_loss = (torch.stack(memory.log_probs) * torch.exp(torch.stack(memory.log_probs))).mean()
    actor_loss = (-torch.stack(memory.log_probs)*advantage.detach()).mean() + entropy_loss * ENTROPY_RATIO
    if GRAD_CLIP_VAL > 0:
            clip_grad_norm_(actor.parameters(), GRAD_CLIP_VAL)
    adam_actor.zero_grad()
    actor_loss.backward()
    adam_actor.step()


    return critic_loss, actor_loss

episode_rewards = []
episode_num = 0 
critic_losses = []
actor_losses = []

trace_idx = 0
while True:
    done = False
    total_reward = 0
    if (episode_num % 700 == 0):
        trace_idx = int(episode_num / 700)
        print( "changed trace to: ", trace_idx )
    state = env.reset(trace_idx=trace_idx)
    state = obs_normalizer.normalize(state)
    steps = 0

    while not done:
        probs = actor(t(state))
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        
        next_state, reward, done, info = env.step(action.detach().data.numpy())

        next_state = obs_normalizer.normalize(next_state)
        
        total_reward += reward
        steps += 1
        memory.add(dist.log_prob(action), critic(t(state)), reward, done)
        
        state = next_state
        
        # train if done or num steps > max_steps
        if done or (steps % max_steps == 0):
            last_q_val = critic(t(next_state)).detach().data.numpy()
            critic_loss, actor_loss = train(memory, last_q_val)
            memory.clear()

    episode_num += 1        
    episode_rewards.append(total_reward)
    print("Episode number: %d, Reward: %d" % (episode_num, total_reward))
    critic_losses.append(critic_loss.detach().numpy())
    actor_losses.append(actor_loss.detach().numpy())
    if episode_num % 50 == 0:
        evaluate_policy()
    
    np.savetxt("episodic_reward.csv", episode_rewards, delimiter =", ")
    np.savetxt("critic_losses.csv", critic_losses, delimiter =", ")
    np.savetxt("actor_losses.csv", actor_losses, delimiter =", ")