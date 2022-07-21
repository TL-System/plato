import numpy as np
import torch
import gym
from torch import nn
import matplotlib.pyplot as plt
import park
import csv
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.autograd import Variable
import sys

print("Number of arguments: ", len(sys.argv), " arguments.")
print("Argument List:", str(sys.argv))

seed_number = int(''.join(filter(str.isdigit, str(sys.argv[1]))))

ENTROPY_RATIO = 10.0
ENTROPY_DECAY = 0.00004
ENTROPY_MIN = 0
SEED = seed_number
GRAD_CLIP_VAL = 10
MAX_EPISODES = 24000
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
actor_loss = None
critic_loss = None
entropy_loss = None
gamma = 0.99
memory = Memory() 
obs_normalizer = StateNormalizer(env.observation_space)
max_steps = 256
traces_per_task = 10
difficulty_levels = 6

def evaluate_policy(eval_episodes = traces_per_task):
    avg_rewards = []
    for trace_idx in range(difficulty_levels):
        avg_reward = 0
        for epi in range(eval_episodes):
            episode_reward = 0
            done = False
            state = env.reset(trace_idx=trace_idx * traces_per_task + epi, test= True)
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
    with open("evaluations_"+str(SEED)+".csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(avg_rewards)

# train function
def train(memory, q_val):
    # sorry for seeing global variables here!
    global critic_loss
    global actor_loss
    global entropy_loss

    l2_loss = torch.nn.MSELoss(reduction='mean')
    values = torch.stack(memory.values)
    q_vals = np.zeros((len(memory), 1))

    # target values are calculated backward
    # it's super important to handle correctly done states,
    # for those cases we want our to target to be equal to the reward only
    for i, (_, _, reward, done) in enumerate(memory.reversed()):
        q_val = reward + gamma*q_val*(1.0-done)
        q_vals[len(memory)-1 - i] = q_val # store values from the end to the beginning
        
    critic_loss = l2_loss(values, torch.Tensor(q_vals)) #advantage.pow(2).mean()
    
    advantage = torch.Tensor(q_vals) - values.detach()
    entropy_loss = (torch.stack(memory.log_probs) * torch.exp(torch.stack(memory.log_probs))).mean()
    entropy_coef = max((ENTROPY_RATIO - (episode_num/max_steps) * ENTROPY_DECAY), ENTROPY_MIN)
    actor_loss = (-torch.stack(memory.log_probs)*advantage.detach()).mean() + entropy_loss * entropy_coef
    if GRAD_CLIP_VAL > 0:
            clip_grad_norm_(actor.parameters(), GRAD_CLIP_VAL)

def estimate_fisher(last_q_val, trace_idx):
    train(memory, last_q_val) #updates critic and actor loss
    adam_critic.zero_grad()
    critic_loss.backward()

    adam_actor.zero_grad()
    actor_loss.backward()
    
    fisher_critic = {}
    optpar_critic = {}
    fisher_actor = {}
    optpar_actor = {}

    # Get fisher for critic model
    fisher_critic[trace_idx] = [] #self.net.parameters().grad.data.clone().pow(2)
    optpar_critic[trace_idx] = [] #self.net.parameters().data.clone()
    for p in critic.parameters():
      pd = p.data.clone()
      pg = p.grad.data.clone().pow(2)
      optpar_critic[trace_idx].append(pd)
      fisher_critic[trace_idx].append(pg)

    # Get fisher for actor model
    fisher_actor[trace_idx] = []
    optpar_actor[trace_idx] = []
    for p in actor.parameters():
      pd = p.data.clone()
      pg = p.grad.data.clone().pow(2)
      optpar_actor[trace_idx].append(pd)
      fisher_actor[trace_idx].append(pg)

    return fisher_critic, fisher_actor


episode_rewards = []
episode_num = 0 
critic_losses = []
actor_losses = []
entropy_losses = []
fisher_actors = []
fisher_critics = []

trace_idx = 0
while episode_num < MAX_EPISODES:
    done = False
    total_reward = 0
    #if (episode_num % 700 == 0):
    #    trace_idx = int(episode_num / 700)
    #    print( "changed trace to: ", trace_idx )
    state = env.reset()
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
            critic_fisher, actor_fisher = estimate_fisher(last_q_val, trace_idx)
            adam_critic.step()
            adam_actor.step()
            memory.clear()

    episode_num += 1        
    episode_rewards.append(total_reward)
    print("Episode number: %d, Reward: %d" % (episode_num, total_reward))
    critic_losses.append(critic_loss.detach().numpy())
    actor_losses.append(actor_loss.detach().numpy())
    entropy_losses.append(entropy_loss.detach().numpy())
    
    
    actor_fisher_sum = 0
    
    for i, p in enumerate(actor.parameters()):
        l = Variable(actor_fisher[trace_idx][i])
        actor_fisher_sum += l.sum()

    fisher_actors.append(actor_fisher_sum)
    
    critic_fisher_sum = 0
    
    for i, p in enumerate(critic.parameters()):
        l = Variable(critic_fisher[trace_idx][i])
        critic_fisher_sum += l.sum()
    fisher_critics.append(critic_fisher_sum)

    if episode_num % 50 == 0:
        evaluate_policy()

    np.savetxt("episodic_reward_"+str(SEED)+".csv", episode_rewards, delimiter =", ")
    np.savetxt("critic_losses_"+str(SEED)+".csv", critic_losses, delimiter =", ")
    np.savetxt("actor_losses_"+str(SEED)+".csv", actor_losses, delimiter =", ")
    np.savetxt("entropy_losses_"+str(SEED)+".csv", entropy_losses, delimiter =", ")
    np.savetxt("fisher_actors_"+str(SEED)+".csv", fisher_actors, delimiter =", ")
    np.savetxt("fisher_critics_"+str(SEED)+".csv", fisher_critics, delimiter =", ")