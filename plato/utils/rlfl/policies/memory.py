import random

import numpy as np
import torch

from plato.utils.rlfl.config import TD3Config as Config


class ReplayMemory:
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_size,
                 capacity,
                 seed,
                 recurrent=False,
                 varied_per_round=False):
        random.seed(seed)
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0
        self.recurrent = recurrent
        self.varied_per_round = varied_per_round

        if self.recurrent:
            self.h = np.zeros((self.capacity, hidden_size))
            self.nh = np.zeros((self.capacity, hidden_size))
            self.c = np.zeros((self.capacity, hidden_size))
            self.nc = np.zeros((self.capacity, hidden_size))
            # if varied_per_round:
            self.state = [0] * self.capacity
            self.action = [0] * self.capacity
            self.reward = [0] * self.capacity
            self.next_state = [0] * self.capacity
            self.done = [0] * self.capacity
        else:
            self.state = np.zeros((self.capacity, state_dim))
            self.action = np.zeros((self.capacity, action_dim))
            self.reward = np.zeros((self.capacity, 1))
            self.next_state = np.zeros((self.capacity, state_dim))
            self.done = np.zeros((self.capacity, 1))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def push(self, data):
        self.state[self.ptr] = data[0]
        self.action[self.ptr] = data[1]
        self.reward[self.ptr] = data[2]
        self.next_state[self.ptr] = data[3]
        self.done[self.ptr] = data[4]

        if self.recurrent:
            self.h[self.ptr] = data[5].detach().cpu()
            self.c[self.ptr] = data[6].detach().cpu()
            self.nh[self.ptr] = data[7].detach().cpu()
            self.nc[self.ptr] = data[8].detach().cpu()

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self):
        ind = np.random.randint(0, self.size, size=int(Config().batch_size))

        if not self.recurrent:
            state = self.state[ind]
            action = self.action[ind]
            reward = self.reward[ind]
            next_state = self.next_state[ind]
            done = self.done[ind]

            return state, action, reward, next_state, done

        h = torch.tensor(self.h[ind][None, ...],
                         requires_grad=True,
                         dtype=torch.float).to(self.device)
        c = torch.tensor(self.c[ind][None, ...],
                         requires_grad=True,
                         dtype=torch.float).to(self.device)
        nh = torch.tensor(self.nh[ind][None, ...],
                          requires_grad=True,
                          dtype=torch.float).to(self.device)
        nc = torch.tensor(self.nc[ind][None, ...],
                          requires_grad=True,
                          dtype=torch.float).to(self.device)

        # state = torch.FloatTensor([self.state[i] for i in ind]).to(self.device)
        # action = torch.FloatTensor([self.action[i] for i in ind]).to(self.device)
        # reward = torch.FloatTensor([self.reward[i] for i in ind]).to(self.device)
        # next_state = torch.FloatTensor([self.next_state[i] for i in ind]).to(
        #     self.device)
        # done = torch.FloatTensor([self.done[i] for i in ind]).to(self.device)
        state = [torch.FloatTensor(self.state[i]).to(self.device) for i in ind]
        action = [
            torch.FloatTensor(self.action[i]).to(self.device) for i in ind
        ]
        reward = [self.reward[i] for i in ind]
        next_state = [
            torch.FloatTensor(self.next_state[i]).to(self.device) for i in ind
        ]
        done = [self.done[i] for i in ind]

        return state, action, reward, next_state, done, h, c, nh, nc

    def __len__(self):
        return self.size


class ReplayMemory2:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, data):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        batch = random.sample(self.buffer, Config().batch_size)
        if Config().recurrent_actor:
            state, action, reward, next_state, done, h, c, nh, nc = map(
                np.stack, zip(*batch))
            return state, action, reward, next_state, done, h, c, nh, nc
        else:
            state, action, reward, next_state, done = map(
                np.stack, zip(*batch))
            return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
