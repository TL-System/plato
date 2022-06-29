import gym
import torch
from torch import nn
import park
from plato.config import Config

# Actor module, categorical actions only
class A2CActor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim,16),
            nn.LeakyReLU(),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, n_actions),
            nn.Softmax(dim = 0)
        )
        with torch.no_grad():
            for param in self.model.parameters():
                param.data = nn.parameter.Parameter(torch.ones_like(param))
    
    def forward(self, X):
        return self.model(X)

# Critic module
class A2CCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

        with torch.no_grad():
            for param in self.model.parameters():
                param.data = nn.parameter.Parameter(torch.ones_like(param))
    
    def forward(self, X):
        return self.model(X)


class Model:
    """Wrapper class that holds both models"""
    def __init__(self):

        env = park.make(Config().algorithm.env_park_name)

        state_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        self.state_dim = state_dim
        self.n_action = n_actions
        self.actor = A2CActor(state_dim, n_actions)
        self.critic = A2CCritic(state_dim)

    def get_state_dim(self):
        return self.state_dim

    def cpu(self):
        self.actor.cpu()
        self.critic.cpu()

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    @staticmethod
    def get_model(*args):
        """ Obtaining an instance of this model. """
        return Model()