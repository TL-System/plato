import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import copy
from .stale import compute_stale_grad_alpha
from plato.config import Config
from .model_search import Network

class Architect(nn.Module):
  def __init__(self, model=None, momentum=0.9,weight_decay=3e-4,arch_learning_rate=3e-3,arch_weight_decay=3e-1,arch_baseline_decay=0.99):
    super().__init__()
    self.network_momentum = Config().parameters.architect.momentum
    self.network_weight_decay = Config().parameters.architect.weight_decay
    self.model = Network()
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=Config().parameters.architect.arch_learning_rate, betas=(0.5, 0.999), weight_decay=Config().parameters.architect.arch_weight_decay)
    self.baseline = None
    self.baseline_decay = Config().parameters.architect.arch_baseline_decay


  def step(self, epoch_acc, epoch_index_normal, epoch_index_reduce):
    self._compute_grad(self.model.alphas_normal, epoch_acc, epoch_index_normal)
    self._compute_grad(self.model.alphas_reduce, epoch_acc, epoch_index_reduce)
    self.optimizer.step()
    self.optimizer.zero_grad()

  def _compute_grad(self, alphas, accuracy_list, index_list):
    grad = torch.zeros(alphas.size())
    prob = F.softmax(alphas, dim=-1)
    rewards = self._compute_reward(accuracy_list)
    for client_idx in range(len(rewards)):
      reward = rewards[client_idx]
      index = index_list[client_idx]
      client_grad = torch.Tensor(prob.shape)
      client_grad.copy_(prob)
      # nabla _alpha { log(p(g_i)) } = (p_1, ..., p_i-1, ..., p_N)
      for edge_idx in range(client_grad.shape[0]):
        index_prob = client_grad[edge_idx][index[edge_idx]]
        client_grad[edge_idx][index[edge_idx]] = index_prob -1
      grad += reward * client_grad
    grad /= len(rewards)
    alphas.grad = grad

  def _compute_reward(self,accuracy_list):
    # scale accuracy to 0-1
    avg_acc = torch.mean(torch.Tensor(accuracy_list)) / 100
    if self.baseline is None:
      self.baseline = avg_acc
    else:
      self.baseline += self.baseline_decay * (avg_acc - self.baseline)
    # reward = accuracy - baseline
    return [accuracy_list[i]/100 - self.baseline for i in range(len(accuracy_list))]

  def stale_step(self, epoch_acc, epoch_index_normal, epoch_index_reduce, stale_alphas_normal, stale_alphas_reduce, stale_acc, stale_index_normal, stale_index_reduce):
    self._compute_stale_grad(self.model.alphas_normal, epoch_acc, epoch_index_normal, stale_alphas_normal, stale_acc, stale_index_normal)
    self._compute_stale_grad(self.model.alphas_reduce, epoch_acc, epoch_index_reduce, stale_alphas_reduce , stale_acc, stale_index_reduce)
    self.optimizer.step()
    self.optimizer.zero_grad()

  def _compute_stale_grad(self, alphas, accuracy_list, index_list, old_alphas, old_accuracy, old_index):
    grad = torch.zeros(alphas.size())
    prob = F.softmax(alphas, dim=-1)
    rewards = self._compute_reward(accuracy_list)
    for client_idx in range(len(rewards)):
      reward = rewards[client_idx]
      index = index_list[client_idx]
      client_grad = torch.Tensor(prob.shape)
      client_grad.copy_(prob)
      # nabla _alpha { log(p(g_i)) } = (p_1, ..., p_i-1, ..., p_N)
      for edge_idx in range(client_grad.shape[0]):
        index_prob = client_grad[edge_idx][index[edge_idx]]
        client_grad[edge_idx][index[edge_idx]] = index_prob -1
      grad += reward * client_grad

    # stale update
    old_reward = self._compute_reward(old_accuracy)
    for stale_idx in range(len(old_alphas)):
      stale_grad = compute_stale_grad_alpha(old_index[stale_idx], old_alphas[stale_idx], alphas)
      grad += old_reward[stale_idx] * stale_grad
    grad /= (len(rewards)+len(old_alphas))
    alphas.grad = grad





if __name__ == '__main__':
  from model_search import Network
  class TMP:
    def __init__(self):
      self.momentum = 0.9
      self.weight_decay = 3E-4
      self.arch_learning_rate = 1E-3
      self.arch_weight_decay = 3e-4
  args = TMP()
  criterion = nn.CrossEntropyLoss()
  model = Network(16, 10, 8, criterion)
  architect = Architect(model,args)
  pass