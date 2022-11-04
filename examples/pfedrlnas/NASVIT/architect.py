import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import copy
from .models.attentive_nas_dynamic_model import AttentiveNasDynamicModel
from plato.config import Config
import logging
from .misc.bigconfig import get_config


class Architect(nn.Module):
    def __init__(
        self,
        model=None,
        momentum=0.9,
        weight_decay=3e-4,
        arch_learning_rate=3e-3,
        arch_weight_decay=3e-1,
        arch_baseline_decay=0.99,
    ):
        super().__init__()
        self.model = AttentiveNasDynamicModel(supernet=get_config().supernet_config)
        if hasattr(Config().parameters.architect, "pretrain_path"):
            weight = torch.load(
                Config().parameters.architect.pretrain_path, map_location="cpu"
            )["model"]
            del weight["classifier.linear.linear.weight"]
            del weight["classifier.linear.linear.bias"]
            self.model.load_state_dict(weight, strict=False)

        self.client_nums = Config().clients.total_clients
        if (
            hasattr(Config().parameters.architect, " personalize_last")
            and Config().parameters.architect.personalize_last
        ):
            self.lasts = [
                copy.deepcopy(self.model.classifier) for _ in range(self.client_nums)
            ]
        self._initialize_alphas()
        self.optimizer = torch.optim.Adam(
            self.arch_parameters(),
            lr=Config().parameters.architect.learning_rate,
            betas=(0.5, 0.999),
            weight_decay=Config().parameters.architect.weight_decay,
        )
        # self.optimizer=torch.optim.SGD(self.arch_parameters(),lr=0.5,momentum=0.9)
        self.baseline = None  # 1.0 / Config().parameters.model.num_classes
        self.baseline_decay = Config().parameters.architect.baseline_decay
        self.device = Config().device()
        # value net
        if hasattr(Config().parameters.architect, "value_net"):
            pass

    def step(self, epoch_acc, neg_ratio, epoch_index, client_id_list):
        rewards = self._compute_reward(epoch_acc, neg_ratio, client_id_list)
        if (
            hasattr(Config().parameters.architect, "natural")
            and Config().parameters.architect.natural
        ):
            grad = self._compute_grad_natural(
                self.alphas, rewards, epoch_index, client_id_list
            )
        else:
            grad = self._compute_grad(self.alphas, rewards, epoch_index, client_id_list)
        logging.info(str(grad[client_id_list[0] - 1][0]))
        self.alphas.grad.copy_(grad)
        self.optimizer.step()
        self.optimizer.zero_grad()
        logging.info(str(self.alphas[client_id_list[0] - 1][0]))

    def _compute_grad(self, alphas, rewards, index_list, client_id_list):
        grad = torch.zeros(alphas.size())
        for client_idx in range(len(rewards)):
            for edge_idx in range(len(self.stop_index) - 1):
                prob = F.softmax(
                    alphas[client_id_list[client_idx] - 1][
                        self.stop_index[edge_idx]
                        + 1 : self.stop_index[edge_idx + 1]
                        + 1
                    ],
                    dim=-1,
                )
                reward = rewards[client_idx]
                index = index_list[client_idx]
                client_grad = torch.Tensor(prob.shape)
                client_grad.copy_(prob)
                # nabla _alpha { log(p(g_i)) } = (p_1, ..., p_i-1, ..., p_N)
                index_prob = client_grad[index[edge_idx]]
                client_grad[index[edge_idx]] = index_prob - 1
                grad[client_id_list[client_idx] - 1][
                    self.stop_index[edge_idx] + 1 : self.stop_index[edge_idx + 1] + 1
                ] += (reward * client_grad)
            grad /= len(rewards)
        return grad

    def _compute_reward(self, accuracy_list, neg_ratio, client_id_list):
        # scale accuracy to 0-1
        accuracy = np.array(accuracy_list)
        if hasattr(Config().parameters.architect, "value_net"):
            reward = torch.from_numpy(accuracy)
            alphas = torch.stack(
                [
                    self.alphas_normal[torch.tensor(client_id_list) - 1],
                    self.alphas_reduce[torch.tensor(client_id_list) - 1],
                ],
                dim=-1,
            )
            alphas, reward = alphas.to(self.device), reward.to(self.device)
            v = self.value_net.forward(alphas)
            v = v[:, 0]
            least_value = 1.0 / self.client_nums
            v = torch.where(
                v < least_value, torch.ones(v.size()).to(self.device) * least_value, v
            )
            reward = reward - v
            return reward.detach().cpu()
        else:
            if self.baseline == None:
                self.baseline = np.mean(accuracy)  # - np.mean(neg_ratio)
            reward = accuracy - self.baseline  # - neg_ratio
            logging.info("reward: %s", str(reward))
            self.baseline = (
                self.baseline_decay * np.mean(accuracy)
                + (1 - self.baseline_decay) * self.baseline
            )
            return reward

    def _compute_grad_natural(self, alphas, rewards, index_list, client_id_list):
        grad = torch.zeros(alphas.size())
        for client_idx in range(len(rewards)):
            for edge_idx in range(len(self.stop_index) - 1):
                prob = F.softmax(
                    alphas[client_id_list[client_idx] - 1][
                        self.stop_index[edge_idx]
                        + 1 : self.stop_index[edge_idx + 1]
                        + 1
                    ],
                    dim=-1,
                )
                print(prob.shape[0])
                reward = rewards[client_idx]
                index = index_list[client_idx]
                client_grad = torch.Tensor(prob.shape)
                client_grad.copy_(prob)
                # nabla _alpha { log(p(g_i)) } = (p_1, ..., p_i-1, ..., p_N)
                for edge_idx in range(client_grad.shape[0]):
                    index_prob = client_grad[edge_idx][index[edge_idx]]
                    client_grad[edge_idx][index[edge_idx]] = index_prob - 1
                    dalpha = client_grad
                    Fish = torch.matmul(dalpha, torch.transpose(dalpha, 0, 1))
                    Finverse = torch.pinverse(Fish)
                grad[client_id_list[client_idx] - 1][
                    self.stop_index[edge_idx] + 1 : self.stop_index[edge_idx] + 1
                ] += reward * torch.matmul(Finverse, client_grad)
            grad /= len(rewards)
        return grad

    def _initialize_alphas(self):
        cfg_candidate = self.model.cfg_candidates
        stop_index = [-1]
        resolution = cfg_candidate["resolution"]
        stop_index.append(len(resolution) - 1)
        width = cfg_candidate["width"]
        for width_candidate in width:
            stop_index.append(stop_index[-1] + len(width_candidate))
        depth = cfg_candidate["depth"]
        for depth_candidate in depth:
            stop_index.append(stop_index[-1] + len(depth_candidate))
        kernel_size = cfg_candidate["kernel_size"]
        for kernel_size_candidate in kernel_size:
            stop_index.append(stop_index[-1] + len(kernel_size_candidate))
        expand_ratio = cfg_candidate["expand_ratio"]
        for expand_ratio_candidate in expand_ratio:
            stop_index.append(stop_index[-1] + len(expand_ratio_candidate))

        self.stop_index = stop_index
        self.alphas = Variable(
            torch.zeros(self.client_nums, self.stop_index[-1] + 1), requires_grad=True
        )
        self._arch_parameters = [self.alphas]

        input = Variable(torch.Tensor(self.alphas.size()), requires_grad=False)
        logits = input * self.alphas
        loss = torch.mean(logits)
        loss.backward()

        self.model.zero_grad()
        self.zero_grad()

    def arch_parameters(self):
        return self._arch_parameters

    def get_index(self, prob, length):
        prob = F.softmax(prob, dim=-1)
        prob = prob.detach().numpy()
        prob /= prob.sum()
        mask_idx = np.random.choice(range(length), 1, replace=False, p=prob)
        return mask_idx[0]

    def sample_config(self, client_id):
        cfg_candidate = self.model.cfg_candidates
        alpha = self.alphas[client_id]
        cfg = {}
        stop_index_point = 1
        length = self.stop_index[stop_index_point] + 1
        prob = alpha[: self.stop_index[stop_index_point] + 1]
        cfg["resolution"] = cfg_candidate["resolution"][self.get_index(prob, length)]
        stop_index_point += 1
        for candidate_name in ["width", "depth", "kernel_size", "expand_ratio"]:
            candidate_list = []
            for candidate_candidate in cfg_candidate[candidate_name]:
                length = (
                    self.stop_index[stop_index_point]
                    - self.stop_index[stop_index_point - 1]
                )
                prob = alpha[
                    self.stop_index[stop_index_point - 1]
                    + 1 : self.stop_index[stop_index_point]
                    + 1
                ]
                candidate_list.append(candidate_candidate[self.get_index(prob, length)])
                stop_index_point += 1
            cfg[candidate_name] = candidate_list
        return cfg

    def extract_index(self, subnets_config):
        cfg_candidate = self.model.cfg_candidates
        epoch_index = []
        for cfg in subnets_config:
            index = []
            resolution = cfg_candidate["resolution"]
            index.append(resolution.index(cfg["resolution"]))
            for candidate_name in ["width", "depth", "kernel_size", "expand_ratio"]:
                candidate = cfg_candidate[candidate_name]
                for candidate, candidate_list in zip(cfg[candidate_name], candidate):
                    index.append(candidate_list.index(candidate))
            epoch_index.append(index)
        return epoch_index


if __name__ == "__main__":
    arch = Architect()
