import torch
import torch.nn as nn
from timm.models.layers.mlp import Mlp
from plato.config import Config
import logging


class ValueNet(nn.Module):
    def __init__(self, input_channel):
        super(ValueNet, self).__init__()

        self.f = Mlp(in_features=input_channel, out_features=1, act_layer=nn.Sigmoid)
        self.flatten = nn.Flatten()
        self.least_value = 0
        if hasattr(Config().parameters.model, "num_classes"):
            self.least_value = 1.0 / Config().parameters.model.num_classes

        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=Config().parameters.architect.value_net.learning_rate
        )
        self.criterion = nn.HuberLoss()

    def forward(self, alpha):
        alpha = self.flatten(alpha)
        acc = self.f(alpha)
        return acc

    def update(self, accuracy_list, client_id_list, alphas_normal, alphas_reduce):
        accuracy = torch.tensor(accuracy_list)
        alphas = torch.stack(
            [
                alphas_normal[torch.tensor(client_id_list) - 1],
                alphas_reduce[torch.tensor(client_id_list) - 1],
            ],
            dim=-1,
        )
        accuracy, alphas = accuracy.to(Config().device()), alphas.to(Config().device())
        v = self.forward(alphas)
        accuracy = torch.reshape(accuracy, (v.shape[0], 1))
        self.optimizer.zero_grad()
        loss = self.criterion(v, accuracy)
        loss.backward()
        self.optimizer.step()

        logging.info("estimated value: %s", str(v[:, 0]))
        logging.info("actual accuracy: %s", str(accuracy[:, 0]))
        logging.info("value net loss %.3f", loss.item())
