"""
An honest-but-curious federated learning server which can
analyze periodic gradients from certain clients to
perform the gradient leakage attacks and
reconstruct the training data of the victim clients.


Reference:

Zhu et al., "Deep Leakage from Gradients,"
in Advances in Neural Information Processing Systems 2019.

https://papers.nips.cc/paper/2019/file/60a6c4002cc7b29142def8871531281a-Paper.pdf
"""

import asyncio
import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from plato.config import Config
from plato.servers import fedavg
from torchvision import transforms

""" Helper methods """


def label_to_onehot(target, num_classes):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(
        0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


criterion = cross_entropy_for_onehot
tt = transforms.ToPILImage()


class Server(fedavg.Server):
    """ An honest-but-curious federated learning server with gradient leakage attack. """

    def __init__(self):
        super().__init__()

    async def process_reports(self):
        """Process the client reports: before aggregating their weights,
           perform the gradient leakage attacks and reconstruct the training data.
        """
        if self.current_round == Config().algorithm.attack_round:
            self.deep_leakage_from_gradients(self.updates)
        await self.aggregate_weights(self.updates)

    def deep_leakage_from_gradients(self, updates):
        """ Analyze periodic gradients from certain clients. """
        # Obtain the local updates from clients
        deltas_received = self.compute_weight_deltas(updates)

        # Generate dummy items
        torch.manual_seed(50)
        data_size = self.testset.data[0].shape
        if len(data_size) == 2:
            data_size = (1, 1, data_size[0], data_size[1])
        else:
            data_size = (1, data_size[2], data_size[0], data_size[1])
        dummy_data = torch.randn(data_size).to(
            Config().device()).requires_grad_(True)
        dummy_label = torch.randn(1).to(
            Config().device()).requires_grad_(True)
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

        plt.imshow(tt(dummy_data[0].cpu()))
        plt.title("Dummy data")
        logging.info("[Gradient Leakage Attacking...] Dummy label is %d.",
                     torch.argmax(dummy_label, dim=-1).item())

        # TODO: the server actually has no idea about the local learning rate
        # Convert local updates to gradients
        target_grad = []
        for delta in deltas_received[Config().algorithm.victim_client].values():
            target_grad.append(- delta / Config().trainer.learning_rate)
        # target_grad = - deltas_received[self.victim_client] / Config().trainer.learning_rate

        # TODO: periodic analysis, which round?
        # Gradient matching
        history = []
        for iters in range(Config().algorithm.num_iters):
            def closure():
                optimizer.zero_grad()
                dummy_pred = self.trainer.model(dummy_data)
                dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                dummy_loss = criterion(dummy_pred, dummy_onehot_label)
                dummy_grad = torch.autograd.grad(
                    dummy_loss, self.trainer.model.parameters(), create_graph=True)

                grad_diff = sum(((dummy_g - traget_g) ** 2).sum()
                                for dummy_g, traget_g in zip(dummy_grad, target_grad))

                grad_diff.backward()
                return grad_diff

            optimizer.step(closure)
            if iters % 10 == 0:
                current_loss = closure()
                logging.info("[Gradient Leakage Attacking...] Iter %d: Gradient Difference %.4f",
                             iters, current_loss.item())
                history.append(tt(dummy_data[0].cpu()))

        plt.figure(figsize=(12, 8))
        for i in range(Config().algorithm.num_iters // Config().algorithm.log_interval):
            plt.subplot(3, 10, i + 1)
            plt.imshow(history[i])
            plt.title("iter=%d" % (i * 10))
            plt.axis('off')
        logging.info("[Gradient Leakage Attacking...] Reconstructed label is %d.",
                     torch.argmax(dummy_label, dim=-1).item())
        plt.show()
