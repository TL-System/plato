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

import numpy as np
import torch
import torch.nn.functional as F
from plato.config import Config
from plato.servers import fedavg

""" Helper methods """
def label_to_onehot(target, num_classes):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

criterion = cross_entropy_for_onehot

class Server(fedavg.Server):
    """ An honest-but-curious federated learning server with gradient leakage attack. """

    def __init__(self):
        super().__init__()
        # TODO: Obtain the sized of data and labels from the dataset
        data_size = (1, 3, 32, 32)
        label_size = (1, 100)
        # One particular client, i.e., the first selected client
        self.victim_client = 0
        self.dummy_data = torch.randn(data_size)
        self.dummy_label =  torch.randn(label_size)
        self.optimizer = torch.optim.LBFGS([self.dummy_data, self.dummy_label])
        self.num_iter = 300

    async def process_reports(self):
        """Process the client reports: before aggregating their weights,
           perform the gradient leakage attacks and reconstruct the training data.
        """
        self.deep_leakage_from_gradients(self.updates)
        await self.aggregate_weights(self.updates)

    def deep_leakage_from_gradients(self, updates):
        """ Analyze periodic gradients from certain clients. """
        # Obtain the local updates from clients
        deltas_received = self.compute_weight_deltas(updates)

        # TODO: the server actually has no idea about the local learning rate
        # Convert local updates to gradients
        target_grad = - deltas_received[self.victim_client] / Config().trainer.learning_rate

        # TODO: periodic analysis, which round?
        # Gradient matching
        for iters in range(self.num_iter):
            def closure():
                self.optimizer.zero_grad()
                dummy_pred = self.model(self.dummy_data)
                dummy_onehot_label = F.softmax(self.dummy_label, dim=-1)
                dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
                dummy_grad = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True)

                grad_diff = sum(((dummy_g - traget_g) ** 2).sum() \
                    for dummy_g, traget_g in zip(dummy_grad, target_grad))

                grad_diff.backward()
                return grad_diff
            
            self.optimizer.step(closure)
            if iters % 10 == 0:
                current_loss = closure()
                logging.info(iters, "%.4f" % current_loss.item())

        # TODO: Plot image history
