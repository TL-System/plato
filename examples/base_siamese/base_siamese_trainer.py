"""
Implement the trainer for base siamese method.

"""

import torch
import torch.nn as nn

from plato.config import Config
from plato.trainers import basic


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs, label):
        output1, output2 = outputs
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)

        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) + (label) *
            torch.pow(torch.clamp(self.margin -
                                  euclidean_distance, min=0.0), 2))

        return loss_contrastive


class Trainer(basic.Trainer):

    def __init__(self, model=None):
        super().__init__(model)

    def loss_criterion(self, model):
        """ The loss computation. """
        # define the loss computation instance
        defined_margin = Config().trainer.margin
        constrative_loss_computer = ContrastiveLoss(margin=defined_margin)
        return constrative_loss_computer