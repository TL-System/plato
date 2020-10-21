# pylint: skip-file

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import utils.load_data as load_data

# Training settings
lr = 0.01
momentum = 0.5
log_interval = 10


class Generator(load_data.Generator):  # CHECKME
    """Generator for UNNAMED dataset."""

    # Extract UNNAMED data using torchvision datasets
    def read(self, path):
        self.trainset = datasets.UNNAMED(
            path, train=True, download=True, transform=transforms.Compose([
                """
                    Add transforms here...
                """
            ]))
        self.testset = datasets.UNNAMED(
            path, train=False, transform=transforms.Compose([
                """
                    Add transforms here...
                """
            ]))
        self.labels = list(self.trainset.classes)


class Net(nn.Module):  # CHECKME
    def __init__(self):
        super(Net, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


@property
def optimizer(model):  # CHECKME
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


@property
def train_loader(trainset, batch_size):  # CHECKME
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


@property
def test_loader(testset, batch_size):  # CHECKME
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


def extract_weights(model):  # CHECKME
    weights = []
    for UNNAMED, weight in model.UNNAMEDd_parameters():
        if weight.requires_grad:
            weights.append((UNNAMED, weight.data))

    return weights


def load_weights(model, weights):  # CHECKME
    updated_weights_dict = {}
    for UNNAMED, weight in weights:
        updated_weights_dictUNNAMED = weight

    model.load_state_dict(updated_weights_dict, strict=False)


def train(model, train_loader, optimizer, epochs):  # CHECKME
    """
        Set up for training here...
    """

    for epoch in range(1, epochs + 1):
        for batch_id, (image, label) in enumerate(train_oader):
            """
                Train model here...
            """

            if batch_id % log_interval == 0:
                logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, epochs, loss.item()))


def test(model, testloader):  # CHECKME
    """
        Set up for testing here...
    """

    correct = 0
    total = 0
    with torch.no_grad():
        for image, label in testloader:
            """
                Test model here...
            """

    accuracy = correct / total
    logging.debug('Accuracy: {:.2f}%'.format(100 * accuracy))

    return accuracy
