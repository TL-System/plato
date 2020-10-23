"""
The FashionMNIST model and data generator.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from utils import dataloader

# Training settings
lr = 0.01
momentum = 0.5
log_interval = 10


class Generator(dataloader.Generator):
    """Generator for FashionMNIST dataset."""

    # Extract FashionMNIST data using torchvision datasets
    def read(self, path):
        self.trainset = datasets.FashionMNIST(
            path, train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
        self.testset = datasets.FashionMNIST(
            path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
        self.labels = list(self.trainset.classes)


class Net(nn.Module):
    """A feedforward neural network."""
    
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        """Defining the feedforward neural network."""
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def get_optimizer(model):
    """Obtain the optimizer used for training the model."""
    return optim.Adam(model.parameters(), lr=lr)


def get_trainloader(trainset, batch_size):
    """Obtain the data loader for the training dataset."""
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


def get_testloader(testset, batch_size):
    """Obtain the data loader for the testing dataset."""
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


def extract_weights(model):
    """Extract weights from a model passed in as a parameter."""
    weights = []
    for name, weight in model.named_parameters():
        weights.append((name, weight.data))

    return weights


def load_weights(model, weights):
    """Load the model weights passed in as a parameter."""
    updated_state_dict = {}
    for name, weight in weights:
        updated_state_dict[name] = weight

    model.load_state_dict(updated_state_dict, strict=False)


def train(model, trainloader, optimizer, epochs):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        for batch_id, (image, label) in enumerate(trainloader):
            output = model(image)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_id % log_interval == 0:
                logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, epochs, loss.item()))


def test(model, testloader):
    """Test the model."""
    with torch.no_grad():
        correct = 0
        total = 0
        for image, label in testloader:
            outputs = model(image)
            predicted = torch.argmax(
                outputs, dim=1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    accuracy = correct / total
    logging.debug('Accuracy: {:.2f}%'.format(100 * accuracy))

    return accuracy
