log_interval = 10
"""
The training and testing loop.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from models import base

# CUDA settings
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def extract_weights(model):
    """Extract weights from a model passed in as a parameter."""
    weights = []
    for name, weight in model.to(torch.device('cpu')).named_parameters():
        if weight.requires_grad:
            weights.append((name, weight.data))

    return weights


def load_weights(model, weights):
    """Load the model weights passed in as a parameter."""
    updated_state_dict = {}
    for name, weight in weights:
        updated_state_dict[name] = weight

    model.load_state_dict(updated_state_dict, strict=False)


def get_trainloader(trainset, batch_size):
    """Obtain the data loader for the training dataset."""
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


def get_testloader(testset, batch_size):
    """Obtain the data loader for the testing dataset."""
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


def train(model, train_loader, optimizer, epochs):
    """Train the model."""
    model.to(device)
    model.train()

    criterion = model.loss_criterion

    for epoch in range(1, epochs + 1):
        for batch_id, (image, label) in enumerate(train_loader):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            if batch_id % log_interval == 0:
                logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, epochs, loss.item()))


def test(model, test_loader):
    """Test the model."""
    model.to(device)
    # We should set the model to evaluation mode to accommodate Dropouts
    model.eval()

    criterion = model.loss_criterion

    test_loss = 0
    correct = 0
    total = len(test_loader.dataset)

    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            # sum up the batch loss
            test_loss += criterion(output, label).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    accuracy = correct / total
    logging.debug('Accuracy: {:.2f}%'.format(100 * accuracy))

    return accuracy
