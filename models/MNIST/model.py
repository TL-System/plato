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

# CUDA settings
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class Generator(load_data.Generator):
    """ A generator for the MNIST dataset. """

    # Extract MNIST data using torchvision datasets
    def read(self, path):
        self.trainset = datasets.MNIST(
            path, train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
        self.testset = datasets.MNIST(
            path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
        self.labels = list(self.trainset.classes)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def optimizer(model):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def train_loader(trainset, batch_size):
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


def test_loader(testset, batch_size):
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


def extract_weights(model):
    weights = []
    for name, weight in model.to(torch.device('cpu')).named_parameters():  # pylint: disable=no-member
        if weight.requires_grad:
            weights.append((name, weight.data))

    return weights


def load_weights(model, weights):
    updated_state_dict = {}
    for name, weight in weights:
        updated_state_dict[name] = weight

    model.load_state_dict(updated_state_dict, strict=False)


def train(model, train_loader, optimizer, epochs):
    model.to(device)
    model.train()
    for epoch in range(1, epochs + 1):
        for batch_id, (image, label) in enumerate(train_loader):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = F.nll_loss(output, label)
            loss.backward()
            optimizer.step()
            if batch_id % log_interval == 0:
                logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, epochs, loss.item()))


def test(model, testloader):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    total = len(testloader.dataset)
    with torch.no_grad():
        for image, label in testloader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            # sum up batch loss
            test_loss += F.nll_loss(output, label, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    accuracy = correct / total
    logging.debug('Accuracy: {:.2f}%'.format(100 * accuracy))

    return accuracy
