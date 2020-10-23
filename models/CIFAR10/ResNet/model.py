"""
The CIFAR-10 model and data generator.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from utils import dataloader

# Training settings
lr = 0.001
log_interval = 10

# Cuda settings
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class Generator(dataloader.Generator):
    """A generator for the CIFAR-10 dataset."""

    def __init__(self):
        super().__init__()
        self.testset = None


    def read(self, path):
        """Extract the CIFAR-10 data using torchvision datasets."""
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.247, 0.243, 0.261])

        transform = transforms.Compose([transforms.Resize((32,32)),
                                        transforms.ToTensor(),
                                        normalize
                                        ])
        transform_train = transforms.Compose([
                            transforms.Resize((32,32)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(10),
                            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                            transforms.ToTensor(),
                            normalize
                            ])

        self.trainset = datasets.CIFAR10(path, train=True,
                                        download=True, transform= transform_train)
        self.testset = datasets.CIFAR10(path, train=False,
                                        download=True, transform= transform)

        self.labels = list(self.trainset.classes)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def Net():
    return ResNet18()


def get_optimizer(model):
    """Obtain the optimizer used for training the model."""
    return optim.Adam(model.parameters(), lr=lr)


def get_trainloader(trainset, batch_size):
    """Obtain the data loader for the training dataset."""
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


def get_testloader(testset, batch_size):
    """Obtain the data loader for the testing dataset."""
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


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


def train(model, trainloader, optimizer, epochs):
    """Train the model."""
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        for batch_id, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_id % log_interval == 0:
                logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, epochs, loss.item()))


def test(model, testloader):
    """Test the model."""
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    logging.debug('Accuracy: {:.2f}%'.format(100 * accuracy))

    return accuracy
