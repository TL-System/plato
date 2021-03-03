import os
import sys
import torch
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets, transforms

# To import modules from the parent directory
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from datasources import base


class DataSource(base.DataSource):
    """The MNIST dataset."""
    def __init__(self, path):
        super().__init__(path)

        _transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        self.trainset = datasets.MNIST(root=self._path,
                                       train=True,
                                       download=True,
                                       transform=_transform)
        self.testset = datasets.MNIST(root=self._path,
                                      train=False,
                                      download=True,
                                      transform=_transform)

    def num_train_examples(self):
        return 60000

    def num_test_examples(self):
        return 10000


datasource = DataSource('data').trainset
batch_size = 16
validation_split = .2
shuffle_dataset = True
random_seed = 42

# Creating data indices for training and validation splits:
dataset_size = len(datasource)
indices = list(range(dataset_size))
split = [[] for j in range(10)]
train_loader = torch.utils.data.DataLoader(datasource, batch_size=1)
index = 0
for batch_index, (faces, labels) in enumerate(train_loader):
    split[int(labels)].append(index)
    index += 1

train_sampler = SubsetRandomSampler(split[2])
valid_sampler = SubsetRandomSampler(split[1])

train_loader = torch.utils.data.DataLoader(datasource,
                                           batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(datasource,
                                                batch_size=batch_size,
                                                sampler=valid_sampler)

# Usage Example:
num_epochs = 10
for epoch in range(num_epochs):
    # Train:
    for batch_index, (faces, labels) in enumerate(train_loader):
        print(labels)
