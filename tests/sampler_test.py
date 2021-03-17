import os
import sys
import numpy as np
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
        super().__init__()

        _transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        self.trainset = datasets.MNIST(root=path,
                                       train=True,
                                       download=True,
                                       transform=_transform)
        self.testset = datasets.MNIST(root=path,
                                      train=False,
                                      download=True,
                                      transform=_transform)

    def num_train_examples(self):
        return 60000

    def num_test_examples(self):
        return 10000


datasource = DataSource('data').trainset

# Creating data indices for training and validation splits:
dataset_size = len(datasource)

np.random.seed(0)
indices = list(range(dataset_size))
np.random.shuffle(indices)

partition_size = 10000
total_clients = 8
total_size = partition_size * total_clients
# add extra samples to make it evenly divisible
indices += indices[:(total_size - len(indices))]
assert len(indices) == total_size
client_id = 2
subset_indices = indices[(client_id - 1):total_size:total_clients]
print(len(indices))
print(indices[:10])
print(len(subset_indices))
print(subset_indices[:10])

gen = torch.Generator()
gen.manual_seed(10)

train_sampler = SubsetRandomSampler(subset_indices, generator=gen)

train_loader = torch.utils.data.DataLoader(datasource,
                                           batch_size=16,
                                           sampler=train_sampler)

num_epochs = 10
for epoch in range(num_epochs):
    for batch_index, (_, labels) in enumerate(train_loader):
        print(labels)
