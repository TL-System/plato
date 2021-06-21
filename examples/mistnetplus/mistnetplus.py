import os

import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from plato.datasources import base
from plato.trainers import 

class DataSource(base.DataSource):
    """A custom datasource with custom training and validation
       datasets.
    """
    def __init__(self):
        super().__init__()

        self.trainset = MNIST("./data",
                              train=True,
                              download=True,
                              transform=ToTensor())
        self.testset = MNIST("./data",
                             train=False,
                             download=True,
                             transform=ToTensor())






