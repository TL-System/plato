"""
The convolutional neural network model for the MNIST dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import base


class Model(base.Model):
    '''A convolutional neural network model for MNIST.'''

    def __init__(self):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)


    def forward(self, x):
        """Defining the convolutional neural network."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


    @staticmethod
    def is_valid_model_name(model_name):
        return model_name.startswith('mnist_cnn')


    @staticmethod
    def get_model_from_name(model_name):
        """Obtaining an instance of this model provided that the name is valid."""

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        return Model()


    @property
    def loss_criterion(self):
        return self.criterion
