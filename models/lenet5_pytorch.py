"""
The convolutional neural network model for the MNIST dataset.
"""
import collections
import torch.nn as nn
import torch.nn.functional as F

from models import base


class Model(base.Model):
    '''The LeNet-5 model.

    Reference:

    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to
    document recognition." Proceedings of the IEEE, November 1998.

    Arguments:
        num_classes (int): The number of classes. Default: 10.
        dropout: The dropout ratio for the dropout layer.
    '''
    def __init__(self, num_classes=10, dropout=0.0):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()

        # We pad the image to get an input size of 32x32 as for the
        # original network in the LeCunn paper
        self.conv1 = nn.Conv2d(1, 6, 5, padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, 5, bias=True)
        self.bn3 = nn.BatchNorm2d(120)
        self.relu3 = nn.ReLU()
        self.flatten = lambda x: x.view(x.shape[0], -1)
        self.fc4 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc5 = nn.Linear(84, num_classes)

        # Preparing named layers so that the model can be split and straddle
        # across the client and the server
        self.layers = []
        self.layerdict = collections.OrderedDict()
        self.layerdict['conv1'] = self.conv1
        self.layerdict['bn1'] = self.bn1
        self.layerdict['relu1'] = self.relu1
        self.layerdict['pool1'] = self.pool1
        self.layerdict['conv2'] = self.conv2
        self.layerdict['bn2'] = self.bn2
        self.layerdict['relu2'] = self.relu2
        self.layerdict['pool2'] = self.pool2
        self.layerdict['conv3'] = self.conv3
        self.layerdict['bn3'] = self.bn3
        self.layerdict['relu3'] = self.relu3
        self.layerdict['flatten'] = self.flatten
        self.layerdict['fc4'] = self.fc4
        self.layerdict['bn4'] = self.bn4
        self.layerdict['relu4'] = self.relu4
        self.layerdict['dropout'] = self.dropout
        self.layerdict['fc5'] = self.fc5
        self.layers.append('conv1')
        self.layers.append('bn1')
        self.layers.append('relu1')
        self.layers.append('pool1')
        self.layers.append('conv2')
        self.layers.append('bn2')
        self.layers.append('relu2')
        self.layers.append('pool2')
        self.layers.append('conv3')
        self.layers.append('bn3')
        self.layers.append('relu3')
        self.layers.append('flatten')
        self.layers.append('fc4')
        self.layers.append('bn4')
        self.layers.append('relu4')
        self.layers.append('dropout')
        self.layers.append('fc5')

    def forward(self, x):
        """Forward pass."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc5(x)

        return F.log_softmax(x, dim=1)

    def forward_to(self, x, cut_layer):
        """Forward pass, but only to the layer specified by cut_layer."""
        layer_index = self.layers.index(cut_layer)
        for i in range(0, layer_index + 1):
            x = self.layerdict[self.layers[i]](x)
        return x

    def forward_from(self, x, cut_layer):
        """Forward pass, starting from the layer specified by cut_layer."""
        layer_index = self.layers.index(cut_layer)
        for i in range(layer_index + 1, len(self.layers)):
            x = self.layerdict[self.layers[i]](x)

        return F.log_softmax(x, dim=1)

    @staticmethod
    def is_valid_model_name(model_name):
        return model_name.startswith('lenet5_pytorch')

    @staticmethod
    def get_model_from_name(model_name):
        """Obtaining an instance of this model provided that the name is valid."""

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        return Model()

    @property
    def loss_criterion(self):
        return self.criterion
