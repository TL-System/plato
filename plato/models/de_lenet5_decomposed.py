import collections
import math

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch
from torch.nn.modules.utils import _pair
from torch.nn import init


class Model(nn.Module):
    """The LeNet-5 model with decomposed layers for disjoint learning.
    Arguments:
        num_classes (int): The number of classes. Default: 10.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # We pad the image to get an input size of 32x32 as for the
        # original network in the LeCun paper
        self.conv1 = Decomposed_Conv2d(in_channels=1,
                                       out_channels=6,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2,
                                       bias=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Decomposed_Conv2d(in_channels=6,
                                       out_channels=16,
                                       kernel_size=5,
                                       stride=1,
                                       padding=0,
                                       bias=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = Decomposed_Conv2d(in_channels=16,
                                       out_channels=120,
                                       kernel_size=5,
                                       bias=True)
        self.relu3 = nn.ReLU()
        self.fc4 = Decomposed_Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc5 = Decomposed_Linear(84, num_classes)

        # Preparing named layers so that the model can be split and straddle
        # across the client and the server
        self.layers = []
        self.layerdict = collections.OrderedDict()
        self.layerdict['conv1'] = self.conv1
        self.layerdict['relu1'] = self.relu1
        self.layerdict['pool1'] = self.pool1
        self.layerdict['conv2'] = self.conv2
        self.layerdict['relu2'] = self.relu2
        self.layerdict['pool2'] = self.pool2
        self.layerdict['conv3'] = self.conv3
        self.layerdict['relu3'] = self.relu3
        self.layerdict['flatten'] = self.flatten
        self.layerdict['fc4'] = self.fc4
        self.layerdict['relu4'] = self.relu4
        self.layerdict['fc5'] = self.fc5
        self.layers.append('conv1')
        self.layers.append('relu1')
        self.layers.append('pool1')
        self.layers.append('conv2')
        self.layers.append('relu2')
        self.layers.append('pool2')
        self.layers.append('conv3')
        self.layers.append('relu3')
        self.layers.append('flatten')
        self.layers.append('fc4')
        self.layers.append('relu4')
        self.layers.append('fc5')

    def flatten(self, x):
        """Flatten the tensor."""
        return x.view(x.size(0), -1)

    def forward(self, x):
        """Forward pass."""

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc4(x)
        x = self.relu4(x)
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
    def get_model(*args):
        """Obtaining an instance of this model."""
        return Model()

    def get_psi(self):
        # extract psi from each decomposed layer as variables and return a list of them
        self.psi = []
        for key in self.layerdict:
            if ('pool' not in key) & ('flatten' not in key) & ('relu'
                                                               not in key):
                self.psi.append(self.layerdict[key].psi)

        return self.psi

    def get_sigma(self):
        # extract psi from each decomposed layer as variables and return a list of them
        self.sigma = []
        for key in self.layerdict:
            if ('pool' not in key) & ('flatten' not in key) & ('relu'
                                                               not in key):
                self.sigma.append(self.layerdict[key].sigma)
        return self.sigma


class Decomposed_Linear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 device=None,
                 dtype=None) -> None:

        super().__init__(in_features, out_features, bias, device, dtype)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.sigma = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs))
        self.psi = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs))

        init.kaiming_uniform_(self.sigma, a=math.sqrt(5))
        init.kaiming_uniform_(self.psi, a=math.sqrt(5))

    def forward(self, input):

        self.weight = Parameter(torch.add(self.sigma, self.psi))

        return F.linear(input, self.sigma, self.bias)


class Decomposed_Conv2d(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode, device,
                         dtype)

        kernel_size = _pair(kernel_size)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.sigma = Parameter(
            torch.empty((out_channels, in_channels, *kernel_size),
                        **factory_kwargs))
        self.psi = Parameter(
            torch.empty((out_channels, in_channels, *kernel_size),
                        **factory_kwargs))

        init.kaiming_uniform_(self.sigma, a=math.sqrt(5))

        init.kaiming_uniform_(self.psi, a=math.sqrt(5))

    def forward(self, input):

        self.weight = Parameter(torch.add(self.sigma, self.psi))

        return super()._conv_forward(input, self.sigma, self.bias)
