import collections
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from torch.nn import init


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


class Decomposed_Model(nn.Module):
    def __init__(self, given_layer_dict) -> None:
        super().__init__()
        print("Starting initialize model")

        self.layers = []
        self.layerdict = collections.OrderedDict()

        for item in given_layer_dict:
            if "conv" in item:  # conv -> decomposed_conv
                self.layerdict[item] = Decomposed_Conv2d(
                    in_channels=given_layer_dict[item].in_channels,
                    out_channels=given_layer_dict[item].out_channels,
                    kernel_size=given_layer_dict[item].kernel_size,
                    stride=given_layer_dict[item].stride,
                    padding=given_layer_dict[item].padding,
                    bias=True)  #given_layer_dict[item].bias)
            elif "fc" in item:  # liner -> decomposed_linear
                self.layerdict[item] = Decomposed_Linear(
                    in_features=given_layer_dict[item].in_features,
                    out_features=given_layer_dict[item].out_features,
                    bias=True)  #given_layer_dict[item].bias)
            else:  # relu & pool & flatten -> unchanged
                self.layerdict[item] = given_layer_dict[item]
            setattr(self, item, self.layerdict[item])

            self.layers.append(item)
        print("Finished decomposition trans")

    def forward(self, x):
        """Forward pass."""
        for layer_name in self.layerdict:
            x = self.layerdict[layer_name](x)

        return F.log_softmax(x, dim=1)

    def forward_to(self, x, cut_layer):
        """Forward pass, but only to the layer specified by cut_layer."""
        for layer_name in self.layerdict:
            x = self.layerdict[layer_name](x)

            if layer_name == cut_layer:
                break
        return x

    def forward_from(self, x, cut_layer):
        """Forward pass, starting from the layer specified by cut_layer."""
        flag = False
        for layer_name in self.layerdict:
            if flag:
                x = self.layerdict[layer_name](x)
            if layer_name == cut_layer:
                flag = True

        return F.log_softmax(x, dim=1)

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

    @staticmethod
    def get_decomposed_model(given_model):
        """Obtaining an instance of this model."""
        given_layer_dict = given_model.layerdict
        print("getting model ...")

        return Decomposed_Model(given_layer_dict)


#model = lenet5.Model()
#test = DecomposedModel.get_model(model)
