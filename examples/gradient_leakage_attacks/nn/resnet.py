"""
The ResNet model used in Geiping's implementation.

Reference:
Geiping et al., "Inverting Gradients - How easy is it to break privacy in federated learning?,"
in the Proceedings of NeurIPS 2020.
https://github.com/JonasGeiping/invertinggradients
"""

import random

import numpy as np
import torch
import torchvision
from torch import nn

from plato.config import Config


def set_random_seed(seed=233):
    """233 = 144 + 89 is my favorite number."""
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)


class Model(torchvision.models.ResNet):
    """ResNet generalization for CIFAR thingies."""

    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        base_width=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        strides=[1, 2, 2, 2],
        pool="avg",
    ):
        """Initialize as usual. Layers and strides are scriptable."""
        super(torchvision.models.ResNet, self).__init__()  # nn.Module
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False, False]
        if len(replace_stride_with_dilation) != 4:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 4-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups

        self.inplanes = base_width
        self.base_width = 64  # Do this to circumvent BasicBlock errors. The value is not actually used.
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layers = torch.nn.ModuleList()
        width = self.inplanes
        for idx, layer in enumerate(layers):
            self.layers.append(
                self._make_layer(
                    block,
                    width,
                    layer,
                    stride=strides[idx],
                    dilate=replace_stride_with_dilation[idx],
                )
            )
            width *= 2

        self.pool = (
            nn.AdaptiveAvgPool2d((1, 1))
            if pool == "avg"
            else nn.AdaptiveMaxPool2d((1, 1))
        )
        self.fc = nn.Linear(width // 2 * block.expansion, num_classes)

        # TODO: move initialization?
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, torchvision.models.resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, torchvision.models.resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _forward_impl(self, x):
        """Model forwarding."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for layer in self.layers:
            x = layer(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward_feature(self, x):
        """Model forwarding returning intermediate feature."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for layer in self.layers:
            x = layer(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        feature = x.clone()
        x = self.fc(x)

        return x, feature

    @staticmethod
    def is_valid_model_type(model_type):
        return (
            model_type.startswith("resnet_")
            and len(model_type.split("_")) == 2
            and int(model_type.split("_")[1]) in [18, 34, 50, 101, 152, 32]
        )

    @staticmethod
    def resnet18():
        return Model(
            torchvision.models.resnet.BasicBlock,
            [2, 2, 2, 2],
            Config().parameters.model.num_classes,
            base_width=64,
        )

    @staticmethod
    def resnet32():
        return Model(
            torchvision.models.resnet.BasicBlock,
            [5, 5, 5],
            Config().parameters.model.num_classes,
            base_width=16 * 10,
        )

    @staticmethod
    def resnet34():
        return Model(
            torchvision.models.resnet.BasicBlock,
            [3, 4, 6, 3],
            Config().parameters.model.num_classes,
            base_width=64,
        )

    @staticmethod
    def resnet50():
        return Model(
            torchvision.models.resnet.Bottleneck,
            [3, 4, 6, 3],
            Config().parameters.model.num_classes,
            base_width=64,
        )

    @staticmethod
    def resnet101():
        return Model(
            torchvision.models.resnet.Bottleneck,
            [3, 4, 23, 3],
            Config().parameters.model.num_classes,
            base_width=64,
        )

    @staticmethod
    def resnet152():
        return Model(
            torchvision.models.resnet.Bottleneck,
            [3, 8, 36, 3],
            Config().parameters.model.num_classes,
            base_width=64,
        )


def get(model_name=None):
    """Returns a suitable ResNet model according to its type."""
    set_random_seed(1)

    if not Model.is_valid_model_type(model_name):
        raise ValueError(f"Invalid Resnet model name: {model_name}")

    resnet_type = int(model_name.split("_")[1])

    if resnet_type == 18:
        return Model.resnet18
    elif resnet_type == 32:
        return Model.resnet32
    elif resnet_type == 34:
        return Model.resnet34
    elif resnet_type == 50:
        return Model.resnet50
    elif resnet_type == 101:
        return Model.resnet101
    elif resnet_type == 152:
        return Model.resnet152
    return None
